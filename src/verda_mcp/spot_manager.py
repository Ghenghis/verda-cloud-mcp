"""Spot Instance Manager for Verda MCP Server.

Handles:
- Auto-switching between Spot and On-Demand instances
- Spot eviction detection and recovery
- Checkpoint-aware deployment recommendations
- Fail-safe mechanisms for seamless training continuity
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class InstanceMode(Enum):
    """Instance pricing mode."""
    SPOT = "spot"
    ON_DEMAND = "on_demand"


class SpotStatus(Enum):
    """Spot instance status."""
    RUNNING = "running"
    EVICTED = "evicted"
    UNAVAILABLE = "unavailable"
    SWITCHING = "switching"


# =============================================================================
# COMPLETE GPU PRICING WITH SPOT RATES
# Spot typically 25-35% of on-demand (estimated based on market)
# =============================================================================
GPU_PRICING_FULL = {
    # NVLink GPUs
    "GB300": {
        "on_demand": 5.45, "spot": 1.36, "vram_gb": 288, "multi_gpu": [1, 2, 4],
        "spot_savings": "75%", "spot_available": True
    },
    "B300": {
        "on_demand": 4.95, "spot": 1.24, "vram_gb": 262, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "B200": {
        "on_demand": 3.79, "spot": 0.95, "vram_gb": 180, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "H200": {
        "on_demand": 2.99, "spot": 0.75, "vram_gb": 141, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "H100": {
        "on_demand": 2.29, "spot": 0.57, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "A100_80G": {
        "on_demand": 1.29, "spot": 0.32, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "A100_40G": {
        "on_demand": 0.7211, "spot": 0.18, "vram_gb": 40, "multi_gpu": [1, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "V100": {
        "on_demand": 0.1381, "spot": 0.035, "vram_gb": 16, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    # General Compute
    "RTX_PRO_6000": {
        "on_demand": 1.39, "spot": 0.35, "vram_gb": 96, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "L40S": {
        "on_demand": 0.9143, "spot": 0.23, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "RTX_6000_ADA": {
        "on_demand": 0.8262, "spot": 0.21, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "75%", "spot_available": True
    },
    "A6000": {
        "on_demand": 0.49, "spot": 0.12, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8],
        "spot_savings": "76%", "spot_available": True
    },
}


@dataclass
class SpotDeploymentConfig:
    """Configuration for spot deployment with failover."""
    gpu_type: str
    gpu_count: int
    prefer_spot: bool = True
    auto_failover: bool = True  # Switch to on-demand if spot evicted
    checkpoint_interval_minutes: int = 10  # CRITICAL for spot!
    volume_id: Optional[str] = None
    script_id: Optional[str] = None
    max_spot_retries: int = 3
    fallback_locations: List[str] = field(default_factory=lambda: ["FIN-01", "FIN-02", "FIN-03"])


@dataclass
class SpotSession:
    """Active spot training session."""
    instance_id: str
    instance_ip: Optional[str]
    gpu_type: str
    gpu_count: int
    mode: InstanceMode
    started_at: datetime
    last_checkpoint: Optional[datetime] = None
    checkpoint_count: int = 0
    eviction_count: int = 0
    total_cost: float = 0.0
    is_active: bool = True


class SpotManager:
    """Manages spot instances with auto-failover to on-demand."""
    
    def __init__(self):
        self.active_session: Optional[SpotSession] = None
        self.eviction_history: List[Dict[str, Any]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval: int = 60  # Check every 60 seconds
    
    def get_pricing(self, gpu_type: str) -> Dict[str, Any]:
        """Get full pricing info for a GPU type."""
        gpu_key = gpu_type.upper().replace("-", "_")
        return GPU_PRICING_FULL.get(gpu_key, {
            "on_demand": 0, "spot": 0, "vram_gb": 0, 
            "multi_gpu": [1], "spot_savings": "0%", "spot_available": False
        })
    
    def calculate_savings(self, gpu_type: str, gpu_count: int, hours: float) -> Dict[str, Any]:
        """Calculate savings using spot vs on-demand."""
        pricing = self.get_pricing(gpu_type)
        
        on_demand_cost = pricing["on_demand"] * gpu_count * hours
        spot_cost = pricing["spot"] * gpu_count * hours
        savings = on_demand_cost - spot_cost
        savings_pct = (savings / on_demand_cost * 100) if on_demand_cost > 0 else 0
        
        return {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "hours": hours,
            "on_demand_hourly": pricing["on_demand"] * gpu_count,
            "spot_hourly": pricing["spot"] * gpu_count,
            "on_demand_total": on_demand_cost,
            "spot_total": spot_cost,
            "savings": savings,
            "savings_percent": savings_pct,
            "recommendation": "SPOT" if pricing.get("spot_available", False) else "ON_DEMAND",
            "checkpoint_required": True,
            "checkpoint_interval_recommended": 10,  # minutes
        }
    
    async def check_spot_availability(self, gpu_type: str, gpu_count: int) -> Dict[str, Any]:
        """Check if spot is available for the requested GPU config."""
        try:
            from .client import get_client
            client = get_client()
            result = await client.check_spot_availability(gpu_type, gpu_count)
            
            return {
                "available": result.available,
                "location": result.location,
                "instance_type": result.instance_type,
                "mode": "spot" if result.available else "on_demand_fallback",
            }
        except Exception as e:
            logger.error(f"Spot availability check failed: {e}")
            return {
                "available": False,
                "location": None,
                "instance_type": None,
                "mode": "on_demand_fallback",
                "error": str(e),
            }
    
    async def smart_deploy(self, config: SpotDeploymentConfig) -> Dict[str, Any]:
        """Deploy with smart spot/on-demand selection and failover.
        
        Strategy:
        1. Try spot first (if prefer_spot=True)
        2. If spot unavailable, try other locations
        3. If all spot fails, fall back to on-demand
        4. Set up monitoring for eviction
        """
        from .client import get_client
        client = get_client()
        
        result = {
            "success": False,
            "mode": None,
            "instance_id": None,
            "instance_ip": None,
            "location": None,
            "message": "",
            "checkpoint_reminder": f"⚠️ CRITICAL: Save checkpoints every {config.checkpoint_interval_minutes} minutes!",
        }
        
        # Try spot first
        if config.prefer_spot:
            for location in config.fallback_locations:
                try:
                    # Check availability
                    avail = await client.check_spot_availability(
                        config.gpu_type, config.gpu_count, location
                    )
                    
                    if avail.available:
                        # Deploy spot instance
                        instance = await client.create_instance(
                            gpu_type=config.gpu_type,
                            gpu_count=config.gpu_count,
                            location=location,
                            volume_ids=[config.volume_id] if config.volume_id else None,
                            script_id=config.script_id,
                            is_spot=True,
                        )
                        
                        # Wait for ready
                        instance = await client.wait_for_ready(instance.id)
                        
                        # Create session
                        self.active_session = SpotSession(
                            instance_id=instance.id,
                            instance_ip=instance.ip_address,
                            gpu_type=config.gpu_type,
                            gpu_count=config.gpu_count,
                            mode=InstanceMode.SPOT,
                            started_at=datetime.now(),
                        )
                        
                        result.update({
                            "success": True,
                            "mode": "SPOT",
                            "instance_id": instance.id,
                            "instance_ip": instance.ip_address,
                            "location": location,
                            "message": f"✅ SPOT instance deployed! Saving {self.get_pricing(config.gpu_type)['spot_savings']}!",
                        })
                        
                        # Start monitoring if auto_failover enabled
                        if config.auto_failover:
                            await self.start_monitoring(config)
                        
                        return result
                        
                except Exception as e:
                    logger.warning(f"Spot deploy failed at {location}: {e}")
                    continue
        
        # Fallback to on-demand
        if config.auto_failover or not config.prefer_spot:
            try:
                instance = await client.create_instance(
                    gpu_type=config.gpu_type,
                    gpu_count=config.gpu_count,
                    location=config.fallback_locations[0],
                    volume_ids=[config.volume_id] if config.volume_id else None,
                    script_id=config.script_id,
                    is_spot=False,
                )
                
                instance = await client.wait_for_ready(instance.id)
                
                self.active_session = SpotSession(
                    instance_id=instance.id,
                    instance_ip=instance.ip_address,
                    gpu_type=config.gpu_type,
                    gpu_count=config.gpu_count,
                    mode=InstanceMode.ON_DEMAND,
                    started_at=datetime.now(),
                )
                
                result.update({
                    "success": True,
                    "mode": "ON_DEMAND",
                    "instance_id": instance.id,
                    "instance_ip": instance.ip_address,
                    "location": config.fallback_locations[0],
                    "message": "✅ On-demand instance deployed (spot unavailable).",
                })
                
                return result
                
            except Exception as e:
                result["message"] = f"❌ Deployment failed: {e}"
                return result
        
        result["message"] = "❌ No instances available."
        return result
    
    async def handle_eviction(self, config: SpotDeploymentConfig) -> Dict[str, Any]:
        """Handle spot eviction with automatic recovery.
        
        Steps:
        1. Log eviction event
        2. Try to get new spot instance
        3. If spot unavailable, switch to on-demand
        4. Resume training from last checkpoint
        """
        if not self.active_session:
            return {"success": False, "message": "No active session to recover"}
        
        # Record eviction
        eviction_record = {
            "timestamp": datetime.now().isoformat(),
            "instance_id": self.active_session.instance_id,
            "mode": self.active_session.mode.value,
            "duration_minutes": (datetime.now() - self.active_session.started_at).total_seconds() / 60,
        }
        self.eviction_history.append(eviction_record)
        self.active_session.eviction_count += 1
        
        logger.warning(f"⚠️ SPOT EVICTION DETECTED! Attempting recovery...")
        
        # Try to redeploy
        result = await self.smart_deploy(config)
        
        if result["success"]:
            result["recovery"] = {
                "eviction_count": self.active_session.eviction_count,
                "recovered": True,
                "resume_from": "Last checkpoint (should be within 10 minutes)",
            }
            result["message"] = f"🔄 RECOVERED from eviction! Mode: {result['mode']}"
        
        return result
    
    async def start_monitoring(self, config: SpotDeploymentConfig):
        """Start background monitoring for spot eviction."""
        if self._monitor_task and not self._monitor_task.done():
            return
        
        async def monitor_loop():
            while self.active_session and self.active_session.is_active:
                try:
                    from .client import get_client
                    client = get_client()
                    
                    # Check instance status
                    instance = await client.get_instance(self.active_session.instance_id)
                    
                    if instance.status in ("terminated", "stopped", "error"):
                        logger.warning(f"Instance status: {instance.status} - possible eviction!")
                        await self.handle_eviction(config)
                        break
                    
                    await asyncio.sleep(self._monitor_interval)
                    
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    await asyncio.sleep(self._monitor_interval)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("Spot eviction monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self.active_session:
            self.active_session.is_active = False
        logger.info("Spot eviction monitoring stopped")
    
    async def switch_mode(self, to_mode: InstanceMode) -> Dict[str, Any]:
        """Manually switch between spot and on-demand.
        
        IMPORTANT: This will:
        1. Create new instance in target mode
        2. Wait for it to be ready
        3. User should resume training from checkpoint
        4. Delete old instance (optional)
        """
        if not self.active_session:
            return {"success": False, "message": "No active session"}
        
        current_mode = self.active_session.mode
        if current_mode == to_mode:
            return {"success": True, "message": f"Already in {to_mode.value} mode"}
        
        config = SpotDeploymentConfig(
            gpu_type=self.active_session.gpu_type,
            gpu_count=self.active_session.gpu_count,
            prefer_spot=(to_mode == InstanceMode.SPOT),
            auto_failover=False,
        )
        
        old_instance_id = self.active_session.instance_id
        
        # Deploy new instance
        result = await self.smart_deploy(config)
        
        if result["success"]:
            result["switched_from"] = current_mode.value
            result["old_instance_id"] = old_instance_id
            result["message"] = f"🔄 Switched from {current_mode.value} to {to_mode.value}!"
            result["action_required"] = "Resume training from last checkpoint"
        
        return result
    
    def format_savings_report(self, savings: Dict[str, Any]) -> str:
        """Format savings calculation for display."""
        return f"""# 💰 Spot vs On-Demand Comparison

## Configuration
- **GPU**: {savings['gpu_type']} x{savings['gpu_count']}
- **Duration**: {savings['hours']} hours

## Pricing Comparison
| Mode | Hourly Rate | Total Cost |
|------|-------------|------------|
| **SPOT** | ${savings['spot_hourly']:.2f}/hr | **${savings['spot_total']:.2f}** |
| On-Demand | ${savings['on_demand_hourly']:.2f}/hr | ${savings['on_demand_total']:.2f} |

## 💵 SAVINGS WITH SPOT
- **You Save**: ${savings['savings']:.2f} ({savings['savings_percent']:.0f}%)
- **Recommendation**: **{savings['recommendation']}**

## ⚠️ SPOT REQUIREMENTS
- ✅ **Checkpoints every {savings['checkpoint_interval_recommended']} minutes** (CRITICAL!)
- ✅ Training script must support checkpoint resume
- ✅ Volume attached for persistent storage
- ⚠️ Spot can be evicted at ANY time
- ⚠️ Auto-failover to on-demand available
"""
    
    def format_session_status(self) -> str:
        """Format current session status."""
        if not self.active_session:
            return "# 📊 No Active Session\n\nNo training session is currently running."
        
        s = self.active_session
        duration = datetime.now() - s.started_at
        hours = duration.total_seconds() / 3600
        pricing = self.get_pricing(s.gpu_type)
        rate = pricing["spot"] if s.mode == InstanceMode.SPOT else pricing["on_demand"]
        cost = rate * s.gpu_count * hours
        
        return f"""# 📊 Active Training Session

## Instance Info
- **Instance ID**: {s.instance_id}
- **IP Address**: {s.instance_ip}
- **GPU**: {s.gpu_type} x{s.gpu_count}
- **Mode**: **{s.mode.value.upper()}**

## Session Stats
- **Started**: {s.started_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {hours:.2f} hours
- **Checkpoints**: {s.checkpoint_count}
- **Evictions**: {s.eviction_count}

## Cost
- **Rate**: ${rate * s.gpu_count:.2f}/hr
- **Current Cost**: ${cost:.2f}

## Status
- **Active**: {'✅ Yes' if s.is_active else '❌ No'}
- **Monitoring**: {'✅ Enabled' if self._monitor_task else '❌ Disabled'}
"""


# Global spot manager instance
_spot_manager: Optional[SpotManager] = None


def get_spot_manager() -> SpotManager:
    """Get the global spot manager."""
    global _spot_manager
    if _spot_manager is None:
        _spot_manager = SpotManager()
    return _spot_manager


# =============================================================================
# Async wrapper functions for MCP tools
# =============================================================================

async def compare_spot_vs_ondemand(gpu_type: str, gpu_count: int, hours: float) -> str:
    """Compare spot vs on-demand pricing."""
    manager = get_spot_manager()
    savings = manager.calculate_savings(gpu_type, gpu_count, hours)
    return manager.format_savings_report(savings)


async def smart_deploy_instance(
    gpu_type: str,
    gpu_count: int = 1,
    prefer_spot: bool = True,
    auto_failover: bool = True,
    checkpoint_minutes: int = 10,
    volume_id: str = "",
    script_id: str = "",
) -> str:
    """Smart deploy with spot preference and auto-failover."""
    manager = get_spot_manager()
    
    config = SpotDeploymentConfig(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        prefer_spot=prefer_spot,
        auto_failover=auto_failover,
        checkpoint_interval_minutes=checkpoint_minutes,
        volume_id=volume_id if volume_id else None,
        script_id=script_id if script_id else None,
    )
    
    result = await manager.smart_deploy(config)
    
    lines = [
        f"# {'✅' if result['success'] else '❌'} Deployment Result",
        "",
        f"**Status**: {'Success' if result['success'] else 'Failed'}",
        f"**Mode**: {result.get('mode', 'N/A')}",
        f"**Instance ID**: {result.get('instance_id', 'N/A')}",
        f"**IP Address**: {result.get('instance_ip', 'N/A')}",
        f"**Location**: {result.get('location', 'N/A')}",
        "",
        f"**Message**: {result['message']}",
        "",
        result.get('checkpoint_reminder', ''),
    ]
    
    if result['success'] and result.get('mode') == 'SPOT':
        pricing = manager.get_pricing(gpu_type)
        lines.extend([
            "",
            f"## 💰 You're Saving {pricing['spot_savings']}!",
            f"- Spot Rate: ${pricing['spot'] * gpu_count:.2f}/hr",
            f"- On-Demand would be: ${pricing['on_demand'] * gpu_count:.2f}/hr",
        ])
    
    return "\n".join(lines)


async def switch_instance_mode(to_spot: bool = True) -> str:
    """Switch between spot and on-demand."""
    manager = get_spot_manager()
    target_mode = InstanceMode.SPOT if to_spot else InstanceMode.ON_DEMAND
    result = await manager.switch_mode(target_mode)
    
    if result['success']:
        return f"""# 🔄 Mode Switch Complete

**Switched to**: {result.get('mode', target_mode.value)}
**New Instance ID**: {result.get('instance_id', 'N/A')}
**New IP**: {result.get('instance_ip', 'N/A')}

## ⚠️ Action Required
{result.get('action_required', 'Resume training from last checkpoint')}

**Old Instance**: {result.get('old_instance_id', 'N/A')}
(You may want to delete the old instance after confirming new one works)
"""
    else:
        return f"# ❌ Switch Failed\n\n{result.get('message', 'Unknown error')}"


async def get_session_status() -> str:
    """Get current training session status."""
    manager = get_spot_manager()
    return manager.format_session_status()


async def stop_session_monitoring() -> str:
    """Stop monitoring and mark session as complete."""
    manager = get_spot_manager()
    manager.stop_monitoring()
    return "✅ Session monitoring stopped. Training session marked as complete."
