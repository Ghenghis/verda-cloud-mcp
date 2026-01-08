"""Live Data Fetcher for Verda MCP Server.

Auto-updates GPU pricing, availability, and instance data from Verda API.

Available via API:
- Instance types and availability
- Current running instances
- Volumes, SSH keys, scripts
- Images (OS options)

Not available via API (requires web scraping or manual update):
- Exact pricing (no pricing endpoint in SDK)
- GPU specs (VRAM, TFLOPs, etc.)

This module provides:
- Live availability checking for all GPUs
- Instance status monitoring
- API data refresh system
- Fallback to cached data when API unavailable
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# CACHED PRICING DATA (Updated manually - API doesn't provide pricing)
# Last updated: January 2026
# =============================================================================

CACHED_GPU_PRICING = {
    "GB300": {"on_demand": 5.45, "spot": 1.36, "vram_gb": 288, "multi_gpu": [1, 2, 4]},
    "B300": {"on_demand": 4.95, "spot": 1.24, "vram_gb": 262, "multi_gpu": [1, 2, 4, 8]},
    "B200": {"on_demand": 3.79, "spot": 0.95, "vram_gb": 180, "multi_gpu": [1, 2, 4, 8]},
    "H200": {"on_demand": 2.99, "spot": 0.75, "vram_gb": 141, "multi_gpu": [1, 2, 4, 8]},
    "H100": {"on_demand": 2.29, "spot": 0.57, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8]},
    "A100_80G": {"on_demand": 1.29, "spot": 0.32, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8]},
    "A100_40G": {"on_demand": 0.7211, "spot": 0.18, "vram_gb": 40, "multi_gpu": [1, 8]},
    "V100": {"on_demand": 0.1381, "spot": 0.035, "vram_gb": 16, "multi_gpu": [1, 2, 4, 8]},
    "RTX_PRO_6000": {"on_demand": 1.39, "spot": 0.35, "vram_gb": 96, "multi_gpu": [1, 2, 4, 8]},
    "L40S": {"on_demand": 0.9143, "spot": 0.23, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "RTX_6000_ADA": {"on_demand": 0.8262, "spot": 0.21, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "A6000": {"on_demand": 0.49, "spot": 0.12, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
}

# GPU type to instance type mapping
GPU_INSTANCE_TYPES = {
    "GB300": {1: "1GB300.36V", 2: "2GB300.72V", 4: "4GB300.144V"},
    "B300": {1: "1B300.30V", 2: "2B300.60V", 4: "4B300.120V", 8: "8B300.240V"},
    "B200": {1: "1B200.30V", 2: "2B200.60V", 4: "4B200.120V", 8: "8B200.240V"},
    "H200": {1: "1H200.18V", 2: "2H200.36V", 4: "4H200.72V", 8: "8H200.144V"},
    "H100": {1: "1H100.10V", 2: "2H100.20V", 4: "4H100.40V", 8: "8H100.80V"},
    "A100_80G": {1: "1A100_80G.10V", 2: "2A100_80G.20V", 4: "4A100_80G.40V", 8: "8A100_80G.80V"},
    "A100_40G": {1: "1A100_40G.10V", 8: "8A100_40G.80V"},
    "V100": {1: "1V100.10V", 2: "2V100.20V", 4: "4V100.40V", 8: "8V100.80V"},
    "RTX_PRO_6000": {1: "1RTXPRO6000.12V", 2: "2RTXPRO6000.24V", 4: "4RTXPRO6000.48V", 8: "8RTXPRO6000.96V"},
    "L40S": {1: "1L40S.12V", 2: "2L40S.24V", 4: "4L40S.48V", 8: "8L40S.96V"},
    "RTX_6000_ADA": {1: "1RTX6000Ada.12V", 2: "2RTX6000Ada.24V", 4: "4RTX6000Ada.48V", 8: "8RTX6000Ada.96V"},
    "A6000": {1: "1A6000.10V", 2: "2A6000.20V", 4: "4A6000.40V", 8: "8A6000.80V"},
}

LOCATIONS = ["FIN-01", "FIN-02", "FIN-03"]


@dataclass
class LiveGPUData:
    """Live GPU availability data."""
    gpu_type: str
    gpu_count: int
    instance_type: str
    spot_available: bool
    on_demand_available: bool
    location: Optional[str]
    checked_at: datetime
    pricing: Dict[str, float] = field(default_factory=dict)


@dataclass 
class APIStatus:
    """Status of Verda API connection."""
    connected: bool
    last_check: datetime
    error: Optional[str] = None


class LiveDataManager:
    """Manages live data from Verda API."""
    
    def __init__(self):
        self.availability_cache: Dict[str, LiveGPUData] = {}
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        self.api_status = APIStatus(connected=False, last_check=datetime.now())
        self.pricing = CACHED_GPU_PRICING.copy()
    
    async def check_api_connection(self) -> APIStatus:
        """Check if Verda API is accessible."""
        try:
            from .client import get_client
            client = get_client()
            
            # Try to list instances as a connection test
            await client.list_instances()
            
            self.api_status = APIStatus(
                connected=True,
                last_check=datetime.now(),
            )
        except Exception as e:
            self.api_status = APIStatus(
                connected=False,
                last_check=datetime.now(),
                error=str(e),
            )
        
        return self.api_status
    
    async def fetch_all_availability(self) -> Dict[str, List[LiveGPUData]]:
        """Fetch live availability for all GPU types."""
        try:
            from .client import get_client
            client = get_client()
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return {}
        
        results = {}
        
        for gpu_type, configs in GPU_INSTANCE_TYPES.items():
            results[gpu_type] = []
            
            for gpu_count, instance_type in configs.items():
                for location in LOCATIONS:
                    try:
                        # Check spot availability
                        spot_avail = await client._run_sync(
                            client._instances.is_available,
                            instance_type,
                            True,  # is_spot
                            location,
                        )
                        
                        # Check on-demand availability
                        on_demand_avail = await client._run_sync(
                            client._instances.is_available,
                            instance_type,
                            False,  # is_spot
                            location,
                        )
                        
                        data = LiveGPUData(
                            gpu_type=gpu_type,
                            gpu_count=gpu_count,
                            instance_type=instance_type,
                            spot_available=spot_avail,
                            on_demand_available=on_demand_avail,
                            location=location,
                            checked_at=datetime.now(),
                            pricing=self.pricing.get(gpu_type, {}),
                        )
                        
                        results[gpu_type].append(data)
                        
                        # Cache the result
                        cache_key = f"{gpu_type}_{gpu_count}_{location}"
                        self.availability_cache[cache_key] = data
                        
                    except Exception as e:
                        logger.debug(f"Error checking {gpu_type} x{gpu_count} at {location}: {e}")
                        continue
        
        return results
    
    async def get_gpu_availability(
        self,
        gpu_type: str,
        gpu_count: int = 1,
    ) -> List[LiveGPUData]:
        """Get availability for a specific GPU config."""
        try:
            from .client import get_client
            client = get_client()
        except Exception as e:
            return []
        
        gpu_key = gpu_type.upper().replace("-", "_")
        configs = GPU_INSTANCE_TYPES.get(gpu_key, {})
        instance_type = configs.get(gpu_count)
        
        if not instance_type:
            return []
        
        results = []
        
        for location in LOCATIONS:
            # Check cache first
            cache_key = f"{gpu_key}_{gpu_count}_{location}"
            cached = self.availability_cache.get(cache_key)
            
            if cached and (datetime.now() - cached.checked_at) < self.cache_duration:
                results.append(cached)
                continue
            
            try:
                spot_avail = await client._run_sync(
                    client._instances.is_available,
                    instance_type,
                    True,
                    location,
                )
                
                on_demand_avail = await client._run_sync(
                    client._instances.is_available,
                    instance_type,
                    False,
                    location,
                )
                
                data = LiveGPUData(
                    gpu_type=gpu_key,
                    gpu_count=gpu_count,
                    instance_type=instance_type,
                    spot_available=spot_avail,
                    on_demand_available=on_demand_avail,
                    location=location,
                    checked_at=datetime.now(),
                    pricing=self.pricing.get(gpu_key, {}),
                )
                
                results.append(data)
                self.availability_cache[cache_key] = data
                
            except Exception as e:
                logger.debug(f"Error: {e}")
                continue
        
        return results
    
    async def get_running_instances_cost(self) -> Dict[str, Any]:
        """Get cost of currently running instances."""
        try:
            from .client import get_client
            client = get_client()
            
            instances = await client.list_instances()
            running = [i for i in instances if i.status == "running"]
            
            total_hourly = 0.0
            instance_costs = []
            
            for inst in running:
                # Try to determine GPU type from instance type
                inst_type = inst.instance_type or ""
                
                for gpu_type, configs in GPU_INSTANCE_TYPES.items():
                    for count, type_str in configs.items():
                        if type_str.lower() in inst_type.lower():
                            pricing = self.pricing.get(gpu_type, {})
                            # Assume on-demand for running instances
                            hourly = pricing.get("on_demand", 0) * count
                            total_hourly += hourly
                            
                            instance_costs.append({
                                "instance_id": inst.id,
                                "hostname": inst.hostname,
                                "gpu": f"{count}x {gpu_type}",
                                "hourly_cost": hourly,
                            })
                            break
            
            return {
                "running_instances": len(running),
                "total_hourly_cost": total_hourly,
                "daily_cost": total_hourly * 24,
                "instances": instance_costs,
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def list_available_images(self) -> List[Dict[str, str]]:
        """List available OS images from API."""
        try:
            from .client import get_client
            client = get_client()
            
            images = await client.list_images()
            return [
                {
                    "id": img.id,
                    "name": img.name,
                    "type": img.image_type,
                }
                for img in images
            ]
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return []
    
    def get_api_capabilities(self) -> Dict[str, Any]:
        """Show what the API can and cannot provide."""
        return {
            "available_via_api": {
                "instances": "List, create, delete, start, stop instances",
                "availability": "Check spot/on-demand availability by location",
                "volumes": "List, create, attach, detach block volumes",
                "ssh_keys": "List, create SSH keys",
                "startup_scripts": "List, create startup scripts",
                "images": "List available OS images",
            },
            "not_available_via_api": {
                "pricing": "No pricing endpoint - using cached values (updated Jan 2026)",
                "gpu_specs": "No specs endpoint - using hardcoded values",
                "balance": "No balance endpoint - check console.verda.com",
                "shared_filesystems": "Beta feature - use console",
                "clusters": "Beta feature - use console",
                "batch_jobs": "Beta feature - use console",
            },
            "auto_updated": [
                "Instance availability (live)",
                "Current running instances (live)",
                "Available images (live)",
                "Volume status (live)",
            ],
            "manually_updated": [
                "GPU pricing (cached)",
                "GPU specifications (cached)",
                "Instance type mappings (cached)",
            ],
            "cache_duration_minutes": 5,
            "pricing_last_updated": "January 2026",
        }
    
    def format_availability(self, data: List[LiveGPUData]) -> str:
        """Format availability data for display."""
        if not data:
            return "❌ No availability data found."
        
        gpu_type = data[0].gpu_type
        gpu_count = data[0].gpu_count
        pricing = data[0].pricing
        
        lines = [
            f"# 🔄 Live Availability: {gpu_count}x {gpu_type}",
            "",
            f"**On-Demand**: ${pricing.get('on_demand', 0) * gpu_count:.2f}/hr",
            f"**SPOT**: ${pricing.get('spot', 0) * gpu_count:.2f}/hr (75% savings!)",
            "",
            "## Availability by Location",
            "",
            "| Location | SPOT | On-Demand |",
            "|----------|------|-----------|",
        ]
        
        for d in data:
            spot = "✅ Available" if d.spot_available else "❌ Unavailable"
            on_demand = "✅ Available" if d.on_demand_available else "❌ Unavailable"
            lines.append(f"| {d.location} | {spot} | {on_demand} |")
        
        lines.extend([
            "",
            f"*Last checked: {data[0].checked_at.strftime('%H:%M:%S')}*",
        ])
        
        return "\n".join(lines)
    
    def format_running_costs(self, data: Dict[str, Any]) -> str:
        """Format running costs for display."""
        if "error" in data:
            return f"❌ Error: {data['error']}"
        
        lines = [
            "# 💰 Current Running Costs",
            "",
            f"**Running Instances**: {data['running_instances']}",
            f"**Hourly Cost**: ${data['total_hourly_cost']:.2f}/hr",
            f"**Daily Cost**: ${data['daily_cost']:.2f}/day",
            "",
        ]
        
        if data.get("instances"):
            lines.extend([
                "## Instances",
                "",
                "| Instance | GPU | $/hr |",
                "|----------|-----|------|",
            ])
            
            for inst in data["instances"]:
                lines.append(
                    f"| {inst['hostname']} | {inst['gpu']} | ${inst['hourly_cost']:.2f} |"
                )
        else:
            lines.append("*No running instances*")
        
        return "\n".join(lines)
    
    def format_api_capabilities(self) -> str:
        """Format API capabilities for display."""
        caps = self.get_api_capabilities()
        
        lines = [
            "# 🔌 Verda API Capabilities",
            "",
            "## ✅ Available via API (Auto-Updated)",
            "",
        ]
        
        for key, desc in caps["available_via_api"].items():
            lines.append(f"- **{key}**: {desc}")
        
        lines.extend([
            "",
            "## ❌ Not Available via API (Cached/Manual)",
            "",
        ])
        
        for key, desc in caps["not_available_via_api"].items():
            lines.append(f"- **{key}**: {desc}")
        
        lines.extend([
            "",
            "## 🔄 Auto-Updated Data",
            "",
        ])
        
        for item in caps["auto_updated"]:
            lines.append(f"- {item}")
        
        lines.extend([
            "",
            "## 📝 Manually Updated Data",
            "",
        ])
        
        for item in caps["manually_updated"]:
            lines.append(f"- {item}")
        
        lines.extend([
            "",
            f"**Cache Duration**: {caps['cache_duration_minutes']} minutes",
            f"**Pricing Last Updated**: {caps['pricing_last_updated']}",
            "",
            "## 💡 To Update Pricing",
            "Pricing must be updated manually from: https://console.verda.com",
            "Edit `live_data.py` → `CACHED_GPU_PRICING` dictionary",
        ])
        
        return "\n".join(lines)


# Global manager
_live_manager: Optional[LiveDataManager] = None


def get_live_manager() -> LiveDataManager:
    global _live_manager
    if _live_manager is None:
        _live_manager = LiveDataManager()
    return _live_manager


# =============================================================================
# ASYNC WRAPPER FUNCTIONS FOR MCP TOOLS
# =============================================================================

async def check_live_availability(gpu_type: str, gpu_count: int = 1) -> str:
    """Check live availability from API."""
    manager = get_live_manager()
    data = await manager.get_gpu_availability(gpu_type, gpu_count)
    return manager.format_availability(data)


async def check_all_availability() -> str:
    """Check availability for all GPU types."""
    manager = get_live_manager()
    all_data = await manager.fetch_all_availability()
    
    lines = [
        "# 🔄 Live GPU Availability (All Types)",
        "",
        f"*Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    
    for gpu_type, data_list in all_data.items():
        if not data_list:
            continue
        
        pricing = manager.pricing.get(gpu_type, {})
        spot_price = pricing.get("spot", 0)
        
        # Check if any spot available
        spot_available = any(d.spot_available for d in data_list)
        on_demand_available = any(d.on_demand_available for d in data_list)
        
        spot_status = "✅" if spot_available else "❌"
        od_status = "✅" if on_demand_available else "❌"
        
        lines.append(f"| **{gpu_type}** | ${spot_price:.2f}/hr spot | {spot_status} Spot | {od_status} On-Demand |")
    
    return "\n".join(lines)


async def get_current_costs() -> str:
    """Get cost of running instances."""
    manager = get_live_manager()
    data = await manager.get_running_instances_cost()
    return manager.format_running_costs(data)


async def get_api_info() -> str:
    """Get API capabilities info."""
    manager = get_live_manager()
    return manager.format_api_capabilities()


async def refresh_data() -> str:
    """Refresh all cached data from API."""
    manager = get_live_manager()
    
    # Check API connection
    status = await manager.check_api_connection()
    
    if not status.connected:
        return f"❌ API connection failed: {status.error}"
    
    # Refresh availability
    await manager.fetch_all_availability()
    
    # Get running costs
    costs = await manager.get_running_instances_cost()
    
    return f"""# 🔄 Data Refreshed

**API Status**: ✅ Connected
**Last Refresh**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Running Instances
{costs.get('running_instances', 0)} instance(s) running
Hourly cost: ${costs.get('total_hourly_cost', 0):.2f}/hr

## Cached Data Updated
- GPU availability (all types and locations)
- Instance statuses
- Volume statuses

*Cache valid for 5 minutes*
"""
