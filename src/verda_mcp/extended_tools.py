"""Extended Tools for Verda MCP Server.

Additional utilities for cost management, health monitoring, and training assistance.
"""

import asyncio
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# GPU Pricing (USD per hour) - Updated Jan 2026
GPU_PRICING = {
    "B300": {"on_demand": 4.95, "spot": 1.24},
    "B200": {"on_demand": 3.79, "spot": 0.95},
    "GB300": {"on_demand": 5.95, "spot": 1.49},
    "H200": {"on_demand": 2.99, "spot": 0.75},
    "H100": {"on_demand": 2.49, "spot": 0.62},
    "L40S": {"on_demand": 1.19, "spot": 0.30},
    "A100_80G": {"on_demand": 1.89, "spot": 0.47},
    # Budget options
    "A6000": {"on_demand": 0.49, "spot": 0.49},  # Fixed pricing only
    "RTX_A6000": {"on_demand": 0.49, "spot": 0.49},
    "RTX6000ADA": {"on_demand": 0.93, "spot": 0.93},
    "V100": {"on_demand": 0.44, "spot": 0.44},
}


class CostEstimator:
    """Estimate training costs before deployment."""
    
    @staticmethod
    def estimate_cost(
        gpu_type: str,
        gpu_count: int,
        hours: float,
        is_spot: bool = True,
    ) -> Dict[str, Any]:
        """Estimate training cost.
        
        Args:
            gpu_type: GPU type (B300, B200, etc.)
            gpu_count: Number of GPUs
            hours: Expected training hours
            is_spot: Whether using spot pricing
        
        Returns:
            Cost breakdown dictionary.
        """
        pricing = GPU_PRICING.get(gpu_type.upper(), {"on_demand": 0, "spot": 0})
        rate_type = "spot" if is_spot else "on_demand"
        hourly_rate = pricing[rate_type] * gpu_count
        total_cost = hourly_rate * hours
        
        return {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "hours": hours,
            "is_spot": is_spot,
            "hourly_rate": hourly_rate,
            "total_cost": total_cost,
            "comparison": {
                "spot_total": pricing["spot"] * gpu_count * hours,
                "on_demand_total": pricing["on_demand"] * gpu_count * hours,
                "savings_with_spot": (pricing["on_demand"] - pricing["spot"]) * gpu_count * hours,
            }
        }
    
    @staticmethod
    def format_estimate(estimate: Dict[str, Any]) -> str:
        """Format estimate for display."""
        return f"""# 💰 Cost Estimate

## Configuration
- **GPU**: {estimate['gpu_type']} x{estimate['gpu_count']}
- **Duration**: {estimate['hours']} hours
- **Pricing**: {'Spot' if estimate['is_spot'] else 'On-Demand'}

## Cost Breakdown
- **Hourly Rate**: ${estimate['hourly_rate']:.2f}/hr
- **Total Cost**: ${estimate['total_cost']:.2f}

## Comparison
| Pricing Type | Total Cost |
|--------------|------------|
| Spot | ${estimate['comparison']['spot_total']:.2f} |
| On-Demand | ${estimate['comparison']['on_demand_total']:.2f} |
| **Savings with Spot** | ${estimate['comparison']['savings_with_spot']:.2f} |
"""


class TrainingLogParser:
    """Parse and analyze training logs."""
    
    @staticmethod
    def parse_logs(log_content: str) -> Dict[str, Any]:
        """Parse training logs to extract metrics.
        
        Args:
            log_content: Raw log content string.
        
        Returns:
            Dictionary with parsed metrics.
        """
        metrics = {
            "steps": [],
            "losses": [],
            "learning_rates": [],
            "epochs": [],
            "eval_losses": [],
            "timestamps": [],
            "errors": [],
            "warnings": [],
            "checkpoints_saved": [],
        }
        
        # Common patterns
        patterns = {
            "step": r"(?:step|Step|global_step)[:\s=]+(\d+)",
            "loss": r"(?:loss|Loss)[:\s=]+([0-9.]+)",
            "lr": r"(?:lr|learning_rate|LR)[:\s=]+([0-9.e\-]+)",
            "epoch": r"(?:epoch|Epoch)[:\s=]+([0-9.]+)",
            "eval_loss": r"(?:eval_loss|validation_loss)[:\s=]+([0-9.]+)",
            "checkpoint": r"(?:Saving|Checkpoint saved|checkpoint-\d+)",
            "error": r"(?:Error|ERROR|Exception|EXCEPTION|Traceback)",
            "warning": r"(?:Warning|WARNING|WARN)",
        }
        
        for line in log_content.split("\n"):
            # Extract step
            match = re.search(patterns["step"], line)
            if match:
                metrics["steps"].append(int(match.group(1)))
            
            # Extract loss
            match = re.search(patterns["loss"], line)
            if match:
                try:
                    metrics["losses"].append(float(match.group(1)))
                except ValueError:
                    pass
            
            # Extract learning rate
            match = re.search(patterns["lr"], line)
            if match:
                try:
                    metrics["learning_rates"].append(float(match.group(1)))
                except ValueError:
                    pass
            
            # Extract epoch
            match = re.search(patterns["epoch"], line)
            if match:
                try:
                    metrics["epochs"].append(float(match.group(1)))
                except ValueError:
                    pass
            
            # Extract eval loss
            match = re.search(patterns["eval_loss"], line)
            if match:
                try:
                    metrics["eval_losses"].append(float(match.group(1)))
                except ValueError:
                    pass
            
            # Detect checkpoints
            if re.search(patterns["checkpoint"], line):
                metrics["checkpoints_saved"].append(line.strip())
            
            # Detect errors
            if re.search(patterns["error"], line, re.IGNORECASE):
                metrics["errors"].append(line.strip())
            
            # Detect warnings
            if re.search(patterns["warning"], line, re.IGNORECASE):
                metrics["warnings"].append(line.strip())
        
        # Calculate summary stats
        metrics["summary"] = {
            "total_steps": max(metrics["steps"]) if metrics["steps"] else 0,
            "latest_loss": metrics["losses"][-1] if metrics["losses"] else None,
            "min_loss": min(metrics["losses"]) if metrics["losses"] else None,
            "checkpoints_count": len(metrics["checkpoints_saved"]),
            "errors_count": len(metrics["errors"]),
            "warnings_count": len(metrics["warnings"]),
        }
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """Format parsed metrics for display."""
        summary = metrics.get("summary", {})
        
        result = [
            "# 📊 Training Metrics Analysis",
            "",
            "## Summary",
            f"- **Total Steps**: {summary.get('total_steps', 'N/A')}",
            f"- **Latest Loss**: {summary.get('latest_loss', 'N/A')}",
            f"- **Best Loss**: {summary.get('min_loss', 'N/A')}",
            f"- **Checkpoints Saved**: {summary.get('checkpoints_count', 0)}",
            f"- **Errors**: {summary.get('errors_count', 0)}",
            f"- **Warnings**: {summary.get('warnings_count', 0)}",
        ]
        
        if metrics.get("errors"):
            result.append("\n## ❌ Errors Detected")
            for err in metrics["errors"][-5:]:  # Last 5 errors
                result.append(f"```\n{err[:200]}\n```")
        
        if metrics.get("checkpoints_saved"):
            result.append("\n## 💾 Recent Checkpoints")
            for cp in metrics["checkpoints_saved"][-3:]:
                result.append(f"- {cp[:100]}")
        
        return "\n".join(result)


class HealthChecker:
    """Check instance and training health."""
    
    @staticmethod
    async def comprehensive_health_check(instance_ip: str) -> Dict[str, Any]:
        """Perform comprehensive health check on an instance.
        
        Args:
            instance_ip: IP address of the instance.
        
        Returns:
            Health check results.
        """
        from .ssh_tools import get_ssh_manager
        
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        checks = {
            "timestamp": datetime.now().isoformat(),
            "instance_ip": instance_ip,
            "gpu_health": {"status": "unknown", "details": ""},
            "disk_health": {"status": "unknown", "details": ""},
            "memory_health": {"status": "unknown", "details": ""},
            "training_health": {"status": "unknown", "details": ""},
            "network_health": {"status": "unknown", "details": ""},
        }
        
        # GPU Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits")
            )
            if code == 0:
                parts = stdout.strip().split(",")
                if len(parts) >= 4:
                    mem_used = float(parts[0].strip())
                    mem_total = float(parts[1].strip())
                    temp = float(parts[2].strip())
                    util = float(parts[3].strip())
                    
                    mem_pct = (mem_used / mem_total) * 100
                    
                    if temp > 85:
                        checks["gpu_health"]["status"] = "warning"
                        checks["gpu_health"]["details"] = f"High temperature: {temp}°C"
                    elif mem_pct > 95:
                        checks["gpu_health"]["status"] = "warning"
                        checks["gpu_health"]["details"] = f"High memory usage: {mem_pct:.1f}%"
                    else:
                        checks["gpu_health"]["status"] = "healthy"
                        checks["gpu_health"]["details"] = f"Temp: {temp}°C, Mem: {mem_pct:.1f}%, Util: {util}%"
        except Exception as e:
            checks["gpu_health"]["status"] = "error"
            checks["gpu_health"]["details"] = str(e)
        
        # Disk Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "df -h /workspace | tail -1 | awk '{print $5}'")
            )
            if code == 0:
                disk_pct = int(stdout.strip().replace("%", ""))
                if disk_pct > 90:
                    checks["disk_health"]["status"] = "warning"
                    checks["disk_health"]["details"] = f"Low disk space: {100-disk_pct}% free"
                elif disk_pct > 95:
                    checks["disk_health"]["status"] = "critical"
                    checks["disk_health"]["details"] = f"Critical: Only {100-disk_pct}% disk space free!"
                else:
                    checks["disk_health"]["status"] = "healthy"
                    checks["disk_health"]["details"] = f"{100-disk_pct}% disk space available"
        except Exception as e:
            checks["disk_health"]["status"] = "error"
            checks["disk_health"]["details"] = str(e)
        
        # Memory Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "free -m | grep Mem | awk '{print $3/$2 * 100}'")
            )
            if code == 0:
                mem_pct = float(stdout.strip())
                if mem_pct > 90:
                    checks["memory_health"]["status"] = "warning"
                    checks["memory_health"]["details"] = f"High RAM usage: {mem_pct:.1f}%"
                else:
                    checks["memory_health"]["status"] = "healthy"
                    checks["memory_health"]["details"] = f"RAM usage: {mem_pct:.1f}%"
        except Exception as e:
            checks["memory_health"]["status"] = "error"
            checks["memory_health"]["details"] = str(e)
        
        # Training Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "ps aux | grep -E 'python.*train' | grep -v grep")
            )
            if stdout.strip():
                checks["training_health"]["status"] = "running"
                checks["training_health"]["details"] = "Training process active"
            else:
                checks["training_health"]["status"] = "stopped"
                checks["training_health"]["details"] = "No training process detected"
        except Exception as e:
            checks["training_health"]["status"] = "error"
            checks["training_health"]["details"] = str(e)
        
        # Network Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "ping -c 1 google.com > /dev/null 2>&1 && echo 'OK'")
            )
            if "OK" in stdout:
                checks["network_health"]["status"] = "healthy"
                checks["network_health"]["details"] = "Internet connectivity OK"
            else:
                checks["network_health"]["status"] = "warning"
                checks["network_health"]["details"] = "Limited connectivity"
        except Exception as e:
            checks["network_health"]["status"] = "error"
            checks["network_health"]["details"] = str(e)
        
        # Overall status
        statuses = [c["status"] for c in checks.values() if isinstance(c, dict) and "status" in c]
        if "critical" in statuses or "error" in statuses:
            checks["overall"] = "critical"
        elif "warning" in statuses or "stopped" in statuses:
            checks["overall"] = "warning"
        else:
            checks["overall"] = "healthy"
        
        return checks
    
    @staticmethod
    def format_health_check(checks: Dict[str, Any]) -> str:
        """Format health check results for display."""
        status_emoji = {
            "healthy": "✅",
            "running": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "error": "❌",
            "stopped": "⏹️",
            "unknown": "❓",
        }
        
        overall = checks.get("overall", "unknown")
        
        result = [
            f"# 🏥 Health Check Report",
            f"**Overall Status**: {status_emoji.get(overall, '❓')} {overall.upper()}",
            f"**Timestamp**: {checks.get('timestamp', 'N/A')}",
            f"**Instance**: {checks.get('instance_ip', 'N/A')}",
            "",
            "## Component Status",
        ]
        
        components = ["gpu_health", "disk_health", "memory_health", "training_health", "network_health"]
        for comp in components:
            if comp in checks and isinstance(checks[comp], dict):
                status = checks[comp].get("status", "unknown")
                details = checks[comp].get("details", "")
                emoji = status_emoji.get(status, "❓")
                name = comp.replace("_health", "").upper()
                result.append(f"- {emoji} **{name}**: {details}")
        
        return "\n".join(result)


class CheckpointManager:
    """Manage training checkpoints."""
    
    @staticmethod
    async def list_checkpoints(instance_ip: str, checkpoint_dir: str = "/workspace/outputs") -> list:
        """List available checkpoints on the instance.
        
        Args:
            instance_ip: IP address of the instance.
            checkpoint_dir: Directory containing checkpoints.
        
        Returns:
            List of checkpoint info.
        """
        from .ssh_tools import get_ssh_manager
        
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, f"ls -la {checkpoint_dir}/checkpoint-* 2>/dev/null || echo 'No checkpoints'")
        )
        
        checkpoints = []
        for line in stdout.split("\n"):
            if "checkpoint-" in line:
                parts = line.split()
                if len(parts) >= 9:
                    checkpoints.append({
                        "name": parts[-1],
                        "size": parts[4],
                        "date": " ".join(parts[5:8]),
                    })
        
        return checkpoints
    
    @staticmethod
    async def backup_checkpoint(
        instance_ip: str,
        checkpoint_path: str,
        local_path: str,
    ) -> str:
        """Backup a checkpoint from instance to local machine.
        
        Args:
            instance_ip: IP address of the instance.
            checkpoint_path: Path to checkpoint on instance.
            local_path: Local destination path.
        
        Returns:
            Status message.
        """
        from .gdrive_tools import get_transfer_manager
        
        transfer = get_transfer_manager()
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(
                None, lambda: transfer.download_file(instance_ip, checkpoint_path, local_path)
            )
            return f"✅ Checkpoint backed up to: {local_path}"
        except Exception as e:
            return f"❌ Backup failed: {e}"


# Async wrapper functions for MCP tools

async def estimate_training_cost(
    gpu_type: str,
    gpu_count: int,
    hours: float,
    is_spot: bool = True,
) -> str:
    """Estimate training cost."""
    estimator = CostEstimator()
    estimate = estimator.estimate_cost(gpu_type, gpu_count, hours, is_spot)
    return estimator.format_estimate(estimate)


async def parse_training_logs(instance_ip: str, log_lines: int = 200) -> str:
    """Parse and analyze training logs from instance."""
    from .ssh_tools import get_ssh_manager
    
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    
    logs = await loop.run_in_executor(
        None, lambda: manager.get_training_logs(instance_ip, lines=log_lines)
    )
    
    parser = TrainingLogParser()
    metrics = parser.parse_logs(logs)
    return parser.format_metrics(metrics)


async def health_check(instance_ip: str) -> str:
    """Perform comprehensive health check."""
    checker = HealthChecker()
    checks = await checker.comprehensive_health_check(instance_ip)
    return checker.format_health_check(checks)


async def list_instance_checkpoints(instance_ip: str) -> str:
    """List checkpoints on instance."""
    manager = CheckpointManager()
    checkpoints = await manager.list_checkpoints(instance_ip)
    
    if not checkpoints:
        return "No checkpoints found in /workspace/outputs/"
    
    result = ["# 💾 Available Checkpoints", ""]
    for cp in checkpoints:
        result.append(f"- **{cp['name']}** ({cp['size']} bytes) - {cp['date']}")
    
    return "\n".join(result)


async def backup_latest_checkpoint(instance_ip: str, local_dir: str) -> str:
    """Backup the latest checkpoint to local machine."""
    manager = CheckpointManager()
    checkpoints = await manager.list_checkpoints(instance_ip)
    
    if not checkpoints:
        return "❌ No checkpoints found to backup."
    
    latest = checkpoints[-1]
    checkpoint_path = f"/workspace/outputs/{latest['name']}"
    local_path = f"{local_dir}/{latest['name']}"
    
    return await manager.backup_checkpoint(instance_ip, checkpoint_path, local_path)
