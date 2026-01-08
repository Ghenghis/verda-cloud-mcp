"""Extended Tools for Verda MCP Server.

Additional utilities for cost management, health monitoring, and training assistance.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# COMPLETE VERDA GPU PRICING - All models (USD per hour per GPU)
# Updated: January 2026
# Note: Verda now uses FIXED pricing only (no spot)
# =============================================================================
GPU_PRICING = {
    # -----------------------------------------------------------------
    # NVLink GPUs (High-Performance AI/ML)
    # -----------------------------------------------------------------
    "GB300": {"on_demand": 5.45, "vram_gb": 288, "multi_gpu": [1, 2, 4]},
    "B300": {"on_demand": 4.95, "vram_gb": 262, "multi_gpu": [1, 2, 4, 8]},
    "B200": {"on_demand": 3.79, "vram_gb": 180, "multi_gpu": [1, 2, 4, 8]},
    "H200": {"on_demand": 2.99, "vram_gb": 141, "multi_gpu": [1, 2, 4, 8]},
    "H100": {"on_demand": 2.29, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8]},
    "A100_80G": {"on_demand": 1.29, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8]},
    "A100-80G": {"on_demand": 1.29, "vram_gb": 80, "multi_gpu": [1, 2, 4, 8]},
    "A100_40G": {"on_demand": 0.7211, "vram_gb": 40, "multi_gpu": [1, 8]},
    "A100-40G": {"on_demand": 0.7211, "vram_gb": 40, "multi_gpu": [1, 8]},
    "V100": {"on_demand": 0.1381, "vram_gb": 16, "multi_gpu": [1, 2, 4, 8]},
    "TESLA_V100": {"on_demand": 0.1381, "vram_gb": 16, "multi_gpu": [1, 2, 4, 8]},
    # -----------------------------------------------------------------
    # General Compute GPUs
    # -----------------------------------------------------------------
    "RTX_PRO_6000": {"on_demand": 1.39, "vram_gb": 96, "multi_gpu": [1, 2, 4, 8]},
    "RTXPRO6000": {"on_demand": 1.39, "vram_gb": 96, "multi_gpu": [1, 2, 4, 8]},
    "L40S": {"on_demand": 0.9143, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "RTX_6000_ADA": {"on_demand": 0.8262, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "RTX6000ADA": {"on_demand": 0.8262, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "A6000": {"on_demand": 0.49, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
    "RTX_A6000": {"on_demand": 0.49, "vram_gb": 48, "multi_gpu": [1, 2, 4, 8]},
}

# GPU Categories for recommendations
GPU_CATEGORIES = {
    "nvlink_high_end": ["GB300", "B300", "B200"],
    "nvlink_mid": ["H200", "H100"],
    "nvlink_budget": ["A100_80G", "A100_40G", "V100"],
    "general_high": ["RTX_PRO_6000"],
    "general_mid": ["L40S", "RTX_6000_ADA"],
    "general_budget": ["A6000"],
}

# Model size to GPU recommendations
MODEL_SIZE_RECOMMENDATIONS = {
    "7B": {"min_vram": 16, "recommended": ["A6000", "L40S", "V100"]},
    "13B": {"min_vram": 32, "recommended": ["A6000", "L40S", "A100_40G"]},
    "30B": {"min_vram": 64, "recommended": ["A100_80G", "H100", "2xA6000"]},
    "70B": {"min_vram": 140, "recommended": ["H200", "2xA100_80G", "4xA6000"]},
    "180B+": {"min_vram": 256, "recommended": ["B300", "B200", "8xA100_80G"]},
}


class CostEstimator:
    """Estimate training costs before deployment."""

    @staticmethod
    def estimate_cost(
        gpu_type: str,
        gpu_count: int,
        hours: float,
        is_spot: bool = False,  # Verda now uses fixed pricing only
    ) -> Dict[str, Any]:
        """Estimate training cost.

        Args:
            gpu_type: GPU type (B300, B200, etc.)
            gpu_count: Number of GPUs
            hours: Expected training hours
            is_spot: Deprecated - Verda uses fixed pricing only

        Returns:
            Cost breakdown dictionary.
        """
        gpu_key = gpu_type.upper().replace("-", "_")
        pricing = GPU_PRICING.get(gpu_key, {"on_demand": 0, "vram_gb": 0, "multi_gpu": [1]})

        hourly_rate = pricing["on_demand"] * gpu_count
        total_cost = hourly_rate * hours
        vram_total = pricing.get("vram_gb", 0) * gpu_count

        return {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "hours": hours,
            "hourly_rate": hourly_rate,
            "total_cost": total_cost,
            "vram_total_gb": vram_total,
            "per_gpu_cost": pricing["on_demand"],
            "multi_gpu_options": pricing.get("multi_gpu", [1]),
        }

    @staticmethod
    def list_all_gpus() -> Dict[str, Any]:
        """List all available GPUs with pricing and specs."""
        gpus = []
        seen = set()

        for gpu_name, info in GPU_PRICING.items():
            # Skip aliases
            clean_name = gpu_name.replace("-", "_").upper()
            if clean_name in seen:
                continue
            seen.add(clean_name)

            gpus.append(
                {
                    "name": gpu_name,
                    "price_per_hour": info["on_demand"],
                    "vram_gb": info.get("vram_gb", 0),
                    "multi_gpu": info.get("multi_gpu", [1]),
                }
            )

        # Sort by price
        gpus.sort(key=lambda x: x["price_per_hour"])
        return {"gpus": gpus, "count": len(gpus)}

    @staticmethod
    def recommend_gpu(model_size_billions: float, budget_per_hour: float = None) -> Dict[str, Any]:
        """Recommend GPU based on model size and budget.

        Args:
            model_size_billions: Model size in billions of parameters
            budget_per_hour: Optional max budget per hour

        Returns:
            Recommendations with explanations.
        """
        # Estimate VRAM needed (rough: 2GB per billion params for inference, 4GB for training)
        vram_needed = model_size_billions * 4  # For training with gradients

        recommendations = []

        for gpu_name, info in GPU_PRICING.items():
            # Skip aliases
            if "-" in gpu_name:
                continue

            vram = info.get("vram_gb", 0)
            price = info["on_demand"]
            multi_gpu = info.get("multi_gpu", [1])

            # Check if budget allows
            if budget_per_hour and price > budget_per_hour:
                continue

            # Single GPU fits
            if vram >= vram_needed:
                recommendations.append(
                    {
                        "config": f"1x {gpu_name}",
                        "total_vram": vram,
                        "hourly_cost": price,
                        "fits": True,
                        "note": "Single GPU sufficient",
                    }
                )

            # Check multi-GPU configs
            for count in multi_gpu:
                if count == 1:
                    continue
                total_vram = vram * count
                total_cost = price * count

                if budget_per_hour and total_cost > budget_per_hour:
                    continue

                if total_vram >= vram_needed:
                    recommendations.append(
                        {
                            "config": f"{count}x {gpu_name}",
                            "total_vram": total_vram,
                            "hourly_cost": total_cost,
                            "fits": True,
                            "note": f"Multi-GPU: {count} GPUs",
                        }
                    )

        # Sort by cost
        recommendations.sort(key=lambda x: x["hourly_cost"])

        return {
            "model_size_b": model_size_billions,
            "vram_needed_gb": vram_needed,
            "budget_per_hour": budget_per_hour,
            "recommendations": recommendations[:10],  # Top 10
        }

    @staticmethod
    def format_estimate(estimate: Dict[str, Any]) -> str:
        """Format estimate for display."""
        vram = estimate.get("vram_total_gb", 0)
        multi = estimate.get("multi_gpu_options", [1])

        return f"""# 💰 Cost Estimate

## Configuration
- **GPU**: {estimate["gpu_type"]} x{estimate["gpu_count"]}
- **Total VRAM**: {vram} GB
- **Duration**: {estimate["hours"]} hours
- **Available configs**: {multi}

## Cost Breakdown
- **Per GPU**: ${estimate.get("per_gpu_cost", 0):.4f}/hr
- **Hourly Rate**: ${estimate["hourly_rate"]:.2f}/hr
- **Total Cost**: ${estimate["total_cost"]:.2f}"""

    @staticmethod
    def format_gpu_list(gpu_data: Dict[str, Any]) -> str:
        """Format GPU list for display."""
        lines = [
            "# 🖥️ Available Verda GPUs",
            "",
            "| GPU | VRAM | $/hr | Multi-GPU Options |",
            "|-----|------|------|-------------------|",
        ]

        for gpu in gpu_data["gpus"]:
            multi = ", ".join(str(x) for x in gpu["multi_gpu"])
            lines.append(f"| {gpu['name']} | {gpu['vram_gb']}GB | ${gpu['price_per_hour']:.4f} | {multi} |")

        lines.append("")
        lines.append(f"**Total GPUs**: {gpu_data['count']}")
        lines.append("")
        lines.append("*Note: Verda uses fixed pricing (no spot instances)*")

        return "\n".join(lines)

    @staticmethod
    def format_recommendations(rec_data: Dict[str, Any]) -> str:
        """Format GPU recommendations for display."""
        lines = [
            "# 🎯 GPU Recommendations",
            "",
            f"**Model Size**: {rec_data['model_size_b']}B parameters",
            f"**VRAM Needed**: ~{rec_data['vram_needed_gb']:.0f}GB (for training)",
        ]

        if rec_data.get("budget_per_hour"):
            lines.append(f"**Budget**: ${rec_data['budget_per_hour']:.2f}/hr max")

        lines.extend(
            [
                "",
                "## Recommended Configurations",
                "",
                "| Config | Total VRAM | $/hr | Notes |",
                "|--------|------------|------|-------|",
            ]
        )

        for rec in rec_data["recommendations"]:
            lines.append(f"| {rec['config']} | {rec['total_vram']}GB | ${rec['hourly_cost']:.2f} | {rec['note']} |")

        if not rec_data["recommendations"]:
            lines.append("| *No configurations found within budget* | - | - | - |")

        return "\n".join(lines)


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
                None,
                lambda: manager.run_command(
                    instance_ip,
                    "nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits",
                ),
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
                None,
                lambda: manager.run_command(instance_ip, "df -h /workspace | tail -1 | awk '{print $5}'"),
            )
            if code == 0:
                disk_pct = int(stdout.strip().replace("%", ""))
                if disk_pct > 90:
                    checks["disk_health"]["status"] = "warning"
                    checks["disk_health"]["details"] = f"Low disk space: {100 - disk_pct}% free"
                elif disk_pct > 95:
                    checks["disk_health"]["status"] = "critical"
                    checks["disk_health"]["details"] = f"Critical: Only {100 - disk_pct}% disk space free!"
                else:
                    checks["disk_health"]["status"] = "healthy"
                    checks["disk_health"]["details"] = f"{100 - disk_pct}% disk space available"
        except Exception as e:
            checks["disk_health"]["status"] = "error"
            checks["disk_health"]["details"] = str(e)

        # Memory Health
        try:
            stdout, stderr, code = await loop.run_in_executor(
                None,
                lambda: manager.run_command(instance_ip, "free -m | grep Mem | awk '{print $3/$2 * 100}'"),
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
                None,
                lambda: manager.run_command(instance_ip, "ps aux | grep -E 'python.*train' | grep -v grep"),
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
                None,
                lambda: manager.run_command(instance_ip, "ping -c 1 google.com > /dev/null 2>&1 && echo 'OK'"),
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
            "# 🏥 Health Check Report",
            f"**Overall Status**: {status_emoji.get(overall, '❓')} {overall.upper()}",
            f"**Timestamp**: {checks.get('timestamp', 'N/A')}",
            f"**Instance**: {checks.get('instance_ip', 'N/A')}",
            "",
            "## Component Status",
        ]

        components = [
            "gpu_health",
            "disk_health",
            "memory_health",
            "training_health",
            "network_health",
        ]
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
            None,
            lambda: manager.run_command(
                instance_ip,
                f"ls -la {checkpoint_dir}/checkpoint-* 2>/dev/null || echo 'No checkpoints'",
            ),
        )

        checkpoints = []
        for line in stdout.split("\n"):
            if "checkpoint-" in line:
                parts = line.split()
                if len(parts) >= 9:
                    checkpoints.append(
                        {
                            "name": parts[-1],
                            "size": parts[4],
                            "date": " ".join(parts[5:8]),
                        }
                    )

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
                None,
                lambda: transfer.download_file(instance_ip, checkpoint_path, local_path),
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

    logs = await loop.run_in_executor(None, lambda: manager.get_training_logs(instance_ip, lines=log_lines))

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


async def list_all_gpus() -> str:
    """List all available GPUs with pricing and specs."""
    estimator = CostEstimator()
    gpu_data = estimator.list_all_gpus()
    return estimator.format_gpu_list(gpu_data)


async def recommend_gpu_for_model(model_size_billions: float, budget_per_hour: float = None) -> str:
    """Recommend GPU configuration for a model size."""
    estimator = CostEstimator()
    recommendations = estimator.recommend_gpu(model_size_billions, budget_per_hour)
    return estimator.format_recommendations(recommendations)
