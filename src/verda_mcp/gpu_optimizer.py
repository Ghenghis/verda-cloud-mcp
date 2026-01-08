"""GPU Optimizer for Verda MCP Server.

Smart GPU selection and multi-GPU optimization for spot instances.
Key insight: With 75% spot savings, you can often afford 3-4x the GPUs
for the same price as 1x on-demand!

Features:
- Multi-GPU spot vs single GPU on-demand comparison
- Training speed estimator with multi-GPU scaling
- Model-to-GPU recommendations
- Budget-based GPU configurator
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# COMPLETE GPU DATABASE WITH SPOT PRICING
# =============================================================================

GPU_DATABASE = {
    # NVLink GPUs (High-Performance)
    "GB300": {
        "name": "NVIDIA GB300",
        "vram_gb": 288,
        "on_demand": 5.45,
        "spot": 1.36,
        "multi_gpu": [1, 2, 4],
        "fp16_tflops": 1800,  # Estimated
        "memory_bandwidth_gbps": 8000,
        "nvlink": True,
        "architecture": "Blackwell",
        "best_for": ["400B+ models", "Research", "Largest models"],
    },
    "B300": {
        "name": "NVIDIA B300 SXM6",
        "vram_gb": 262,
        "on_demand": 4.95,
        "spot": 1.24,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 1500,
        "memory_bandwidth_gbps": 8000,
        "nvlink": True,
        "architecture": "Blackwell",
        "best_for": ["180B+ models", "Large LLMs", "Production training"],
    },
    "B200": {
        "name": "NVIDIA B200 SXM6",
        "vram_gb": 180,
        "on_demand": 3.79,
        "spot": 0.95,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 1200,
        "memory_bandwidth_gbps": 8000,
        "nvlink": True,
        "architecture": "Blackwell",
        "best_for": ["70B-180B models", "Fast training", "Research"],
    },
    "H200": {
        "name": "NVIDIA H200 SXM5",
        "vram_gb": 141,
        "on_demand": 2.99,
        "spot": 0.75,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 990,
        "memory_bandwidth_gbps": 4800,
        "nvlink": True,
        "architecture": "Hopper",
        "best_for": ["70B models", "Production LLMs", "High memory tasks"],
    },
    "H100": {
        "name": "NVIDIA H100 SXM5",
        "vram_gb": 80,
        "on_demand": 2.29,
        "spot": 0.57,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 990,
        "memory_bandwidth_gbps": 3350,
        "nvlink": True,
        "architecture": "Hopper",
        "best_for": ["30B-70B models", "Fast inference", "Standard training"],
    },
    "A100_80G": {
        "name": "NVIDIA A100 80GB SXM4",
        "vram_gb": 80,
        "on_demand": 1.29,
        "spot": 0.32,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 312,
        "memory_bandwidth_gbps": 2039,
        "nvlink": True,
        "architecture": "Ampere",
        "best_for": ["13B-70B models", "Cost-effective training", "Research"],
    },
    "A100_40G": {
        "name": "NVIDIA A100 40GB SXM4",
        "vram_gb": 40,
        "on_demand": 0.7211,
        "spot": 0.18,
        "multi_gpu": [1, 8],
        "fp16_tflops": 312,
        "memory_bandwidth_gbps": 1555,
        "nvlink": True,
        "architecture": "Ampere",
        "best_for": ["7B-13B models", "Budget training", "Testing"],
    },
    "V100": {
        "name": "Tesla V100 16GB",
        "vram_gb": 16,
        "on_demand": 0.1381,
        "spot": 0.035,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 125,
        "memory_bandwidth_gbps": 900,
        "nvlink": True,
        "architecture": "Volta",
        "best_for": ["Small models", "Testing", "Inference", "Budget"],
    },
    # General Compute
    "RTX_PRO_6000": {
        "name": "RTX PRO 6000",
        "vram_gb": 96,
        "on_demand": 1.39,
        "spot": 0.35,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 91,
        "memory_bandwidth_gbps": 960,
        "nvlink": False,
        "architecture": "Ada Lovelace",
        "best_for": ["30B models", "Graphics + AI", "Rendering"],
    },
    "L40S": {
        "name": "NVIDIA L40S",
        "vram_gb": 48,
        "on_demand": 0.9143,
        "spot": 0.23,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 91,
        "memory_bandwidth_gbps": 864,
        "nvlink": False,
        "architecture": "Ada Lovelace",
        "best_for": ["13B-30B models", "Inference", "Fine-tuning"],
    },
    "RTX_6000_ADA": {
        "name": "RTX 6000 Ada",
        "vram_gb": 48,
        "on_demand": 0.8262,
        "spot": 0.21,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 91,
        "memory_bandwidth_gbps": 960,
        "nvlink": False,
        "architecture": "Ada Lovelace",
        "best_for": ["13B-30B models", "Graphics + AI", "Development"],
    },
    "A6000": {
        "name": "RTX A6000",
        "vram_gb": 48,
        "on_demand": 0.49,
        "spot": 0.12,
        "multi_gpu": [1, 2, 4, 8],
        "fp16_tflops": 38,
        "memory_bandwidth_gbps": 768,
        "nvlink": False,
        "architecture": "Ampere",
        "best_for": ["7B-13B models", "Development", "Testing", "Best value"],
    },
}

# Multi-GPU scaling efficiency (not perfectly linear due to communication overhead)
MULTI_GPU_SCALING = {
    1: 1.0,
    2: 1.85,   # ~92.5% efficiency
    4: 3.5,    # ~87.5% efficiency
    8: 6.5,    # ~81.25% efficiency
}


@dataclass
class GPUConfig:
    """GPU configuration."""
    gpu_type: str
    gpu_count: int
    mode: str  # "spot" or "on_demand"
    
    @property
    def hourly_cost(self) -> float:
        gpu = GPU_DATABASE.get(self.gpu_type, {})
        rate = gpu.get(self.mode, gpu.get("on_demand", 0))
        return rate * self.gpu_count
    
    @property
    def total_vram(self) -> int:
        gpu = GPU_DATABASE.get(self.gpu_type, {})
        return gpu.get("vram_gb", 0) * self.gpu_count
    
    @property
    def relative_speed(self) -> float:
        gpu = GPU_DATABASE.get(self.gpu_type, {})
        base_tflops = gpu.get("fp16_tflops", 100)
        scaling = MULTI_GPU_SCALING.get(self.gpu_count, 1.0)
        return base_tflops * scaling


class GPUOptimizer:
    """Optimize GPU selection for training."""
    
    def compare_spot_vs_ondemand(
        self,
        gpu_type: str,
        hours: float = 24,
    ) -> Dict[str, Any]:
        """Compare multi-GPU spot vs single GPU on-demand."""
        gpu = GPU_DATABASE.get(gpu_type.upper().replace("-", "_"), {})
        if not gpu:
            return {"error": f"GPU type {gpu_type} not found"}
        
        on_demand_1x = gpu["on_demand"]
        spot_1x = gpu["spot"]
        
        # How many spot GPUs can you get for the price of 1 on-demand?
        spot_equivalent = int(on_demand_1x / spot_1x) if spot_1x > 0 else 1
        
        comparisons = []
        
        # Add single on-demand as baseline
        comparisons.append({
            "config": f"1x {gpu_type} On-Demand",
            "gpus": 1,
            "mode": "on_demand",
            "hourly": on_demand_1x,
            "daily": on_demand_1x * 24,
            "total": on_demand_1x * hours,
            "vram_total": gpu["vram_gb"],
            "relative_speed": 1.0,
            "speedup": "1x (baseline)",
        })
        
        # Compare with various spot configs
        for count in gpu.get("multi_gpu", [1, 2, 4, 8]):
            spot_cost = spot_1x * count
            scaling = MULTI_GPU_SCALING.get(count, 1.0)
            
            # Calculate effective speedup vs single on-demand
            speedup = scaling
            
            comparisons.append({
                "config": f"{count}x {gpu_type} SPOT",
                "gpus": count,
                "mode": "spot",
                "hourly": spot_cost,
                "daily": spot_cost * 24,
                "total": spot_cost * hours,
                "vram_total": gpu["vram_gb"] * count,
                "relative_speed": scaling,
                "speedup": f"{speedup:.1f}x faster",
                "vs_ondemand_1x": f"${on_demand_1x - spot_cost:.2f}/hr {'CHEAPER' if spot_cost < on_demand_1x else 'more'}",
                "savings_pct": f"{(1 - spot_cost/on_demand_1x) * 100:.0f}%" if spot_cost < on_demand_1x else "N/A",
            })
        
        return {
            "gpu_type": gpu_type,
            "gpu_info": gpu,
            "hours": hours,
            "comparisons": comparisons,
            "recommendation": self._get_recommendation(comparisons, gpu_type),
        }
    
    def _get_recommendation(self, comparisons: List[Dict], gpu_type: str) -> str:
        """Get recommendation based on comparisons."""
        on_demand_cost = comparisons[0]["hourly"]
        
        best_value = None
        for comp in comparisons[1:]:  # Skip on-demand baseline
            if comp["hourly"] <= on_demand_cost and comp["gpus"] > 1:
                if best_value is None or comp["gpus"] > best_value["gpus"]:
                    best_value = comp
        
        if best_value:
            return f"🎯 **BEST VALUE**: {best_value['config']} - {best_value['speedup']} for ${best_value['hourly']:.2f}/hr (same or less than 1x on-demand!)"
        else:
            return f"🎯 Use SPOT for 75% savings. Even 1x {gpu_type} SPOT is much cheaper!"
    
    def find_best_config(
        self,
        model_size_billions: float,
        budget_per_hour: float,
        prefer_spot: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find best GPU configurations for model size and budget."""
        # Estimate VRAM needed (4GB per billion for training with gradients)
        vram_needed = model_size_billions * 4
        
        configs = []
        
        for gpu_type, gpu_info in GPU_DATABASE.items():
            vram = gpu_info["vram_gb"]
            spot_price = gpu_info["spot"]
            on_demand_price = gpu_info["on_demand"]
            
            for count in gpu_info.get("multi_gpu", [1]):
                total_vram = vram * count
                
                if total_vram < vram_needed:
                    continue
                
                # Check spot config
                if prefer_spot:
                    spot_cost = spot_price * count
                    if spot_cost <= budget_per_hour:
                        scaling = MULTI_GPU_SCALING.get(count, 1.0)
                        configs.append({
                            "config": f"{count}x {gpu_type} SPOT",
                            "gpu_type": gpu_type,
                            "count": count,
                            "mode": "spot",
                            "hourly": spot_cost,
                            "total_vram": total_vram,
                            "relative_speed": gpu_info["fp16_tflops"] * scaling,
                            "value_score": (gpu_info["fp16_tflops"] * scaling) / spot_cost,
                        })
                
                # Check on-demand config
                on_demand_cost = on_demand_price * count
                if on_demand_cost <= budget_per_hour:
                    scaling = MULTI_GPU_SCALING.get(count, 1.0)
                    configs.append({
                        "config": f"{count}x {gpu_type} On-Demand",
                        "gpu_type": gpu_type,
                        "count": count,
                        "mode": "on_demand",
                        "hourly": on_demand_cost,
                        "total_vram": total_vram,
                        "relative_speed": gpu_info["fp16_tflops"] * scaling,
                        "value_score": (gpu_info["fp16_tflops"] * scaling) / on_demand_cost,
                    })
        
        # Sort by value score (performance per dollar)
        configs.sort(key=lambda x: x["value_score"], reverse=True)
        
        return configs[:15]  # Top 15 configs
    
    def estimate_training_time(
        self,
        model_size_billions: float,
        dataset_tokens_billions: float,
        gpu_type: str,
        gpu_count: int,
    ) -> Dict[str, Any]:
        """Estimate training time based on GPU config."""
        gpu = GPU_DATABASE.get(gpu_type.upper().replace("-", "_"), {})
        if not gpu:
            return {"error": f"GPU type {gpu_type} not found"}
        
        # Rough estimation: tokens per second scales with TFLOPs
        # Base: ~1000 tokens/sec per 100 TFLOPs for 7B model
        base_tokens_per_sec = (gpu["fp16_tflops"] / 100) * 1000
        
        # Scale for model size (larger models = slower per token)
        model_factor = 7 / model_size_billions  # 7B is baseline
        
        # Scale for multi-GPU
        scaling = MULTI_GPU_SCALING.get(gpu_count, 1.0)
        
        tokens_per_sec = base_tokens_per_sec * model_factor * scaling
        total_tokens = dataset_tokens_billions * 1e9
        
        seconds = total_tokens / tokens_per_sec
        hours = seconds / 3600
        days = hours / 24
        
        # Cost calculation
        spot_cost = gpu["spot"] * gpu_count * hours
        on_demand_cost = gpu["on_demand"] * gpu_count * hours
        
        return {
            "model_size_b": model_size_billions,
            "dataset_tokens_b": dataset_tokens_billions,
            "gpu_config": f"{gpu_count}x {gpu_type}",
            "estimated_tokens_per_sec": int(tokens_per_sec),
            "estimated_hours": round(hours, 1),
            "estimated_days": round(days, 1),
            "spot_cost": round(spot_cost, 2),
            "on_demand_cost": round(on_demand_cost, 2),
            "savings_with_spot": round(on_demand_cost - spot_cost, 2),
        }
    
    def format_comparison(self, data: Dict[str, Any]) -> str:
        """Format spot vs on-demand comparison."""
        gpu_info = data.get("gpu_info", {})
        comparisons = data.get("comparisons", [])
        
        lines = [
            f"# 🚀 Multi-GPU Spot Optimizer: {data['gpu_type']}",
            "",
            f"**GPU**: {gpu_info.get('name', data['gpu_type'])}",
            f"**VRAM**: {gpu_info.get('vram_gb', 0)}GB per GPU",
            f"**Architecture**: {gpu_info.get('architecture', 'N/A')}",
            f"**Duration**: {data['hours']} hours",
            "",
            "## 💡 KEY INSIGHT",
            f"With SPOT pricing (75% off), you can get **{int(gpu_info.get('on_demand', 1) / gpu_info.get('spot', 1))}x more GPUs** for the same price as 1x on-demand!",
            "",
            "## 📊 Comparison Table",
            "",
            "| Configuration | $/hr | Total | VRAM | Speed | vs 1x On-Demand |",
            "|---------------|------|-------|------|-------|-----------------|",
        ]
        
        for comp in comparisons:
            vs_od = comp.get("vs_ondemand_1x", "-")
            lines.append(
                f"| **{comp['config']}** | ${comp['hourly']:.2f} | ${comp['total']:.2f} | {comp['vram_total']}GB | {comp['speedup']} | {vs_od} |"
            )
        
        lines.extend([
            "",
            f"## {data.get('recommendation', '')}",
            "",
            "## ⚠️ SPOT REQUIREMENTS",
            "- Save checkpoints every **10 minutes**",
            "- Attach persistent **volume** for data",
            "- Use **auto-failover** to on-demand",
        ])
        
        return "\n".join(lines)
    
    def format_configs(self, configs: List[Dict], model_size: float, budget: float) -> str:
        """Format GPU configurations."""
        lines = [
            f"# 🎯 Best GPU Configs for {model_size}B Model",
            "",
            f"**Budget**: ${budget:.2f}/hr max",
            f"**VRAM Needed**: ~{model_size * 4:.0f}GB (for training)",
            "",
            "## Top Configurations (by value)",
            "",
            "| Rank | Configuration | $/hr | VRAM | Speed Score | Mode |",
            "|------|---------------|------|------|-------------|------|",
        ]
        
        for i, cfg in enumerate(configs[:10], 1):
            mode_emoji = "🟢" if cfg["mode"] == "spot" else "🔵"
            lines.append(
                f"| {i} | **{cfg['config']}** | ${cfg['hourly']:.2f} | {cfg['total_vram']}GB | {cfg['value_score']:.0f} | {mode_emoji} {cfg['mode']} |"
            )
        
        if configs:
            best = configs[0]
            lines.extend([
                "",
                f"## 🏆 RECOMMENDED: {best['config']}",
                f"- **Cost**: ${best['hourly']:.2f}/hr",
                f"- **VRAM**: {best['total_vram']}GB",
                f"- **Best value for your budget!**",
            ])
        
        return "\n".join(lines)
    
    def format_time_estimate(self, data: Dict[str, Any]) -> str:
        """Format training time estimate."""
        return f"""# ⏱️ Training Time Estimate

## Configuration
- **Model**: {data['model_size_b']}B parameters
- **Dataset**: {data['dataset_tokens_b']}B tokens
- **GPU**: {data['gpu_config']}

## Estimates
- **Tokens/sec**: ~{data['estimated_tokens_per_sec']:,}
- **Training Time**: {data['estimated_hours']} hours ({data['estimated_days']} days)

## Cost Comparison
| Mode | Total Cost |
|------|------------|
| **SPOT** | **${data['spot_cost']:,.2f}** |
| On-Demand | ${data['on_demand_cost']:,.2f} |

## 💰 Savings with SPOT: ${data['savings_with_spot']:,.2f}

*Note: Estimates are approximate. Actual times depend on batch size, optimizations, etc.*
"""


# Global optimizer instance
_optimizer: Optional[GPUOptimizer] = None


def get_optimizer() -> GPUOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = GPUOptimizer()
    return _optimizer


# =============================================================================
# ASYNC WRAPPER FUNCTIONS FOR MCP TOOLS
# =============================================================================

async def compare_multi_gpu_spot(gpu_type: str, hours: float = 24) -> str:
    """Compare multi-GPU spot vs single GPU on-demand."""
    optimizer = get_optimizer()
    data = optimizer.compare_spot_vs_ondemand(gpu_type, hours)
    if "error" in data:
        return f"❌ {data['error']}"
    return optimizer.format_comparison(data)


async def find_best_gpu_config(
    model_size_billions: float,
    budget_per_hour: float,
    prefer_spot: bool = True,
) -> str:
    """Find best GPU configuration for model and budget."""
    optimizer = get_optimizer()
    configs = optimizer.find_best_config(model_size_billions, budget_per_hour, prefer_spot)
    if not configs:
        return f"❌ No configurations found for {model_size_billions}B model within ${budget_per_hour}/hr budget."
    return optimizer.format_configs(configs, model_size_billions, budget_per_hour)


async def estimate_training_time(
    model_size_billions: float,
    dataset_tokens_billions: float,
    gpu_type: str,
    gpu_count: int = 1,
) -> str:
    """Estimate training time and cost."""
    optimizer = get_optimizer()
    data = optimizer.estimate_training_time(
        model_size_billions, dataset_tokens_billions, gpu_type, gpu_count
    )
    if "error" in data:
        return f"❌ {data['error']}"
    return optimizer.format_time_estimate(data)


async def list_all_gpus_detailed() -> str:
    """List all GPUs with detailed specs."""
    lines = [
        "# 🖥️ Complete Verda GPU Catalog",
        "",
        "## NVLink GPUs (High-Performance AI/ML)",
        "",
        "| GPU | VRAM | On-Demand | SPOT | Savings | Architecture | Best For |",
        "|-----|------|-----------|------|---------|--------------|----------|",
    ]
    
    nvlink_gpus = [k for k, v in GPU_DATABASE.items() if v.get("nvlink")]
    general_gpus = [k for k, v in GPU_DATABASE.items() if not v.get("nvlink")]
    
    for gpu_type in sorted(nvlink_gpus, key=lambda x: -GPU_DATABASE[x]["vram_gb"]):
        gpu = GPU_DATABASE[gpu_type]
        savings = int((1 - gpu["spot"] / gpu["on_demand"]) * 100)
        best_for = ", ".join(gpu.get("best_for", [])[:2])
        lines.append(
            f"| **{gpu_type}** | {gpu['vram_gb']}GB | ${gpu['on_demand']:.2f} | ${gpu['spot']:.2f} | {savings}% | {gpu['architecture']} | {best_for} |"
        )
    
    lines.extend([
        "",
        "## General Compute GPUs",
        "",
        "| GPU | VRAM | On-Demand | SPOT | Savings | Architecture | Best For |",
        "|-----|------|-----------|------|---------|--------------|----------|",
    ])
    
    for gpu_type in sorted(general_gpus, key=lambda x: -GPU_DATABASE[x]["vram_gb"]):
        gpu = GPU_DATABASE[gpu_type]
        savings = int((1 - gpu["spot"] / gpu["on_demand"]) * 100)
        best_for = ", ".join(gpu.get("best_for", [])[:2])
        lines.append(
            f"| **{gpu_type}** | {gpu['vram_gb']}GB | ${gpu['on_demand']:.2f} | ${gpu['spot']:.2f} | {savings}% | {gpu['architecture']} | {best_for} |"
        )
    
    lines.extend([
        "",
        "## Multi-GPU Configurations",
        "All GPUs support: 1, 2, 4, 8 GPU configurations (except A100_40G: 1, 8 only)",
        "",
        "## 💡 Pro Tips",
        "- **SPOT = 75% savings** - always use for training!",
        "- **Multi-GPU SPOT** often cheaper than 1x On-Demand",
        "- **A6000** = Best value for 7B-13B models",
        "- **H200** = Best value for 70B models",
        "- **B300** = Best for 180B+ models",
    ])
    
    return "\n".join(lines)
