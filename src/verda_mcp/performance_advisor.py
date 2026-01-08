"""
Performance Advisor Module for Verda MCP Server.

Provides intelligent recommendations for:
- Best GPU configurations for speed vs cost
- Training time estimation and planning
- Budget management and spending alerts
- Multi-GPU scaling efficiency
- First-timer best practices and tips
"""

from dataclasses import dataclass
from enum import Enum


class TrainingGoal(Enum):
    """Training optimization goals."""

    FASTEST = "fastest"  # Minimize time, budget flexible
    BALANCED = "balanced"  # Good speed, reasonable cost
    BUDGET = "budget"  # Minimize cost, time flexible
    BEST_VALUE = "best_value"  # Best performance per dollar


class ExperienceLevel(Enum):
    """User experience level."""

    FIRST_TIMER = "first_timer"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class GPUPerformance:
    """GPU performance characteristics."""

    name: str
    vram_gb: int
    tflops: float
    spot_price: float
    ondemand_price: float
    configs: list[int]  # Available GPU counts
    best_for: str
    nvlink: bool = True


@dataclass
class TrainingPlan:
    """Training execution plan."""

    gpu_type: str
    gpu_count: int
    is_spot: bool
    estimated_hours: float
    total_cost: float
    speedup_factor: float
    checkpoint_interval_min: int
    recommendation: str
    warnings: list[str]


@dataclass
class BudgetPlan:
    """Budget management plan."""

    max_budget: float
    recommended_config: str
    max_hours: float
    alert_threshold: float
    stop_threshold: float
    action_plan: list[str]


# Complete GPU Database with Performance Data
GPU_PERFORMANCE = {
    "GB300": GPUPerformance("GB300", 288, 1800, 1.36, 5.45, [1, 2, 4], "400B+ models", True),
    "B300": GPUPerformance("B300", 262, 1500, 1.24, 4.95, [1, 2, 4, 8], "180B+ models", True),
    "B200": GPUPerformance("B200", 180, 1200, 0.95, 3.79, [1, 2, 4, 8], "70B-180B models", True),
    "H200": GPUPerformance("H200", 141, 990, 0.75, 2.99, [1, 2, 4, 8], "70B models", True),
    "H100": GPUPerformance("H100", 80, 990, 0.57, 2.29, [1, 2, 4, 8], "30B-70B models", True),
    "A100_80G": GPUPerformance("A100_80G", 80, 312, 0.32, 1.29, [1, 2, 4, 8], "13B-70B models", True),
    "A100_40G": GPUPerformance("A100_40G", 40, 312, 0.18, 0.72, [1, 8], "7B-13B models", True),
    "V100": GPUPerformance("V100", 16, 125, 0.035, 0.14, [1, 2, 4, 8], "Small models", True),
    "RTX_PRO_6000": GPUPerformance("RTX_PRO_6000", 96, 91, 0.35, 1.39, [1, 2, 4, 8], "30B models", False),
    "L40S": GPUPerformance("L40S", 48, 91, 0.23, 0.91, [1, 2, 4, 8], "13B-30B models", False),
    "RTX_6000_ADA": GPUPerformance("RTX_6000_ADA", 48, 91, 0.21, 0.83, [1, 2, 4, 8], "Development", False),
    "A6000": GPUPerformance("A6000", 48, 38, 0.12, 0.49, [1, 2, 4, 8], "7B-13B BEST VALUE", False),
}

# Multi-GPU Scaling Efficiency
SCALING_EFFICIENCY = {
    1: 1.0,  # 100% efficiency
    2: 1.85,  # 92.5% per GPU
    4: 3.5,  # 87.5% per GPU
    8: 6.5,  # 81.25% per GPU
}


class PerformanceAdvisor:
    """Intelligent performance advisor for GPU training."""

    def __init__(self):
        self.gpu_db = GPU_PERFORMANCE

    def get_best_speed_config(
        self, model_size_b: float, budget_per_hour: float = 10.0, prefer_spot: bool = True
    ) -> dict:
        """
        Get the FASTEST configuration within budget.

        Args:
            model_size_b: Model size in billions of parameters
            budget_per_hour: Maximum hourly budget
            prefer_spot: Prefer spot instances (75% savings!)

        Returns:
            Best configuration for speed
        """
        min_vram = self._estimate_vram_needed(model_size_b)
        candidates = []

        for gpu_name, gpu in self.gpu_db.items():
            for count in gpu.configs:
                total_vram = gpu.vram_gb * count
                if total_vram < min_vram:
                    continue

                price = gpu.spot_price if prefer_spot else gpu.ondemand_price
                total_price = price * count

                if total_price > budget_per_hour:
                    continue

                effective_tflops = gpu.tflops * SCALING_EFFICIENCY.get(count, count * 0.8)
                value_score = effective_tflops / total_price

                candidates.append(
                    {
                        "gpu": gpu_name,
                        "count": count,
                        "is_spot": prefer_spot,
                        "price_per_hour": total_price,
                        "total_vram": total_vram,
                        "effective_tflops": effective_tflops,
                        "value_score": value_score,
                        "speedup_vs_single": SCALING_EFFICIENCY.get(count, count * 0.8),
                    }
                )

        if not candidates:
            return {"error": f"No GPU config fits budget ${budget_per_hour}/hr for {model_size_b}B model"}

        # Sort by effective TFLOPs (speed) descending
        candidates.sort(key=lambda x: x["effective_tflops"], reverse=True)

        best = candidates[0]
        return {
            "recommendation": f"🚀 FASTEST: {best['count']}x {best['gpu']} SPOT",
            "config": best,
            "explanation": f"{best['count']}x GPUs = {best['speedup_vs_single']:.1f}x faster than single GPU!",
            "savings_vs_ondemand": "75%" if prefer_spot else "0%",
            "alternatives": candidates[1:4] if len(candidates) > 1 else [],
        }

    def get_best_value_config(self, model_size_b: float, max_budget: float = 5.0) -> dict:
        """
        Get the BEST VALUE (performance per dollar) configuration.

        Args:
            model_size_b: Model size in billions
            max_budget: Maximum hourly budget

        Returns:
            Best value configuration
        """
        result = self.get_best_speed_config(model_size_b, max_budget, prefer_spot=True)

        if "error" in result:
            return result

        # Re-sort by value score
        all_configs = [result["config"]] + result.get("alternatives", [])
        all_configs.sort(key=lambda x: x["value_score"], reverse=True)

        best = all_configs[0]
        return {
            "recommendation": f"💰 BEST VALUE: {best['count']}x {best['gpu']} SPOT",
            "config": best,
            "explanation": f"Best TFLOPs per dollar! {best['value_score']:.0f} TFLOPs/$",
            "tip": "Multi-GPU SPOT often beats single On-Demand at same price!",
        }

    def calculate_training_time(
        self,
        model_size_b: float,
        dataset_tokens_b: float,
        gpu_type: str = "B300",
        gpu_count: int = 1,
    ) -> dict:
        """
        Estimate training time for a configuration.

        Args:
            model_size_b: Model size in billions
            dataset_tokens_b: Dataset size in billions of tokens
            gpu_type: GPU type
            gpu_count: Number of GPUs

        Returns:
            Training time estimate
        """
        gpu = self.gpu_db.get(gpu_type)
        if not gpu:
            return {"error": f"Unknown GPU: {gpu_type}"}

        # Rough estimation: tokens per second based on TFLOPs
        # This is a simplified model - real performance varies
        base_tokens_per_sec = gpu.tflops * 100  # Rough approximation
        effective_tokens_per_sec = base_tokens_per_sec * SCALING_EFFICIENCY.get(gpu_count, gpu_count * 0.8)

        total_tokens = dataset_tokens_b * 1e9
        seconds = total_tokens / effective_tokens_per_sec
        hours = seconds / 3600

        # Cost calculation
        spot_cost = gpu.spot_price * gpu_count * hours
        ondemand_cost = gpu.ondemand_price * gpu_count * hours

        return {
            "gpu_config": f"{gpu_count}x {gpu_type}",
            "estimated_hours": round(hours, 1),
            "estimated_days": round(hours / 24, 1),
            "spot_cost": f"${spot_cost:.2f}",
            "ondemand_cost": f"${ondemand_cost:.2f}",
            "savings_with_spot": f"${ondemand_cost - spot_cost:.2f} (75%)",
            "speedup_factor": SCALING_EFFICIENCY.get(gpu_count, gpu_count * 0.8),
            "checkpoint_recommendation": "Every 10 minutes for SPOT!",
        }

    def create_budget_plan(
        self, total_budget: float, model_size_b: float, goal: TrainingGoal = TrainingGoal.BALANCED
    ) -> BudgetPlan:
        """
        Create a budget management plan.

        Args:
            total_budget: Total available budget in USD
            model_size_b: Model size in billions
            goal: Training optimization goal

        Returns:
            Budget management plan
        """
        # Determine hourly budget based on goal
        if goal == TrainingGoal.FASTEST:
            hourly_budget = total_budget / 4  # Assume 4 hours max
        elif goal == TrainingGoal.BUDGET:
            hourly_budget = total_budget / 24  # Stretch over 24 hours
        else:
            hourly_budget = total_budget / 10  # Balanced: ~10 hours

        config = self.get_best_speed_config(model_size_b, hourly_budget, prefer_spot=True)

        if "error" in config:
            recommended = "Start with 1x A6000 SPOT ($0.12/hr)"
            max_hours = total_budget / 0.12
        else:
            cfg = config["config"]
            recommended = f"{cfg['count']}x {cfg['gpu']} SPOT"
            max_hours = total_budget / cfg["price_per_hour"]

        return BudgetPlan(
            max_budget=total_budget,
            recommended_config=recommended,
            max_hours=max_hours,
            alert_threshold=total_budget * 0.7,
            stop_threshold=total_budget * 0.95,
            action_plan=[
                f"✅ Start with {recommended}",
                f"⏰ Max training time: {max_hours:.1f} hours",
                f"⚠️ Alert at 70%: ${total_budget * 0.7:.2f} spent",
                f"🛑 Auto-stop at 95%: ${total_budget * 0.95:.2f} spent",
                "💾 Checkpoint every 10 minutes (CRITICAL for spot!)",
                "📊 Monitor with: train_intel(action='status')",
            ],
        )

    def compare_speeds(self, gpu_type: str = "B300") -> dict:
        """
        Compare training speeds across GPU configurations.

        Args:
            gpu_type: GPU type to compare

        Returns:
            Speed comparison table
        """
        gpu = self.gpu_db.get(gpu_type)
        if not gpu:
            return {"error": f"Unknown GPU: {gpu_type}"}

        comparisons = []
        base_speed = gpu.tflops

        for count in gpu.configs:
            effective = base_speed * SCALING_EFFICIENCY.get(count, count * 0.8)
            spot_cost = gpu.spot_price * count
            ondemand_cost = gpu.ondemand_price * count

            comparisons.append(
                {
                    "config": f"{count}x {gpu_type}",
                    "effective_tflops": effective,
                    "speedup": f"{SCALING_EFFICIENCY.get(count, count * 0.8):.2f}x",
                    "spot_cost": f"${spot_cost:.2f}/hr",
                    "ondemand_cost": f"${ondemand_cost:.2f}/hr",
                    "value_rating": "⭐" * min(5, int(effective / spot_cost / 200)),
                }
            )

        return {
            "gpu": gpu_type,
            "base_tflops": base_speed,
            "comparisons": comparisons,
            "recommendation": f"🏆 Best: {gpu.configs[-1]}x {gpu_type} SPOT for maximum speed!",
        }

    def _estimate_vram_needed(self, model_size_b: float) -> int:
        """Estimate VRAM needed for model size."""
        # Rough rule: 2 bytes per parameter for inference, 4x for training
        return int(model_size_b * 8)  # GB needed for training


class FirstTimerGuide:
    """Best practices guide for first-time users."""

    @staticmethod
    def get_quick_start() -> dict:
        """Get quick start guide for first-timers."""
        return {
            "title": "🎯 FIRST-TIMER QUICK START GUIDE",
            "steps": [
                {
                    "step": 1,
                    "title": "Choose Your GPU",
                    "action": "recommend_gpu(model_size_billions=7)",
                    "tip": "Start small! 7B models are great for learning.",
                },
                {
                    "step": 2,
                    "title": "Use SPOT (75% Savings!)",
                    "action": "smart_deploy(gpu_type='A6000', prefer_spot=True)",
                    "tip": "ALWAYS use SPOT for training. Same GPUs, 75% cheaper!",
                },
                {
                    "step": 3,
                    "title": "Enable Checkpoints",
                    "action": "create_checkpoint_script(checkpoint_minutes=10)",
                    "tip": "CRITICAL! Save every 10 min. Spot can be interrupted.",
                },
                {
                    "step": 4,
                    "title": "Monitor Training",
                    "action": "train_intel(action='status', skill_level='beginner')",
                    "tip": "Watch your progress in simple English!",
                },
                {
                    "step": 5,
                    "title": "Manage Budget",
                    "action": "set_training_cost_alert(threshold_usd=10)",
                    "tip": "Set alerts to avoid surprise bills!",
                },
            ],
        }

    @staticmethod
    def get_tips_and_tricks() -> dict:
        """Get tips and tricks for efficient training."""
        return {
            "title": "💡 TIPS & TRICKS FOR EFFICIENT TRAINING",
            "categories": {
                "💰 Save Money": [
                    "ALWAYS use SPOT instances (75% savings!)",
                    "Multi-GPU SPOT often beats single On-Demand",
                    "Start with A6000 ($0.12/hr SPOT) for testing",
                    "Set budget alerts before starting",
                    "Use checkpoints to resume if interrupted",
                ],
                "⚡ Go Faster": [
                    "4x GPUs = 3.5x speed (87.5% efficiency)",
                    "8x GPUs = 6.5x speed (81% efficiency)",
                    "NVLink GPUs (H100, B300) scale better",
                    "Larger batch sizes = faster training",
                    "Use mixed precision (fp16/bf16)",
                ],
                "🛡️ Stay Safe": [
                    "Checkpoint every 10 minutes (MUST for SPOT!)",
                    "Store checkpoints on persistent volume",
                    "Enable WatchDog monitoring",
                    "Set auto-stop budget limits",
                    "Use deploy_failsafe() for production",
                ],
                "📊 Monitor Well": [
                    "Use train_intel() for real-time status",
                    "Check train_viz(format='ascii') for terminal",
                    "Set skill_level='beginner' for simple output",
                    "Watch loss curve - should go down!",
                    "GPU utilization should be >90%",
                ],
            },
        }

    @staticmethod
    def get_common_mistakes() -> dict:
        """Get common mistakes to avoid."""
        return {
            "title": "⚠️ COMMON MISTAKES TO AVOID",
            "mistakes": [
                {
                    "mistake": "Using On-Demand instead of SPOT",
                    "impact": "Paying 4x more for same GPU!",
                    "fix": "Always use smart_deploy(prefer_spot=True)",
                },
                {
                    "mistake": "Not enabling checkpoints",
                    "impact": "Lose ALL progress if SPOT interrupted!",
                    "fix": "ALWAYS checkpoint every 10 minutes",
                },
                {
                    "mistake": "Wrong GPU for model size",
                    "impact": "Out of memory or wasted resources",
                    "fix": "Use recommend_gpu(model_size_billions=X)",
                },
                {
                    "mistake": "No budget monitoring",
                    "impact": "Surprise bills!",
                    "fix": "set_training_cost_alert(threshold_usd=X)",
                },
                {
                    "mistake": "Using 1 GPU when multi-GPU is same price",
                    "impact": "Training takes 4x longer!",
                    "fix": "Check best_deals_now() for multi-GPU SPOT deals",
                },
            ],
        }

    @staticmethod
    def get_model_size_guide() -> dict:
        """Get guide for choosing model size."""
        return {
            "title": "📏 MODEL SIZE GUIDE",
            "recommendations": [
                {
                    "size": "7B",
                    "vram_needed": "16GB",
                    "best_gpu": "A6000 ($0.12/hr SPOT)",
                    "use_case": "Learning, experimentation, fine-tuning",
                    "tip": "Great starting point!",
                },
                {
                    "size": "13B",
                    "vram_needed": "32GB",
                    "best_gpu": "L40S or 2x A6000",
                    "use_case": "Better quality, still affordable",
                    "tip": "Good balance of quality and cost",
                },
                {
                    "size": "30B",
                    "vram_needed": "64GB",
                    "best_gpu": "H100 or RTX PRO 6000",
                    "use_case": "High quality, serious projects",
                    "tip": "Consider 2x GPUs for speed",
                },
                {
                    "size": "70B",
                    "vram_needed": "140GB",
                    "best_gpu": "H200 or 2x H100",
                    "use_case": "Production quality",
                    "tip": "Multi-GPU required",
                },
                {
                    "size": "180B+",
                    "vram_needed": "360GB+",
                    "best_gpu": "4x B200 or 2x B300",
                    "use_case": "Cutting edge, research",
                    "tip": "Budget carefully!",
                },
            ],
        }


# Main advisor functions for MCP tools
def get_fastest_config(model_size_b: float, budget: float = 10.0) -> str:
    """Get fastest GPU configuration within budget."""
    advisor = PerformanceAdvisor()
    result = advisor.get_best_speed_config(model_size_b, budget, prefer_spot=True)
    return _format_result(result)


def get_best_value(model_size_b: float, budget: float = 5.0) -> str:
    """Get best value GPU configuration."""
    advisor = PerformanceAdvisor()
    result = advisor.get_best_value_config(model_size_b, budget)
    return _format_result(result)


def calculate_training(
    model_size_b: float,
    dataset_tokens_b: float,
    gpu_type: str = "B300",
    gpu_count: int = 1,
) -> str:
    """Calculate training time and cost."""
    advisor = PerformanceAdvisor()
    result = advisor.calculate_training_time(model_size_b, dataset_tokens_b, gpu_type, gpu_count)
    return _format_result(result)


def create_budget(total_budget: float, model_size_b: float, goal: str = "balanced") -> str:
    """Create budget management plan."""
    advisor = PerformanceAdvisor()
    goal_enum = TrainingGoal(goal) if goal in [g.value for g in TrainingGoal] else TrainingGoal.BALANCED
    result = advisor.create_budget_plan(total_budget, model_size_b, goal_enum)
    return _format_budget_plan(result)


def compare_gpu_speeds(gpu_type: str = "B300") -> str:
    """Compare training speeds for GPU configurations."""
    advisor = PerformanceAdvisor()
    result = advisor.compare_speeds(gpu_type)
    return _format_result(result)


def get_first_timer_guide() -> str:
    """Get quick start guide for first-timers."""
    guide = FirstTimerGuide.get_quick_start()
    return _format_guide(guide)


def get_tips_tricks() -> str:
    """Get tips and tricks."""
    tips = FirstTimerGuide.get_tips_and_tricks()
    return _format_tips(tips)


def get_common_mistakes() -> str:
    """Get common mistakes to avoid."""
    mistakes = FirstTimerGuide.get_common_mistakes()
    return _format_mistakes(mistakes)


def get_model_guide() -> str:
    """Get model size selection guide."""
    guide = FirstTimerGuide.get_model_size_guide()
    return _format_model_guide(guide)


# Formatting helpers
def _format_result(result: dict) -> str:
    """Format result dictionary as string."""
    lines = []
    for key, value in result.items():
        if isinstance(value, dict):
            lines.append(f"\n{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        elif isinstance(value, list):
            lines.append(f"\n{key}:")
            for item in value[:3]:  # Limit to 3
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('config', item.get('gpu', 'N/A'))}")
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_budget_plan(plan: BudgetPlan) -> str:
    """Format budget plan."""
    lines = [
        "=" * 50,
        "💰 BUDGET MANAGEMENT PLAN",
        "=" * 50,
        f"Total Budget: ${plan.max_budget:.2f}",
        f"Recommended: {plan.recommended_config}",
        f"Max Hours: {plan.max_hours:.1f}",
        "",
        "📋 ACTION PLAN:",
    ]
    for action in plan.action_plan:
        lines.append(f"  {action}")
    lines.append("=" * 50)
    return "\n".join(lines)


def _format_guide(guide: dict) -> str:
    """Format quick start guide."""
    lines = ["=" * 50, guide["title"], "=" * 50, ""]
    for step_info in guide["steps"]:
        lines.append(f"STEP {step_info['step']}: {step_info['title']}")
        lines.append(f"  Command: {step_info['action']}")
        lines.append(f"  💡 Tip: {step_info['tip']}")
        lines.append("")
    return "\n".join(lines)


def _format_tips(tips: dict) -> str:
    """Format tips and tricks."""
    lines = ["=" * 50, tips["title"], "=" * 50, ""]
    for category, tip_list in tips["categories"].items():
        lines.append(f"\n{category}")
        lines.append("-" * 30)
        for tip in tip_list:
            lines.append(f"  • {tip}")
    return "\n".join(lines)


def _format_mistakes(mistakes: dict) -> str:
    """Format common mistakes."""
    lines = ["=" * 50, mistakes["title"], "=" * 50, ""]
    for m in mistakes["mistakes"]:
        lines.append(f"❌ {m['mistake']}")
        lines.append(f"   Impact: {m['impact']}")
        lines.append(f"   ✅ Fix: {m['fix']}")
        lines.append("")
    return "\n".join(lines)


def _format_model_guide(guide: dict) -> str:
    """Format model size guide."""
    lines = ["=" * 50, guide["title"], "=" * 50, ""]
    for rec in guide["recommendations"]:
        lines.append(f"📦 {rec['size']} Parameters")
        lines.append(f"   VRAM: {rec['vram_needed']}")
        lines.append(f"   Best GPU: {rec['best_gpu']}")
        lines.append(f"   Use Case: {rec['use_case']}")
        lines.append(f"   💡 {rec['tip']}")
        lines.append("")
    return "\n".join(lines)
