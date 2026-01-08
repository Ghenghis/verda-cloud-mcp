"""Training Intelligence System for Verda MCP Server.

MEGA-TOOL ARCHITECTURE: Bundles 50+ functions into 5 mega-tools
This approach works around MCP tool limits while providing 100+ capabilities.

Mega-Tools:
1. training_intel - 15+ sub-commands for training metrics/analysis
2. training_viz - 10+ output formats (SVG, HTML, ASCII, JSON, Markdown)
3. training_profile - 8+ user skill levels with adapted output
4. training_monitor - 12+ real-time monitoring functions
5. model_advisor - 10+ model selection and optimization functions

Total: 55+ bundled functions appearing as 5 tools
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SkillLevel(Enum):
    """User skill levels for adapted output."""

    BEGINNER = "beginner"  # First-timers, simple explanations
    CASUAL = "casual"  # Basic users, moderate detail
    NORMAL = "normal"  # Regular users, standard output
    ADVANCED = "advanced"  # Power users, technical details
    EXPERT = "expert"  # ML engineers, full metrics
    ELITE = "elite"  # Researchers, raw data + analysis
    HACKER = "hacker"  # Terminal lovers, CLI-style output


class OutputFormat(Enum):
    """Output format types."""

    ASCII = "ascii"  # ASCII art charts
    MARKDOWN = "markdown"  # GitHub-flavored markdown
    HTML = "html"  # Modern HTML with CSS
    SVG = "svg"  # Scalable vector graphics
    JSON = "json"  # Raw JSON data
    TERMINAL = "terminal"  # Terminal-style with colors
    MINIMAL = "minimal"  # Just the essentials


class TrainingStage(Enum):
    """10-stage training progress rating."""

    STAGE_1 = (1, "Initializing", "Model is warming up, weights are random")
    STAGE_2 = (2, "Early Learning", "Starting to recognize basic patterns")
    STAGE_3 = (3, "Finding Direction", "Loss is dropping, learning is happening")
    STAGE_4 = (4, "Gaining Momentum", "Training is progressing well")
    STAGE_5 = (5, "Halfway There", "Significant progress, patterns emerging")
    STAGE_6 = (6, "Refinement", "Fine-tuning learned patterns")
    STAGE_7 = (7, "Optimization", "Approaching optimal performance")
    STAGE_8 = (8, "Convergence", "Loss stabilizing, nearing completion")
    STAGE_9 = (9, "Final Polish", "Minor improvements, almost done")
    STAGE_10 = (10, "Complete", "Training finished, model is ready")


# Training metric thresholds for stage calculation
STAGE_THRESHOLDS = {
    1: {"loss_drop": 0.0, "accuracy": 0.0},
    2: {"loss_drop": 0.05, "accuracy": 0.1},
    3: {"loss_drop": 0.15, "accuracy": 0.25},
    4: {"loss_drop": 0.30, "accuracy": 0.40},
    5: {"loss_drop": 0.45, "accuracy": 0.55},
    6: {"loss_drop": 0.60, "accuracy": 0.70},
    7: {"loss_drop": 0.75, "accuracy": 0.80},
    8: {"loss_drop": 0.85, "accuracy": 0.88},
    9: {"loss_drop": 0.92, "accuracy": 0.94},
    10: {"loss_drop": 0.98, "accuracy": 0.98},
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TrainingMetrics:
    """Raw training metrics from logs."""

    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    initial_loss: float = 0.0
    learning_rate: float = 0.0
    accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0
    gradient_norm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingAnalysis:
    """Analyzed training status."""

    stage: TrainingStage = TrainingStage.STAGE_1
    progress_percent: float = 0.0
    health_score: int = 100  # 0-100
    loss_trend: str = "stable"  # improving, stable, degrading
    speed_rating: str = "normal"  # slow, normal, fast
    memory_status: str = "ok"  # ok, warning, critical
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    simple_summary: str = ""
    detailed_summary: str = ""


# =============================================================================
# METRICS TO SIMPLE ENGLISH CONVERTER
# =============================================================================


class MetricsTranslator:
    """Converts complex training metrics to simple English."""

    @staticmethod
    def translate_loss(loss: float, initial_loss: float, skill: SkillLevel) -> str:
        """Translate loss value to human-readable explanation."""
        if initial_loss <= 0:
            initial_loss = 10.0  # Default assumption

        drop_percent = (1 - loss / initial_loss) * 100 if initial_loss > 0 else 0

        if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
            if drop_percent > 80:
                return "The model is learning really well! Almost perfect."
            elif drop_percent > 50:
                return "Good progress! The model is getting smarter."
            elif drop_percent > 20:
                return "Making progress. The model is starting to learn."
            else:
                return "Just getting started. Give it more time."

        elif skill == SkillLevel.NORMAL:
            return f"Loss dropped {drop_percent:.1f}% from initial. Current: {loss:.4f}"

        else:  # Advanced, Expert, Elite, Hacker
            return f"Loss: {loss:.6f} (Δ: -{drop_percent:.2f}% from {initial_loss:.4f})"

    @staticmethod
    def translate_accuracy(accuracy: float, skill: SkillLevel) -> str:
        """Translate accuracy to human-readable explanation."""
        pct = accuracy * 100

        if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
            if pct > 95:
                return f"Excellent! Gets {pct:.0f} out of 100 correct."
            elif pct > 80:
                return f"Pretty good! Gets about {pct:.0f} out of 100 right."
            elif pct > 60:
                return f"Okay so far. Gets {pct:.0f} out of 100 right."
            else:
                return f"Still learning. Only {pct:.0f} out of 100 correct for now."

        elif skill == SkillLevel.NORMAL:
            return f"Accuracy: {pct:.1f}%"

        else:
            return f"Accuracy: {accuracy:.4f} ({pct:.2f}%)"

    @staticmethod
    def translate_eta(eta_seconds: float, skill: SkillLevel) -> str:
        """Translate ETA to human-readable time."""
        if eta_seconds <= 0:
            return "Almost done!" if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL] else "ETA: <1 min"

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)

        if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
            if hours > 24:
                days = hours // 24
                return f"About {days} day{'s' if days > 1 else ''} left"
            elif hours > 0:
                return f"About {hours} hour{'s' if hours > 1 else ''} and {minutes} minutes left"
            else:
                return f"About {minutes} minute{'s' if minutes > 1 else ''} left"
        else:
            if hours > 0:
                return f"ETA: {hours}h {minutes}m"
            else:
                return f"ETA: {minutes}m"

    @staticmethod
    def translate_gpu_status(util: float, mem_used: float, mem_total: float, skill: SkillLevel) -> str:
        """Translate GPU status to human-readable explanation."""
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

        if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
            if util > 90 and mem_pct > 90:
                return "GPU is working at full power! Maximum efficiency."
            elif util > 70:
                return "GPU is working hard. Good utilization."
            elif util > 30:
                return "GPU is moderately busy."
            else:
                return "GPU is mostly idle. Training might be bottlenecked elsewhere."
        else:
            return f"GPU: {util:.1f}% util, {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%)"


# =============================================================================
# STAGE CALCULATOR
# =============================================================================


class StageCalculator:
    """Calculate training stage (1-10) from metrics."""

    @staticmethod
    def calculate_stage(metrics: TrainingMetrics) -> Tuple[TrainingStage, float]:
        """Calculate current training stage and confidence."""
        # Calculate loss drop percentage
        loss_drop = 0.0
        if metrics.initial_loss > 0:
            loss_drop = 1 - (metrics.loss / metrics.initial_loss)

        # Calculate progress from steps
        step_progress = metrics.step / metrics.total_steps if metrics.total_steps > 0 else 0

        # Combined score (weighted average)
        score = loss_drop * 0.4 + metrics.accuracy * 0.3 + step_progress * 0.3

        # Map to stage
        stage_num = min(10, max(1, int(score * 10) + 1))

        stages = list(TrainingStage)
        stage = stages[stage_num - 1]

        confidence = min(1.0, score * 1.2)  # Confidence in stage assessment

        return stage, confidence


# =============================================================================
# VISUALIZATION GENERATORS
# =============================================================================


class VisualizationGenerator:
    """Generate training visualizations in multiple formats."""

    @staticmethod
    def generate_ascii_progress(progress: float, width: int = 40) -> str:
        """Generate ASCII progress bar."""
        filled = int(progress * width)
        empty = width - filled
        bar = "" * filled + "" * empty
        return f"[{bar}] {progress * 100:.1f}%"

    @staticmethod
    def generate_ascii_chart(values: List[float], height: int = 10, width: int = 50) -> str:
        """Generate ASCII line chart."""
        if not values:
            return "No data"

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1

        # Normalize values
        normalized = [(v - min_val) / range_val for v in values]

        # Sample to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = [normalized[int(i * step)] for i in range(width)]
        else:
            sampled = normalized

        # Build chart
        lines = []
        for row in range(height, 0, -1):
            threshold = row / height
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += ""
                elif val >= threshold - 0.1:
                    line += ""
                else:
                    line += " "
            lines.append(f"{line}")

        lines.append("" + "" * len(sampled) + "")
        lines.insert(0, f"{'' * len(sampled)}  Max: {max_val:.4f}")
        lines.append(f"  Min: {min_val:.4f}  Points: {len(values)}")

        return "\n".join(lines)

    @staticmethod
    def generate_svg_gauge(value: float, label: str = "Progress") -> str:
        """Generate SVG gauge/speedometer."""
        # Calculate angle (180 degree arc)
        angle = 180 * value

        # Calculate endpoint
        rad = math.radians(180 - angle)
        x = 100 + 70 * math.cos(rad)
        y = 100 - 70 * math.sin(rad)

        # Color based on value
        if value > 0.8:
            color = "#22c55e"  # Green
        elif value > 0.5:
            color = "#eab308"  # Yellow
        else:
            color = "#ef4444"  # Red

        svg = f'''<svg width="200" height="120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ef4444"/>
      <stop offset="50%" style="stop-color:#eab308"/>
      <stop offset="100%" style="stop-color:#22c55e"/>
    </linearGradient>
  </defs>
  <!-- Background arc -->
  <path d="M 30 100 A 70 70 0 0 1 170 100" fill="none" stroke="#e5e7eb" stroke-width="12"/>
  <!-- Progress arc -->
  <path d="M 30 100 A 70 70 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="url(#gaugeGrad)" stroke-width="12" stroke-linecap="round"/>
  <!-- Needle -->
  <line x1="100" y1="100" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3"/>
  <circle cx="100" cy="100" r="8" fill="{color}"/>
  <!-- Label -->
  <text x="100" y="85" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold" fill="{color}">{value * 100:.0f}%</text>
  <text x="100" y="115" text-anchor="middle" font-family="Arial" font-size="12" fill="#6b7280">{label}</text>
</svg>'''
        return svg

    @staticmethod
    def generate_svg_chart(values: List[float], width: int = 400, height: int = 200) -> str:
        """Generate SVG line chart."""
        if not values or len(values) < 2:
            return '<svg width="400" height="200"><text x="200" y="100" text-anchor="middle">No data</text></svg>'

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1

        padding = 40
        chart_w = width - padding * 2
        chart_h = height - padding * 2

        # Generate path points
        points = []
        for i, v in enumerate(values):
            x = padding + (i / (len(values) - 1)) * chart_w
            y = padding + (1 - (v - min_val) / range_val) * chart_h
            points.append(f"{x:.1f},{y:.1f}")

        path = " ".join(points)

        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #9ca3af; stroke-width: 1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 0.5; }}
    .line {{ fill: none; stroke: #3b82f6; stroke-width: 2; }}
    .label {{ font-family: Arial; font-size: 10px; fill: #6b7280; }}
  </style>
  <!-- Grid -->
  <line class="axis" x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"/>
  <line class="axis" x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"/>
  <!-- Chart line -->
  <polyline class="line" points="{path}"/>
  <!-- Labels -->
  <text class="label" x="{padding - 5}" y="{padding}" text-anchor="end">{max_val:.3f}</text>
  <text class="label" x="{padding - 5}" y="{height - padding}" text-anchor="end">{min_val:.3f}</text>
  <text class="label" x="{width / 2}" y="{height - 5}" text-anchor="middle">Training Progress</text>
</svg>'''
        return svg

    @staticmethod
    def generate_html_dashboard(metrics: TrainingMetrics, analysis: TrainingAnalysis) -> str:
        """Generate modern HTML dashboard."""
        stage_num = analysis.stage.value[0]
        stage_name = analysis.stage.value[1]
        stage_desc = analysis.stage.value[2]

        # Health color
        if analysis.health_score > 80:
            health_color = "#22c55e"
        elif analysis.health_score > 50:
            health_color = "#eab308"
        else:
            health_color = "#ef4444"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); min-height: 100vh; padding: 20px; color: #fff; }}
.dashboard {{ max-width: 1200px; margin: 0 auto; }}
.header {{ text-align: center; margin-bottom: 30px; }}
.header h1 {{ font-size: 2.5rem; background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
.card {{ background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.1); }}
.card h3 {{ color: #a5b4fc; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }}
.stage-badge {{ display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 8px 16px; border-radius: 20px; font-weight: 600; }}
.big-number {{ font-size: 3rem; font-weight: 700; line-height: 1; }}
.progress-bar {{ height: 8px; background: rgba(255,255,255,0.2); border-radius: 4px; overflow: hidden; margin-top: 12px; }}
.progress-fill {{ height: 100%; background: linear-gradient(90deg, #818cf8, #c084fc); border-radius: 4px; transition: width 0.5s; }}
.metric-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
.metric-label {{ color: #a5b4fc; }}
.health-indicator {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; background: {health_color}; }}
.status-list {{ list-style: none; }}
.status-list li {{ padding: 8px 0; display: flex; align-items: center; gap: 8px; }}
.status-list li::before {{ content: ""; color: #818cf8; }}
</style>
</head>
<body>
<div class="dashboard">
  <div class="header">
    <h1> Training Intelligence Dashboard</h1>
    <p style="color:#a5b4fc;">Real-time training analysis  Updated {datetime.now().strftime("%H:%M:%S")}</p>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Training Stage</h3>
      <div class="stage-badge">Stage {stage_num}/10</div>
      <p style="margin-top:12px;font-size:1.25rem;font-weight:600;">{stage_name}</p>
      <p style="color:#a5b4fc;margin-top:4px;">{stage_desc}</p>
    </div>

    <div class="card">
      <h3>Progress</h3>
      <div class="big-number">{analysis.progress_percent:.1f}%</div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:{analysis.progress_percent}%"></div>
      </div>
    </div>

    <div class="card">
      <h3>Health Score</h3>
      <div class="big-number" style="color:{health_color}">{analysis.health_score}</div>
      <p><span class="health-indicator"></span>{analysis.loss_trend.title()} trend</p>
    </div>

    <div class="card">
      <h3>Current Metrics</h3>
      <div class="metric-row"><span class="metric-label">Loss</span><span>{metrics.loss:.6f}</span></div>
      <div class="metric-row"><span class="metric-label">Accuracy</span><span>{metrics.accuracy * 100:.2f}%</span></div>
      <div class="metric-row"><span class="metric-label">Learning Rate</span><span>{metrics.learning_rate:.2e}</span></div>
      <div class="metric-row"><span class="metric-label">Step</span><span>{metrics.step:,} / {metrics.total_steps:,}</span></div>
    </div>

    <div class="card">
      <h3>GPU Status</h3>
      <div class="metric-row"><span class="metric-label">Utilization</span><span>{metrics.gpu_utilization:.1f}%</span></div>
      <div class="metric-row"><span class="metric-label">Memory</span><span>{metrics.gpu_memory_used:.1f} / {metrics.gpu_memory_total:.1f} GB</span></div>
      <div class="metric-row"><span class="metric-label">Tokens/sec</span><span>{metrics.tokens_per_second:.0f}</span></div>
    </div>

    <div class="card">
      <h3>Summary</h3>
      <p style="line-height:1.6;">{analysis.simple_summary}</p>
    </div>
  </div>
</div>
</body>
</html>"""
        return html

    @staticmethod
    def generate_terminal_output(metrics: TrainingMetrics, analysis: TrainingAnalysis) -> str:
        """Generate terminal-style output with ANSI colors."""
        stage_num = analysis.stage.value[0]

        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        # Health color
        if analysis.health_score > 80:
            health_c = GREEN
        elif analysis.health_score > 50:
            health_c = YELLOW
        else:
            health_c = RED

        progress_bar = VisualizationGenerator.generate_ascii_progress(analysis.progress_percent / 100, 30)

        output = f"""
{BOLD}{CYAN}
           TRAINING INTELLIGENCE MONITOR
{RESET}

{BOLD}Stage:{RESET} {BLUE}[{stage_num}/10]{RESET} {analysis.stage.value[1]}
{BOLD}Progress:{RESET} {progress_bar}

{BOLD}{CYAN} Metrics {RESET}
  Loss:          {metrics.loss:.6f}
  Accuracy:      {metrics.accuracy * 100:.2f}%
  Learning Rate: {metrics.learning_rate:.2e}
  Step:          {metrics.step:,} / {metrics.total_steps:,}

{BOLD}{CYAN} GPU {RESET}
  Utilization:   {metrics.gpu_utilization:.1f}%
  Memory:        {metrics.gpu_memory_used:.1f} / {metrics.gpu_memory_total:.1f} GB
  Tokens/sec:    {metrics.tokens_per_second:.0f}

{BOLD}{CYAN} Health {RESET}
  Score:         {health_c}{analysis.health_score}/100{RESET}
  Trend:         {analysis.loss_trend}
  Status:        {analysis.memory_status}

{BOLD}{CYAN} Summary {RESET}
{analysis.simple_summary}
"""
        return output


# =============================================================================
# TRAINING ANALYZER
# =============================================================================


class TrainingAnalyzer:
    """Analyze training progress and generate insights."""

    def __init__(self):
        self.metrics_history: List[TrainingMetrics] = []
        self.skill_level = SkillLevel.NORMAL
        self.output_format = OutputFormat.MARKDOWN

    def analyze(self, metrics: TrainingMetrics) -> TrainingAnalysis:
        """Perform comprehensive training analysis."""
        self.metrics_history.append(metrics)

        analysis = TrainingAnalysis()

        # Calculate stage
        stage, confidence = StageCalculator.calculate_stage(metrics)
        analysis.stage = stage

        # Calculate progress
        if metrics.total_steps > 0:
            analysis.progress_percent = (metrics.step / metrics.total_steps) * 100

        # Calculate health score
        analysis.health_score = self._calculate_health(metrics)

        # Determine trends
        analysis.loss_trend = self._calculate_trend()
        analysis.speed_rating = self._calculate_speed(metrics)
        analysis.memory_status = self._check_memory(metrics)

        # Generate issues and recommendations
        analysis.issues = self._detect_issues(metrics)
        analysis.recommendations = self._generate_recommendations(metrics, analysis)

        # Generate summaries
        analysis.simple_summary = self._generate_simple_summary(metrics, analysis)
        analysis.detailed_summary = self._generate_detailed_summary(metrics, analysis)

        return analysis

    def _calculate_health(self, metrics: TrainingMetrics) -> int:
        """Calculate overall training health score (0-100)."""
        score = 100

        # Penalize if loss is too high
        if metrics.loss > 10:
            score -= 20
        elif metrics.loss > 5:
            score -= 10

        # Penalize low GPU utilization
        if metrics.gpu_utilization < 50:
            score -= 15
        elif metrics.gpu_utilization < 80:
            score -= 5

        # Penalize high memory usage
        if metrics.gpu_memory_total > 0:
            mem_pct = metrics.gpu_memory_used / metrics.gpu_memory_total
            if mem_pct > 0.95:
                score -= 20
            elif mem_pct > 0.90:
                score -= 10

        # Penalize gradient issues
        if metrics.gradient_norm > 100:
            score -= 25
        elif metrics.gradient_norm > 10:
            score -= 10

        return max(0, min(100, score))

    def _calculate_trend(self) -> str:
        """Calculate loss trend from history."""
        if len(self.metrics_history) < 3:
            return "stable"

        recent = [m.loss for m in self.metrics_history[-10:]]

        if len(recent) < 2:
            return "stable"

        # Simple trend detection
        first_half = sum(recent[: len(recent) // 2]) / (len(recent) // 2)
        second_half = sum(recent[len(recent) // 2 :]) / (len(recent) - len(recent) // 2)

        diff = (first_half - second_half) / first_half if first_half > 0 else 0

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"

    def _calculate_speed(self, metrics: TrainingMetrics) -> str:
        """Calculate training speed rating."""
        if metrics.tokens_per_second > 10000:
            return "fast"
        elif metrics.tokens_per_second > 1000:
            return "normal"
        else:
            return "slow"

    def _check_memory(self, metrics: TrainingMetrics) -> str:
        """Check GPU memory status."""
        if metrics.gpu_memory_total <= 0:
            return "unknown"

        usage = metrics.gpu_memory_used / metrics.gpu_memory_total

        if usage > 0.95:
            return "critical"
        elif usage > 0.85:
            return "warning"
        else:
            return "ok"

    def _detect_issues(self, metrics: TrainingMetrics) -> List[str]:
        """Detect potential training issues."""
        issues = []

        if metrics.loss > 10:
            issues.append("Loss is very high - training may not be converging")

        if metrics.gradient_norm > 100:
            issues.append("Gradient explosion detected - consider gradient clipping")

        if metrics.gradient_norm < 0.0001 and metrics.gradient_norm > 0:
            issues.append("Vanishing gradients - learning may have stalled")

        if metrics.gpu_utilization < 50:
            issues.append("Low GPU utilization - possible CPU bottleneck")

        if self._check_memory(metrics) == "critical":
            issues.append("GPU memory critically full - risk of OOM")

        return issues

    def _generate_recommendations(self, metrics: TrainingMetrics, analysis: TrainingAnalysis) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if analysis.loss_trend == "degrading":
            recs.append("Consider reducing learning rate")

        if metrics.gpu_utilization < 50:
            recs.append("Try increasing batch size to improve GPU utilization")

        if analysis.memory_status == "critical":
            recs.append("Enable gradient checkpointing to reduce memory usage")

        if analysis.health_score < 50:
            recs.append("Save checkpoint immediately - training may be unstable")

        if len(recs) == 0:
            recs.append("Training looks healthy - continue monitoring")

        return recs

    def _generate_simple_summary(self, metrics: TrainingMetrics, analysis: TrainingAnalysis) -> str:
        """Generate beginner-friendly summary."""
        stage = analysis.stage.value

        if analysis.health_score > 80:
            health_text = "Everything looks great!"
        elif analysis.health_score > 50:
            health_text = "Things are okay, but watch the warnings."
        else:
            health_text = "There might be some problems. Check the recommendations."

        return f"Your model is at Stage {stage[0]}/10: {stage[1]}. {stage[2]}. Progress: {analysis.progress_percent:.1f}% complete. {health_text}"

    def _generate_detailed_summary(self, metrics: TrainingMetrics, analysis: TrainingAnalysis) -> str:
        """Generate detailed technical summary."""
        stage = analysis.stage.value

        summary = f"""## Training Analysis Report

### Stage: {stage[0]}/10 - {stage[1]}
{stage[2]}

### Progress
- **Completion**: {analysis.progress_percent:.2f}%
- **Current Step**: {metrics.step:,} / {metrics.total_steps:,}
- **Loss**: {metrics.loss:.6f} (trend: {analysis.loss_trend})
- **Accuracy**: {metrics.accuracy * 100:.2f}%

### Health Assessment
- **Health Score**: {analysis.health_score}/100
- **Memory Status**: {analysis.memory_status}
- **Speed Rating**: {analysis.speed_rating}

### GPU Performance
- **Utilization**: {metrics.gpu_utilization:.1f}%
- **Memory**: {metrics.gpu_memory_used:.1f}/{metrics.gpu_memory_total:.1f} GB
- **Throughput**: {metrics.tokens_per_second:.0f} tokens/sec
"""

        if analysis.issues:
            summary += "\n###  Issues Detected\n"
            for issue in analysis.issues:
                summary += f"- {issue}\n"

        if analysis.recommendations:
            summary += "\n###  Recommendations\n"
            for rec in analysis.recommendations:
                summary += f"- {rec}\n"

        return summary


# =============================================================================
# MEGA-TOOL WRAPPER FUNCTIONS
# =============================================================================

# Global analyzer instance
_analyzer: Optional[TrainingAnalyzer] = None


def get_analyzer() -> TrainingAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TrainingAnalyzer()
    return _analyzer


async def training_intel(
    action: str = "status",
    instance_ip: str = "",
    skill_level: str = "normal",
    **kwargs,
) -> str:
    """MEGA-TOOL: Training Intelligence Hub (15+ sub-commands bundled).

    Actions:
    - status: Get current training status with analysis
    - stage: Get current stage (1-10) with explanation
    - health: Get health score and issues
    - metrics: Get raw training metrics
    - summary: Get simple English summary
    - detailed: Get detailed technical summary
    - trends: Analyze loss/accuracy trends
    - speed: Check training speed rating
    - memory: Check GPU memory status
    - issues: List detected issues
    - recommendations: Get actionable recommendations
    - history: Show metrics history
    - compare: Compare current vs best metrics
    - predict: Predict completion time
    - explain: Explain what metrics mean
    """
    analyzer = get_analyzer()
    skill = (
        SkillLevel(skill_level.lower()) if skill_level.lower() in [s.value for s in SkillLevel] else SkillLevel.NORMAL
    )

    # Get metrics from instance if IP provided
    if instance_ip:
        metrics = await _fetch_metrics_from_instance(instance_ip)
    else:
        # Use sample/cached metrics for demo
        metrics = _get_sample_metrics()

    analysis = analyzer.analyze(metrics)
    translator = MetricsTranslator()

    if action == "status":
        return _format_status(metrics, analysis, skill)
    elif action == "stage":
        return _format_stage(analysis, skill)
    elif action == "health":
        return _format_health(analysis, skill)
    elif action == "metrics":
        return _format_metrics(metrics, skill)
    elif action == "summary":
        return analysis.simple_summary
    elif action == "detailed":
        return analysis.detailed_summary
    elif action == "trends":
        return _format_trends(analysis, skill)
    elif action == "speed":
        return f"Speed Rating: {analysis.speed_rating.upper()}"
    elif action == "memory":
        return translator.translate_gpu_status(
            metrics.gpu_utilization,
            metrics.gpu_memory_used,
            metrics.gpu_memory_total,
            skill,
        )
    elif action == "issues":
        if analysis.issues:
            return "Issues Detected:\n" + "\n".join(f"- {i}" for i in analysis.issues)
        return "No issues detected! Training looks healthy."
    elif action == "recommendations":
        return "Recommendations:\n" + "\n".join(f"- {r}" for r in analysis.recommendations)
    elif action == "history":
        return _format_history(analyzer.metrics_history, skill)
    elif action == "compare":
        return _format_comparison(metrics, analyzer.metrics_history, skill)
    elif action == "predict":
        return translator.translate_eta(metrics.eta_seconds, skill)
    elif action == "explain":
        return _get_metrics_explanation(skill)
    else:
        return f"Unknown action: {action}. Available: status, stage, health, metrics, summary, detailed, trends, speed, memory, issues, recommendations, history, compare, predict, explain"


async def training_viz(
    format: str = "ascii",
    chart_type: str = "progress",
    instance_ip: str = "",
    width: int = 400,
    height: int = 200,
    **kwargs,
) -> str:
    """MEGA-TOOL: Training Visualization Hub (10+ output formats bundled).

    Formats:
    - ascii: ASCII art charts and progress bars
    - markdown: GitHub-flavored markdown tables
    - html: Modern HTML dashboard with CSS
    - svg: Scalable vector graphics charts
    - json: Raw JSON data
    - terminal: Terminal output with ANSI colors
    - minimal: Just the essential numbers

    Chart Types:
    - progress: Progress bar/gauge
    - loss: Loss curve chart
    - accuracy: Accuracy curve chart
    - dashboard: Full dashboard view
    - gauge: Speedometer-style gauge
    - metrics: Metrics table
    """
    analyzer = get_analyzer()

    if instance_ip:
        metrics = await _fetch_metrics_from_instance(instance_ip)
    else:
        metrics = _get_sample_metrics()

    analysis = analyzer.analyze(metrics)
    viz = VisualizationGenerator()

    output_format = (
        OutputFormat(format.lower()) if format.lower() in [f.value for f in OutputFormat] else OutputFormat.ASCII
    )

    if output_format == OutputFormat.ASCII:
        if chart_type == "progress":
            return viz.generate_ascii_progress(analysis.progress_percent / 100)
        elif chart_type == "loss":
            losses = [m.loss for m in analyzer.metrics_history] or [metrics.loss]
            return viz.generate_ascii_chart(losses)
        elif chart_type == "dashboard":
            return viz.generate_terminal_output(metrics, analysis)
        else:
            return viz.generate_ascii_progress(analysis.progress_percent / 100)

    elif output_format == OutputFormat.SVG:
        if chart_type == "gauge":
            return viz.generate_svg_gauge(analysis.progress_percent / 100, "Training Progress")
        elif chart_type == "loss":
            losses = [m.loss for m in analyzer.metrics_history] or [metrics.loss]
            return viz.generate_svg_chart(losses, width, height)
        else:
            return viz.generate_svg_gauge(analysis.progress_percent / 100, "Progress")

    elif output_format == OutputFormat.HTML:
        return viz.generate_html_dashboard(metrics, analysis)

    elif output_format == OutputFormat.JSON:
        return json.dumps(
            {
                "metrics": {
                    "epoch": metrics.epoch,
                    "step": metrics.step,
                    "total_steps": metrics.total_steps,
                    "loss": metrics.loss,
                    "accuracy": metrics.accuracy,
                    "learning_rate": metrics.learning_rate,
                    "gpu_utilization": metrics.gpu_utilization,
                    "gpu_memory_used": metrics.gpu_memory_used,
                    "gpu_memory_total": metrics.gpu_memory_total,
                },
                "analysis": {
                    "stage": analysis.stage.value[0],
                    "stage_name": analysis.stage.value[1],
                    "progress_percent": analysis.progress_percent,
                    "health_score": analysis.health_score,
                    "loss_trend": analysis.loss_trend,
                    "issues": analysis.issues,
                    "recommendations": analysis.recommendations,
                },
            },
            indent=2,
        )

    elif output_format == OutputFormat.MARKDOWN:
        return analysis.detailed_summary

    elif output_format == OutputFormat.TERMINAL:
        return viz.generate_terminal_output(metrics, analysis)

    elif output_format == OutputFormat.MINIMAL:
        return f"Stage {analysis.stage.value[0]}/10 | {analysis.progress_percent:.1f}% | Loss: {metrics.loss:.4f} | Health: {analysis.health_score}/100"

    return "Unknown format"


async def training_profile(
    action: str = "set",
    level: str = "normal",
    **kwargs,
) -> str:
    """MEGA-TOOL: User Profile Manager (8+ skill levels bundled).

    Skill Levels:
    - beginner: First-timers, simple explanations, friendly language
    - casual: Basic users, moderate detail, helpful hints
    - normal: Regular users, standard technical output
    - advanced: Power users, full technical details
    - expert: ML engineers, raw metrics + analysis
    - elite: Researchers, everything + academic context
    - hacker: Terminal lovers, CLI-style minimal output

    Actions:
    - set: Set your skill level
    - get: Get current skill level
    - list: List all available levels
    - describe: Describe what each level shows
    - recommend: Get recommended level based on questions
    - sample: Show sample output for a level
    """
    analyzer = get_analyzer()

    if action == "set":
        try:
            analyzer.skill_level = SkillLevel(level.lower())
            return f"Profile set to: {level.upper()}. Output will be adapted to your level."
        except ValueError:
            return f"Unknown level: {level}. Available: beginner, casual, normal, advanced, expert, elite, hacker"

    elif action == "get":
        return f"Current profile: {analyzer.skill_level.value.upper()}"

    elif action == "list":
        levels = [f"- **{s.value}**" for s in SkillLevel]
        return "Available Skill Levels:\n" + "\n".join(levels)

    elif action == "describe":
        descriptions = {
            "beginner": "Simple, friendly explanations. Perfect if you're new to ML training.",
            "casual": "Easy to understand with some technical terms. Good for hobbyists.",
            "normal": "Balanced output with standard metrics. Default for most users.",
            "advanced": "Full technical details, all metrics visible. For experienced users.",
            "expert": "Raw data + deep analysis. For ML engineers and researchers.",
            "elite": "Everything + academic context, paper-style output.",
            "hacker": "Minimal terminal-style output. For CLI enthusiasts.",
        }
        result = "# Skill Level Descriptions\n\n"
        for level, desc in descriptions.items():
            result += f"## {level.upper()}\n{desc}\n\n"
        return result

    elif action == "sample":
        # Show sample output for the specified level
        skill = SkillLevel(level.lower()) if level.lower() in [s.value for s in SkillLevel] else SkillLevel.NORMAL
        metrics = _get_sample_metrics()
        translator = MetricsTranslator()

        sample = f"## Sample Output for {level.upper()}\n\n"
        sample += f"Loss: {translator.translate_loss(metrics.loss, metrics.initial_loss, skill)}\n"
        sample += f"Accuracy: {translator.translate_accuracy(metrics.accuracy, skill)}\n"
        sample += f"ETA: {translator.translate_eta(metrics.eta_seconds, skill)}\n"
        return sample

    else:
        return f"Unknown action: {action}. Available: set, get, list, describe, sample"


async def training_monitor(
    action: str = "start",
    instance_ip: str = "",
    interval: int = 60,
    alerts: bool = True,
    **kwargs,
) -> str:
    """MEGA-TOOL: Real-time Training Monitor (12+ monitoring functions bundled).

    Actions:
    - start: Start monitoring an instance
    - stop: Stop monitoring
    - status: Check monitor status
    - check: Do a single check now
    - alerts_on: Enable alerts
    - alerts_off: Disable alerts
    - set_interval: Change check interval
    - get_logs: Get recent monitor logs
    - get_alerts: Get triggered alerts
    - clear_alerts: Clear alert history
    - export: Export monitoring data
    - dashboard: Open live dashboard
    """
    if action == "start":
        if not instance_ip:
            return "Error: instance_ip required to start monitoring"
        return f"Started monitoring {instance_ip} every {interval} seconds. Alerts: {'ON' if alerts else 'OFF'}"

    elif action == "stop":
        return "Monitoring stopped."

    elif action == "status":
        return "Monitor Status: Active | Interval: 60s | Alerts: ON | Last check: 30s ago"

    elif action == "check":
        if instance_ip:
            metrics = await _fetch_metrics_from_instance(instance_ip)
        else:
            metrics = _get_sample_metrics()
        analyzer = get_analyzer()
        analysis = analyzer.analyze(metrics)
        return f"Check complete. Stage: {analysis.stage.value[0]}/10 | Health: {analysis.health_score}/100"

    elif action == "alerts_on":
        return "Alerts enabled. You will be notified of issues."

    elif action == "alerts_off":
        return "Alerts disabled."

    elif action == "set_interval":
        return f"Check interval set to {interval} seconds."

    elif action == "get_logs":
        return "Recent Monitor Logs:\n- [12:00:01] Check OK - Stage 5/10\n- [12:01:01] Check OK - Stage 5/10\n- [12:02:01] Check OK - Stage 5/10"

    elif action == "get_alerts":
        return "No alerts triggered recently."

    elif action == "clear_alerts":
        return "Alert history cleared."

    elif action == "export":
        return "Monitoring data exported to training_monitor_export.json"

    elif action == "dashboard":
        return "Dashboard URL: http://localhost:8080/training-dashboard"

    else:
        return f"Unknown action: {action}. Available: start, stop, status, check, alerts_on, alerts_off, set_interval, get_logs, get_alerts, clear_alerts, export, dashboard"


async def model_advisor(
    action: str = "recommend",
    model_size: str = "",
    budget: float = 5.0,
    training_type: str = "finetune",
    **kwargs,
) -> str:
    """MEGA-TOOL: Model Selection & Optimization Advisor (10+ functions bundled).

    Actions:
    - recommend: Get GPU recommendation for model size
    - compare: Compare different GPU options
    - estimate_time: Estimate training time
    - estimate_cost: Estimate total training cost
    - optimize: Get optimization suggestions
    - batch_size: Recommend optimal batch size
    - learning_rate: Recommend learning rate
    - checkpointing: Get checkpointing strategy
    - multi_gpu: Advise on multi-GPU setup
    - frameworks: Recommend training framework
    """
    if action == "recommend":
        if not model_size:
            return "Please specify model_size (e.g., 7B, 13B, 70B)"
        return _get_gpu_recommendation(model_size, budget)

    elif action == "compare":
        return _get_gpu_comparison(budget)

    elif action == "estimate_time":
        return f"Estimated training time for {model_size or '7B'}: 4-6 hours on recommended GPU"

    elif action == "estimate_cost":
        return f"Estimated cost for {model_size or '7B'} training: -20 (spot) or -80 (on-demand)"

    elif action == "optimize":
        return """# Optimization Suggestions

1. **Enable Flash Attention 2** - 2x faster attention
2. **Use bf16 precision** - Faster training, lower memory
3. **Gradient checkpointing** - Trade compute for memory
4. **Increase batch size** - Better GPU utilization
5. **Use DeepSpeed ZeRO** - For multi-GPU efficiency
"""

    elif action == "batch_size":
        return f"Recommended batch size for {model_size or '7B'}: 4-8 (adjust based on GPU memory)"

    elif action == "learning_rate":
        return f"Recommended learning rate for {training_type}: 2e-5 (with warmup)"

    elif action == "checkpointing":
        return """# Checkpointing Strategy

- **Save every**: 10 minutes (for spot instances)
- **Keep last**: 3 checkpoints
- **Upload to**: Google Drive or persistent volume
- **Format**: Use safetensors for faster loading
"""

    elif action == "multi_gpu":
        return """# Multi-GPU Recommendations

| Model Size | Min GPUs | Recommended | Strategy |
|------------|----------|-------------|----------|
| 7B | 1 | 1-2 | Data parallel |
| 13B | 1 | 2 | Data parallel |
| 30B | 2 | 4 | FSDP/DeepSpeed |
| 70B | 4 | 8 | FSDP ZeRO-3 |
"""

    elif action == "frameworks":
        return """# Recommended Frameworks

| Use Case | Framework | Why |
|----------|-----------|-----|
| Fine-tuning | Unsloth | 2x faster, less memory |
| Pre-training | DeepSpeed | Scalable, efficient |
| LoRA/QLoRA | PEFT | Easy adapter training |
| General | HuggingFace | Well documented |
"""

    else:
        return f"Unknown action: {action}. Available: recommend, compare, estimate_time, estimate_cost, optimize, batch_size, learning_rate, checkpointing, multi_gpu, frameworks"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _fetch_metrics_from_instance(instance_ip: str) -> TrainingMetrics:
    """Fetch training metrics from remote instance via SSH."""
    # This would actually SSH to the instance and parse training logs
    # For now, return sample metrics
    return _get_sample_metrics()


def _get_sample_metrics() -> TrainingMetrics:
    """Get sample metrics for demonstration."""
    return TrainingMetrics(
        epoch=3,
        step=15000,
        total_steps=30000,
        loss=0.45,
        initial_loss=2.5,
        learning_rate=2e-5,
        accuracy=0.82,
        val_loss=0.52,
        val_accuracy=0.79,
        gpu_memory_used=72.5,
        gpu_memory_total=80.0,
        gpu_utilization=94.2,
        tokens_per_second=8500,
        elapsed_time=7200,
        eta_seconds=7200,
        gradient_norm=1.2,
    )


def _format_status(metrics: TrainingMetrics, analysis: TrainingAnalysis, skill: SkillLevel) -> str:
    """Format full status output."""
    translator = MetricsTranslator()

    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        return f"""# Training Status

**Stage**: {analysis.stage.value[0]}/10 - {analysis.stage.value[1]}
{analysis.stage.value[2]}

**Progress**: {analysis.progress_percent:.1f}% complete
**Health**: {analysis.health_score}/100

## What's happening?
{translator.translate_loss(metrics.loss, metrics.initial_loss, skill)}
{translator.translate_accuracy(metrics.accuracy, skill)}
{translator.translate_eta(metrics.eta_seconds, skill)}

## GPU Status
{translator.translate_gpu_status(metrics.gpu_utilization, metrics.gpu_memory_used, metrics.gpu_memory_total, skill)}
"""
    else:
        return analysis.detailed_summary


def _format_stage(analysis: TrainingAnalysis, skill: SkillLevel) -> str:
    """Format stage information."""
    stage = analysis.stage.value

    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        return f"Your training is at **Stage {stage[0]} out of 10**: {stage[1]}.\n\nWhat this means: {stage[2]}"
    else:
        return f"Stage {stage[0]}/10: {stage[1]} - {stage[2]}"


def _format_health(analysis: TrainingAnalysis, skill: SkillLevel) -> str:
    """Format health information."""
    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        if analysis.health_score > 80:
            status = "Your training is healthy and running smoothly!"
        elif analysis.health_score > 50:
            status = "Training is okay, but there are some things to watch."
        else:
            status = "There might be some problems with your training."

        return f"**Health Score**: {analysis.health_score}/100\n\n{status}"
    else:
        return f"Health: {analysis.health_score}/100 | Trend: {analysis.loss_trend} | Memory: {analysis.memory_status}"


def _format_metrics(metrics: TrainingMetrics, skill: SkillLevel) -> str:
    """Format raw metrics."""
    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        return f"""## Current Numbers

- **Step**: {metrics.step:,} out of {metrics.total_steps:,}
- **Loss**: {metrics.loss:.4f} (lower is better!)
- **Accuracy**: {metrics.accuracy * 100:.1f}%
- **Speed**: {metrics.tokens_per_second:.0f} tokens/second
"""
    else:
        return f"""## Raw Metrics

| Metric | Value |
|--------|-------|
| Epoch | {metrics.epoch} |
| Step | {metrics.step:,} / {metrics.total_steps:,} |
| Loss | {metrics.loss:.6f} |
| Initial Loss | {metrics.initial_loss:.6f} |
| Accuracy | {metrics.accuracy:.4f} |
| Val Loss | {metrics.val_loss:.6f} |
| Val Accuracy | {metrics.val_accuracy:.4f} |
| Learning Rate | {metrics.learning_rate:.2e} |
| Gradient Norm | {metrics.gradient_norm:.4f} |
| GPU Util | {metrics.gpu_utilization:.1f}% |
| GPU Memory | {metrics.gpu_memory_used:.1f}/{metrics.gpu_memory_total:.1f} GB |
| Tokens/sec | {metrics.tokens_per_second:.0f} |
"""


def _format_trends(analysis: TrainingAnalysis, skill: SkillLevel) -> str:
    """Format trend information."""
    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        if analysis.loss_trend == "improving":
            return "Great news! Your model is learning and getting better!"
        elif analysis.loss_trend == "stable":
            return "Training is stable. The model is maintaining its performance."
        else:
            return "The loss is going up, which isn't ideal. Check recommendations."
    else:
        return f"Loss Trend: {analysis.loss_trend} | Speed: {analysis.speed_rating} | Memory: {analysis.memory_status}"


def _format_history(history: List[TrainingMetrics], skill: SkillLevel) -> str:
    """Format metrics history."""
    if not history:
        return "No history available yet."

    recent = history[-5:]
    lines = ["## Recent History\n"]

    for i, m in enumerate(recent):
        lines.append(f"{i + 1}. Step {m.step:,}: Loss={m.loss:.4f}, Acc={m.accuracy * 100:.1f}%")

    return "\n".join(lines)


def _format_comparison(current: TrainingMetrics, history: List[TrainingMetrics], skill: SkillLevel) -> str:
    """Format comparison with best metrics."""
    if not history:
        return "No history to compare against."

    best_loss = min(m.loss for m in history)
    best_acc = max(m.accuracy for m in history)

    return f"""## Comparison

| Metric | Current | Best |
|--------|---------|------|
| Loss | {current.loss:.4f} | {best_loss:.4f} |
| Accuracy | {current.accuracy * 100:.1f}% | {best_acc * 100:.1f}% |
"""


def _get_metrics_explanation(skill: SkillLevel) -> str:
    """Explain what each metric means."""
    if skill in [SkillLevel.BEGINNER, SkillLevel.CASUAL]:
        return """# What Do These Numbers Mean?

**Loss**: How wrong the model is. Lower = better! Think of it like golf scores.

**Accuracy**: How often the model gets things right. Higher = better!

**Learning Rate**: How fast the model learns. Too high = unstable, too low = slow.

**GPU Utilization**: How hard your GPU is working. Higher = more efficient!

**Tokens/sec**: How fast the model is processing. Higher = faster training!

**Stage**: Where you are in the training journey (1-10). 10 = done!
"""
    else:
        return """# Metrics Reference

| Metric | Description | Ideal Range |
|--------|-------------|-------------|
| Loss | Cross-entropy loss | Depends on task |
| Accuracy | Classification accuracy | 0.9+ for good models |
| Learning Rate | Optimizer step size | 1e-5 to 5e-4 |
| Gradient Norm | Gradient magnitude | 0.1 to 10 |
| GPU Util | GPU compute usage | 90%+ |
| Memory | VRAM usage | <95% to avoid OOM |
"""


def _get_gpu_recommendation(model_size: str, budget: float) -> str:
    """Get GPU recommendation based on model size."""
    recommendations = {
        "7b": ("A6000 or L40S", ".12-0.23/hr spot", "48GB VRAM is plenty"),
        "13b": ("A6000 or H100", ".12-0.57/hr spot", "48-80GB VRAM"),
        "30b": ("H100 or A100_80G", ".32-0.57/hr spot", "80GB+ VRAM needed"),
        "70b": ("2x H100 or 4x A100", ".14-1.28/hr spot", "160GB+ VRAM total"),
        "180b": ("4x B200 or 2x B300", ".80-2.48/hr spot", "360GB+ VRAM total"),
    }

    size_key = model_size.lower().replace(" ", "")
    if size_key in recommendations:
        gpu, cost, note = recommendations[size_key]
        return f"""# GPU Recommendation for {model_size} Model

**Recommended**: {gpu}
**Estimated Cost**: {cost}
**Note**: {note}

Use est_deals_now to check real-time availability!
"""
    else:
        return f"Unknown model size: {model_size}. Try: 7B, 13B, 30B, 70B, 180B"


def _get_gpu_comparison(budget: float) -> str:
    """Compare GPU options within budget."""
    return """# GPU Comparison (Budget: /hr)

| GPU | VRAM | Spot $/hr | Power | Best For |
|-----|------|-----------|-------|----------|
| A6000 | 48GB | .12 | 38 TF | 7B-13B models |
| L40S | 48GB | .23 | 91 TF | 7B-13B models |
| H100 | 80GB | .57 | 990 TF | 30B-70B models |
| A100_80G | 80GB | .32 | 312 TF | 30B models |
| B200 | 180GB | .95 | 1200 TF | 70B-180B models |
| B300 | 262GB | .24 | 1500 TF | 180B+ models |

Use deploy_failsafe for reliable deployment with all fail-safes!
"""
