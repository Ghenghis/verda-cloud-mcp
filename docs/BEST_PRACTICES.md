# 🎯 Verda MCP Best Practices Guide

> Complete guide for first-timers and experienced users

---

## 🚀 Quick Start (5 Steps)

### Step 1: Choose Your GPU
```python
# Get recommendation based on model size
recommend_gpu(model_size_billions=7)

# Or find fastest config within budget
fastest_gpu_config(model_size_b=7, budget=5.0)
```

**💡 Tip:** Start with 7B models - great for learning!

### Step 2: ALWAYS Use SPOT (75% Savings!)
```python
# RECOMMENDED deployment method
smart_deploy(gpu_type="A6000", prefer_spot=True)

# Or with all fail-safes
deploy_failsafe(gpu_type="A6000", prefer_spot=True)
```

**💡 Tip:** SPOT instances are identical GPUs at 75% discount!

### Step 3: Enable Checkpoints (CRITICAL!)
```python
# Generate checkpoint script
create_checkpoint_script(checkpoint_minutes=10)
```

**⚠️ WARNING:** SPOT instances can be interrupted. Save every 10 minutes!

### Step 4: Monitor Training
```python
# Simple status in plain English
train_intel(action="status", skill_level="beginner")

# Visual dashboard
train_viz(format="ascii", chart_type="dashboard")
```

### Step 5: Set Budget Alerts
```python
# Alert when spending exceeds $10
set_training_cost_alert(threshold_usd=10)

# Create full budget plan
budget_plan(total_budget=50, model_size_b=7, goal="balanced")
```

---

## ⚡ Multi-GPU Speed Guide

### Scaling Efficiency

| GPUs | Speedup | Efficiency | Cost Multiplier |
|:----:|:-------:|:----------:|:---------------:|
| 1x | 1.0x | 100% | 1x |
| 2x | 1.85x | 92.5% | 2x |
| 4x | 3.5x | 87.5% | 4x |
| 8x | 6.5x | 81.25% | 8x |

### Same Price, More Power!

For the price of **1x B300 On-Demand ($4.95/hr)**:

| Config | Cost | GPUs | Speed |
|--------|------|:----:|:-----:|
| 1x B300 On-Demand | $4.95/hr | 1 | 1x |
| 4x B300 SPOT | $4.96/hr | 4 | **3.5x** |
| 8x H100 SPOT | $4.56/hr | 8 | **6.5x** |
| 5x B200 SPOT | $4.75/hr | 5 | **4.4x** |

**🏆 Key Insight:** Multi-GPU SPOT often beats single On-Demand!

### When to Use Multi-GPU

| Scenario | Recommendation |
|----------|---------------|
| Learning/Testing | 1x A6000 SPOT ($0.12/hr) |
| 7B-13B Production | 2x A6000 SPOT ($0.24/hr) |
| 30B Models | 4x H100 SPOT ($2.28/hr) |
| 70B Models | 4x H200 SPOT ($3.00/hr) |
| Maximum Speed | 8x B300 SPOT ($9.92/hr) |

---

## 💰 Budget Management

### Budget Planning Goals

| Goal | Strategy | Best For |
|------|----------|----------|
| `fastest` | Max GPUs, short time | Deadlines |
| `balanced` | Good speed, fair cost | Most users |
| `budget` | Stretch hours | Learning |
| `best_value` | Max TFLOPs/$ | Production |

### Alert Thresholds

```python
budget_plan(total_budget=100, model_size_b=7, goal="balanced")
```

Output:
```
✅ Start with 4x A6000 SPOT
⏰ Max training time: 208.3 hours
⚠️ Alert at 70%: $70.00 spent
🛑 Auto-stop at 95%: $95.00 spent
💾 Checkpoint every 10 minutes (CRITICAL for spot!)
📊 Monitor with: train_intel(action='status')
```

### Cost Comparison

| GPU | Spot $/hr | On-Demand $/hr | Savings |
|-----|-----------|----------------|:-------:|
| A6000 | $0.12 | $0.49 | 76% |
| H100 | $0.57 | $2.29 | 75% |
| B300 | $1.24 | $4.95 | 75% |

---

## 🛡️ Safety Best Practices

### Checkpoint Strategy

```python
# CRITICAL for SPOT instances!
create_checkpoint_script(
    framework="huggingface",
    checkpoint_minutes=10,  # Save every 10 min
    checkpoint_dir="/workspace/checkpoints"
)
```

**Why 10 minutes?**
- SPOT eviction gives ~2 min warning
- 10 min = max 10 min lost work
- Resume from latest checkpoint automatically

### Volume Strategy

```python
# Always use persistent volume for checkpoints
create_volume(name="my-checkpoints", size=150)
attach_volume(volume_id="vol-xxx", instance_id="inst-xxx")
```

**⚠️ Important:** Instance storage is lost on termination!

### Monitoring

```python
# Enable WatchDog for automatic monitoring
watchdog_enable(instance_ip="1.2.3.4", interval_minutes=10)

# Check health anytime
instance_health_check(instance_ip="1.2.3.4")
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Using On-Demand Instead of SPOT

**Impact:** Paying 4x more for identical GPUs!

**Fix:**
```python
# ALWAYS prefer spot
smart_deploy(gpu_type="B300", prefer_spot=True)
```

### ❌ Mistake 2: No Checkpoints

**Impact:** Lose ALL progress if SPOT interrupted!

**Fix:**
```python
create_checkpoint_script(checkpoint_minutes=10)
```

### ❌ Mistake 3: Wrong GPU for Model Size

**Impact:** Out of memory or wasted resources

**Fix:**
```python
model_size_guide()  # Get recommendations
recommend_gpu(model_size_billions=70)  # For 70B model
```

### ❌ Mistake 4: No Budget Monitoring

**Impact:** Surprise bills!

**Fix:**
```python
set_training_cost_alert(threshold_usd=50)
budget_plan(total_budget=100, model_size_b=7)
```

### ❌ Mistake 5: Using 1 GPU When Multi-GPU is Same Price

**Impact:** Training takes 3-4x longer!

**Fix:**
```python
best_deals_now(budget=5.0, min_vram=48)
power_deals_now(reference_gpu="B300")
```

---

## 📏 Model Size Guide

### GPU Recommendations by Model Size

| Model Size | Min VRAM | Best GPU | SPOT $/hr |
|------------|----------|----------|-----------|
| **7B** | 16GB | A6000 | $0.12 |
| **13B** | 32GB | 2x A6000 | $0.24 |
| **30B** | 64GB | H100 | $0.57 |
| **70B** | 140GB | H200 or 2x H100 | $0.75-$1.14 |
| **180B+** | 360GB+ | 4x B200 | $3.80 |

### VRAM Requirements

```
Model Parameters × 2 bytes (inference)
Model Parameters × 8 bytes (training with gradients)
```

**Example:** 70B model needs ~140GB VRAM for training

---

## 📊 Monitoring Best Practices

### Skill Level Settings

```python
# For beginners - simple English
train_intel(action="status", skill_level="beginner")

# For experts - full technical details
train_intel(action="status", skill_level="expert")
```

### Output Formats

| Format | Best For |
|--------|----------|
| `ascii` | Terminal, SSH |
| `html` | Browser dashboards |
| `json` | Automation, APIs |
| `markdown` | Documentation |

### Key Metrics to Watch

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| GPU Utilization | >90% | 70-90% | <70% |
| Loss Trend | ↓ Decreasing | → Flat | ↑ Increasing |
| Memory Usage | <95% | 95-99% | 100% (OOM) |
| Disk Space | >20% free | 10-20% | <10% |

---

## 🎓 Tips by Category

### 💰 Save Money

- ✅ ALWAYS use SPOT (75% savings!)
- ✅ Multi-GPU SPOT often beats single On-Demand
- ✅ Start with A6000 ($0.12/hr SPOT) for testing
- ✅ Set budget alerts before starting
- ✅ Use checkpoints to resume if interrupted

### ⚡ Go Faster

- ✅ 4x GPUs = 3.5x speed
- ✅ 8x GPUs = 6.5x speed
- ✅ NVLink GPUs (H100, B300) scale better
- ✅ Larger batch sizes = faster training
- ✅ Use mixed precision (fp16/bf16)

### 🛡️ Stay Safe

- ✅ Checkpoint every 10 minutes (MUST for SPOT!)
- ✅ Store checkpoints on persistent volume
- ✅ Enable WatchDog monitoring
- ✅ Set auto-stop budget limits
- ✅ Use `deploy_failsafe()` for production

### 📊 Monitor Well

- ✅ Use `train_intel()` for real-time status
- ✅ Check `train_viz(format='ascii')` for terminal
- ✅ Set `skill_level='beginner'` for simple output
- ✅ Watch loss curve - should go down!
- ✅ GPU utilization should be >90%

---

## 🔧 Quick Reference Commands

### Deployment
```python
smart_deploy(gpu_type="A6000", gpu_count=4, prefer_spot=True)
deploy_failsafe(gpu_type="B300", prefer_spot=True)
```

### Monitoring
```python
train_intel(action="status")
train_viz(format="html", chart_type="dashboard")
watchdog_enable(instance_ip="1.2.3.4")
```

### Budget
```python
budget_plan(total_budget=100, model_size_b=7)
set_training_cost_alert(threshold_usd=50)
current_running_costs()
```

### Speed Optimization
```python
fastest_gpu_config(model_size_b=7, budget=10)
speed_comparison(gpu_type="B300")
best_deals_now(budget=5, min_vram=48)
```

---

**✅ 96 tools + 55 bundled functions = 150+ capabilities!**

*Follow these best practices for successful GPU training!*
