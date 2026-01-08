# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [2.8.0] - 2026-01-08

### 🚀 v2.8.0 - Enhanced File Transfer & Cost Tracking

#### Upload/Download with Progress (NEW)
- `remote(action="upload")` - SFTP upload with speed tracking
- `remote(action="download")` - SFTP download with speed tracking
- 5-minute timeout for large files
- Progress callback support

#### Auto-Retry on Timeout (NEW)
- `ssh_run_with_retry()` - 3 retries with 2s delay
- Automatic recovery from flaky SSH connections
- Logs each retry attempt for debugging

#### Session Cost Tracking Tool (NEW)
- `session_cost(action="start")` - Start tracking (instance_id, gpu_type)
- `session_cost(action="check")` - Show running cost with projections
- `session_cost(action="end")` - End session and show final cost
- `session_cost(action="estimate")` - Project costs for hours
- `session_cost(action="history")` - Show cumulative spend
- Built-in GPU rate lookup (12 GPU types)

### Changed
- Total tools: **28 mega-tools**
- New actions in `remote` tool: upload, download
- New tool: `session_cost`

---

## [2.7.0] - 2026-01-08

### 🛡️ v2.7.0 - Anti-Freeze & Timeout Protection

#### Multi-Layer Timeout System (CRITICAL FIX)
- **SSH Connect Timeout**: 10 seconds
- **SSH Command Timeout**: 15 seconds
- **Async Wrapper Timeout**: 20 seconds
- **MCP Tool Timeout**: 30 seconds (NEW outer layer)

#### @mcp_safe Decorator (NEW)
- `timeout_utils.py` - New utility module
- `@mcp_safe(timeout=N)` - Combined timeout + crash protection
- `@with_timeout(N)` - Pure timeout decorator
- `safe_return()` - Ensures tools never crash server

#### Tools with Timeout Protection
| Tool | Timeout | Status |
|------|---------|--------|
| remote | 30s | ✅ |
| env_setup | 30s | ✅ |
| model_download | 30s | ✅ |
| benchmark | 45s | ✅ |
| checkpoint | 30s | ✅ |
| logs_stream | 30s | ✅ |
| tensorboard | 30s | ✅ |

#### Verbose Logging
- All SSH operations log elapsed time
- All MCP tools show progress
- Connection reuse logged

### Fixed
- SSH tools no longer freeze indefinitely
- MCP server no longer hangs on timeout
- Crash protection catches all exceptions gracefully

---

## [2.6.0] - 2026-01-08

### 🎯 COMPACT EDITION - 21 Mega-Tools

Major consolidation: **104 tools → 21 mega-tools** (80% reduction)

#### Mega-Tools with Action Parameter

Each tool now uses an `action` parameter to access multiple capabilities:

| Tool | Actions | Purpose |
|------|---------|---------|
| `instance` | deploy, list, start, stop, delete, status, check, images | Instance management |
| `volume` | create, list, attach, detach | Storage management |
| `script` | create, list, get, set_default, ssh_keys | Startup scripts |
| `remote` | run, read, write, gpu, logs, kill, progress, list_dir, upload, download | SSH operations |
| `gpu` | catalog, recommend, optimal, best_value, fastest, images | GPU selection |
| `spot` | savings, compare, deals, matrix, budget, recommend | Spot pricing |
| `live` | check, all, costs, now, matrix | Live availability |
| `training` | setup, start, logs, checkpoints, progress, backup, kill, monitor | Training automation |
| `watchdog` | enable, disable, status, check, report | Monitoring |
| `cost` | estimate, running, budget, optimize, compare, history | Cost analytics |
| `health` | instance, api, ssh, full, network | Health checks |
| `deploy` | smart, failsafe, auto | Smart deployment |
| `cluster` | create, batch, status, scale, shared_fs, list_fs | Multi-GPU clusters |
| `gdrive` | file, folder, backup | Google Drive integration |
| `notify` | send, training, budget, setup | Notifications |
| `guide` | start, tips, mistakes, model_size, frameworks, speed | Training guides |
| `model_hub` | list, search, info, download, category, lora, qlora, presets | 50+ models |
| `dataset_hub` | list, download, prepare, tokenize, preview, stats | Dataset management |
| `templates` | list, get, lora, qlora, full, inference | Training templates |
| `distributed` | list, deepspeed, fsdp, accelerate, torchrun, multi_node | Distributed training |
| `train` | intel, viz, profile, monitor, advisor | Training intelligence |

#### Benefits

- **Cleaner UI**: 21 tools instead of 104
- **Same capabilities**: 200+ functions preserved
- **Intuitive**: `action="list"` pattern across all tools
- **Professional**: Organized by domain

---

## [2.5.0] - 2026-01-08

### 🎯 Final Polish Edition - Full Integration & Automation

This release completes the Verda MCP Server with full dashboard integration,
expanded model support, and production-ready features.

#### 🌐 Dashboard API Server (NEW)

- **FastAPI Backend** (`api_server.py`)
  - REST endpoints for all settings (GET/PUT)
  - WebSocket endpoint for real-time updates
  - Settings persistence to `config.yaml`
  - CORS support for browser access
  - Health check and status endpoints

- **Real-time Features**
  - WebSocket connection manager with auto-reconnect
  - Live GPU metrics broadcasting
  - Training progress streaming
  - Instant settings sync

- **API Endpoints**
  - `/api/health` - Health check
  - `/api/status` - Overall status
  - `/api/settings` - All settings (GET/PUT)
  - `/api/settings/providers` - Provider configs
  - `/api/settings/tools` - Tool toggles
  - `/api/settings/training` - Training configs
  - `/api/settings/alerts` - Notification configs
  - `/api/settings/budget` - Budget settings
  - `/api/gpu` - GPU metrics
  - `/api/training` - Training status
  - `/ws` - WebSocket real-time updates

#### 📦 Model Hub Expansion (50+ Models)

- **LLaMA Family**: 3, 3.1, 3.2 (1B to 405B)
- **Mistral Family**: 7B, Nemo 12B, Mixtral 8x7B/8x22B
- **Qwen Family**: 2, 2.5, Coder (0.5B to 72B)
- **DeepSeek**: V2-Lite, Coder V2, V3 (671B MoE)
- **Microsoft Phi**: 3, 3.5 (mini to medium)
- **Google Gemma**: 2 (2B to 27B)
- **Code Models**: CodeLlama, StarCoder2, Qwen2.5-Coder
- **Vision**: SDXL, SD3, FLUX.1-dev/schnell
- **Audio**: Whisper large-v3, v3-turbo
- **Embeddings**: BGE, E5-Mistral, Nomic
- **Multimodal**: LLaVA 1.6, Idefics2

#### 🔧 LoRA/QLoRA Presets (Based on Research)

- **7 LoRA Presets** (Sebastian Raschka's findings):
  - `minimal`: r=8, quick experiments
  - `standard`: r=16, balanced quality/speed
  - `extended`: r=32, all attention layers
  - `full`: r=64, includes lm_head
  - `maximum`: r=256, alpha=512 (best quality)
  - `qlora_4bit`: 75% VRAM reduction
  - `qlora_8bit`: Memory-efficient training

- **New Model Hub Actions**:
  - `lora_config` - Get LoRA config for preset
  - `qlora_config` - Get QLoRA 4-bit config
  - `lora_presets` - List all presets with recommendations
  - `category` - List models by category
  - `stats` - Model hub statistics

#### 🦙 Ollama Models Expansion (25+ Models)

- LLaMA 3, 3.1, 3.2 variants
- Qwen2, Qwen2.5, Qwen2.5-Coder
- Gemma2 (2B, 9B, 27B)
- DeepSeek-Coder, StarCoder2
- Embedding models (nomic, mxbai)
- Multimodal (LLaVA)

#### 🖥️ Live Dashboard UI

- **Enterprise Dashboard** (`dashboard.html`)
  - Compact sidebar with 11 navigation items
  - GPU metrics with real-time updates
  - Training status and progress display
  - Terminal panel for MCP commands
  - SSH panel for remote instance access
  - Jupyter panel for notebook integration
  - Settings panel with providers/tools/alerts

- **Real API Integration**
  - `/api/connect` - Connect to real Verda instance
  - `/api/ssh/exec` - Execute SSH commands
  - `/api/mcp/{command}` - Run MCP tools from UI
  - GPU refresh uses real `nvidia-smi` via SSH
  - Falls back to demo data when disconnected

#### 🧪 Playwright Test Suite

- **Dashboard UI Tests** (`test_dashboard_ui.py`)
  - Tests all UI sections load correctly
  - Validates JavaScript functions exist
  - Tests API health and GPU endpoints
  - 10/10 tests passing

- **Integration Tests** (`test_playwright_integration.py`)
  - 15 module import tests
  - GPU database validation (12 types)
  - Cost calculation tests (75% savings)
  - Training Intelligence validation
  - 21/21 tests passing (100%)

#### 📊 Technical Highlights

- **87 tools** + **140+ bundled functions**
- **50+ models** across 6 categories
- **7 LoRA presets** based on research
- **FastAPI + WebSocket** real-time dashboard
- **Pydantic models** for type-safe configs
- **Settings persistence** to config.yaml
- **100% test pass rate** with Playwright

#### 🔌 New Dependencies (Optional)

```toml
[project.optional-dependencies]
dashboard = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "websockets>=12.0",
    "pydantic>=2.5.0",
]
```

---

## [2.4.0] - 2026-01-08

### 🏢 Enterprise Edition - 7 Mega-Tools (84 Bundled Functions)

This release introduces enterprise-grade features through compact mega-tools that bundle
multiple functions into single tools, optimizing for client performance while maximizing
capabilities.

#### New Mega-Tools

- **Model Hub** (15 functions): HuggingFace, Ollama, LM Studio integration
  - List, search, info for all model sources
  - Download scripts, GPU recommendations
  - Quantization guides, fine-tune/inference scripts

- **Dataset Hub** (12 functions): Download, prepare, validate datasets
  - Popular datasets: Alpaca, Dolly, OpenAssistant, UltraChat, etc.
  - Tokenization, splitting, format conversion
  - Preview, stats, custom dataset creation

- **Notifications** (10 functions): Multi-channel alerts
  - Discord, Slack, Telegram, Email webhooks
  - Training start/complete notifications
  - Custom alerts and event webhooks

- **Cost Analytics** (12 functions): Spending reports & forecasts
  - Daily/weekly/monthly breakdowns
  - By GPU, by project analysis
  - Budget status, optimization tips
  - Spot savings reports

- **Training Templates** (10 functions): Pre-built configs
  - LLaMA, Mistral, CodeLLaMA LoRA/full
  - SDXL, FLUX image generation
  - Whisper speech, BGE/E5 embeddings
  - Ready-to-run training scripts

- **Distributed Training** (10 functions): Multi-node setup
  - DeepSpeed ZeRO-3 configuration
  - PyTorch FSDP setup
  - HuggingFace Accelerate
  - NCCL tuning, troubleshooting

- **Live Dashboard** (15 functions): Professional Web UI
  - Modern glassmorphism dark theme
  - Real-time GPU metrics & charts
  - Training progress visualization
  - Cost tracking, notification center
  - Interactive controls, live logs

#### Technical Highlights

- **103 visible tools** + **139 bundled functions** = **230+ capabilities**
- Optimized for client performance (Windsurf ~50 tool recommendation)
- Enterprise-grade code quality and documentation
- Full Chart.js integration for visualizations
- TailwindCSS-powered responsive dashboard

---

## [2.3.0] - 2026-01-08

### 🚀 Performance Advisor - Speed Optimization & Best Practices

#### Performance Advisor (9 new tools)
- `fastest_gpu_config` - Get FASTEST GPU config within budget (multi-GPU SPOT!)
- `best_value_gpu` - Get BEST VALUE config (TFLOPs per dollar)
- `training_time_calc` - Calculate training time and cost estimates
- `budget_plan` - Create budget management plan with alerts and actions
- `speed_comparison` - Compare 1x, 2x, 4x, 8x GPU speedups
- `first_timer_guide` - Quick start guide for beginners
- `tips_and_tricks` - Pro tips organized by category
- `common_mistakes` - Mistakes to avoid with fixes
- `model_size_guide` - GPU recommendations by model size

#### Multi-GPU Speed Insights
- 4x GPUs = 3.5x speed (87.5% efficiency)
- 8x GPUs = 6.5x speed (81.25% efficiency)
- Multi-GPU SPOT often beats single On-Demand at same price!

#### Budget Management
- Automatic spending alerts (70%, 95% thresholds)
- Action plans with specific commands
- Training goal optimization (fastest, balanced, budget, best_value)

#### First-Timer Support
- 5-step quick start guide
- Common mistakes to avoid
- Model size → GPU recommendations
- Tips organized by category (Save Money, Go Faster, Stay Safe, Monitor Well)

### Changed
- Total visible tools: **96** (was 87)
- New module: `performance_advisor.py`
- Updated tool count in server.py main()

---

## [2.2.0] - 2026-01-07

###  Training Intelligence - Mega-Tool Architecture

#### State-of-the-Art MCP Compaction
- **5 mega-tools bundle 55+ functions** (11:1 compaction ratio)
- Follows MCP best practices with action parameter routing
- Professional, stable implementation

#### Training Intelligence Mega-Tools
- `train_intel` - 15 sub-commands for training analysis
  - status, stage, health, metrics, summary, detailed
  - trends, issues, recommendations, predict, explain
- `train_viz` - 7 output formats
  - ASCII, SVG, HTML, JSON, Markdown, Terminal, Minimal
- `train_profile` - 7 skill levels
  - Beginner, Casual, Normal, Advanced, Expert, Elite, Hacker
- `train_monitor` - 12 monitoring functions
  - start, stop, status, check, alerts, logs, export
- `train_advisor` - 10 advisor functions
  - GPU recommendations, cost estimates, optimization tips

#### 10-Stage Training Rating System
- Stage 1-10 progress tracking
- Simple English explanations for all skill levels
- Real-time metrics to human-readable conversion

#### Visualization Outputs
- ASCII art progress bars and charts
- SVG gauges and line charts
- Modern HTML dashboards with CSS
- Terminal output with ANSI colors

### Changed
- Total visible tools: 87
- Total bundled functions: **140+**
- README updated with mega-tool documentation
- Architecture diagram shows 82 TOOLS

---
## [2.1.0] - 2026-01-07

### 🛡️ Smart Deployer - 7-Layer Fail-Safe System

#### Smart Deployer (4 new tools)
- `best_deals_now` - Find best deals RIGHT NOW with real-time availability check
- `power_deals_now` - Find MORE POWER for SAME/LESS cost (multi-GPU spot vs on-demand)
- `deploy_failsafe` - **RECOMMENDED** - Deploy with ALL 7 fail-safe layers
- `available_now` - Show ALL GPUs available across all locations in real-time

#### 7 Layers of Fail-Safe Protection
1. **Pre-Flight Check** - Check availability BEFORE deploying
2. **Location Fallback** - Try all locations (FIN-01, FIN-02, FIN-03)
3. **Mode Fallback** - Spot → On-Demand auto-switch
4. **Retry Logic** - 3 retry attempts with 10-second delays
5. **Deployment Verification** - Confirm GPU type matches request
6. **Health Monitoring** - Continuous checks every 60 seconds
7. **Eviction & Crash Recovery** - Auto-detect and recover

#### Multi-GPU Spot Power Deals
- Compare 4x B300 SPOT ($4.96/hr) vs 1x B300 On-Demand ($4.95/hr) - 4x power!
- Find configurations with MORE power for SAME or LESS cost
- Value scoring system (Power/Cost ratio)

### Changed
- Total tools: 78 → **82**
- Updated README with comprehensive fail-safe documentation
- Updated pyproject.toml to version 2.1.0

---

## [2.0.0] - 2026-01-07

### 🚀 Major Release - Complete MCP Overhaul

This release transforms the basic Verda MCP into a comprehensive **78-tool powerhouse** with spot optimization, training automation, and live API integration.

### Added

#### ⚡ Spot Manager (6 tools) - SAVE 75%!
- `spot_savings_calculator` - Compare spot vs on-demand pricing
- `smart_deploy` - **RECOMMENDED** - Auto spot-first with on-demand failover
- `switch_to_spot` - Switch running instance to spot mid-training
- `switch_to_ondemand` - Switch to on-demand for stability
- `training_session_status` - View mode, cost, evictions, duration
- `end_training_session` - Stop monitoring when training complete

#### 🎓 Training Tools (7 tools)
- `check_balance` - Check account funds before deploying
- `create_checkpoint_script` - Generate 10-minute checkpoint scripts (PyTorch/HF/Lightning)
- `create_startup_script` - Framework-specific setup scripts
- `list_frameworks` - Show available framework templates
- `set_training_cost_alert` - Alert when cost exceeds threshold
- `notify_training_event` - Send webhook notifications
- `backup_checkpoint_to_gdrive` - Upload checkpoints to Google Drive

#### 🚀 GPU Optimizer (4 tools)
- `spot_vs_ondemand_comparison` - Multi-GPU spot vs 1x on-demand analysis
- `find_optimal_gpu` - Best config for model size and budget
- `estimate_training` - Training time and cost estimates
- `gpu_catalog` - Complete GPU specs and pricing reference

#### 📡 Live API Data (5 tools)
- `live_gpu_availability` - Real-time spot/on-demand status
- `live_all_gpus_availability` - Scan all GPUs across all locations
- `current_running_costs` - Running instance costs breakdown
- `api_capabilities` - What API can/cannot provide
- `refresh_live_data` - Force refresh cached data

#### 🔧 Advanced Beta Tools (7 tools)
- `shared_filesystems` - List shared filesystems
- `create_shared_fs` - Create multi-instance storage
- `gpu_clusters` - List GPU clusters
- `create_gpu_cluster` - Create distributed training cluster
- `batch_jobs` - List batch training jobs
- `create_training_job` - Submit automated batch job
- `batch_job_logs` - Get batch job output logs

#### 📊 Extended Tools (7 tools)
- `cost_estimate` - Estimate training cost
- `analyze_logs` - Parse training logs for metrics
- `instance_health_check` - Comprehensive health check
- `list_checkpoints` - List saved checkpoints
- `backup_checkpoint` - Backup checkpoint locally
- `list_available_gpus` - All GPUs with pricing
- `recommend_gpu` - GPU recommendation for model

#### 🔌 SSH Remote Access (8 tools)
- `remote_run_command` - Execute bash commands on instance
- `remote_gpu_status` - Get nvidia-smi output
- `remote_training_logs` - Get recent training logs
- `remote_training_progress` - Full progress report
- `remote_read_file` - Read files from instance
- `remote_write_file` - Write files to instance
- `remote_list_dir` - List directory contents
- `remote_kill_training` - Kill training processes

#### 📁 Google Drive Integration (6 tools)
- `gdrive_download_file` - Download file to local
- `gdrive_download_folder` - Download folder to local
- `upload_to_verda` - Upload to instance via SCP
- `download_from_verda` - Download from instance via SCP
- `automated_setup` - Full training environment setup
- `automated_start_training` - Start training in screen session

#### 👁️ WatchDog Monitor (5 tools)
- `watchdog_enable` - Start automatic monitoring
- `watchdog_disable` - Stop monitoring
- `watchdog_status` - Get monitoring status
- `watchdog_latest_report` - Get most recent report
- `watchdog_check_now` - Manual check (one-time)

#### 🧪 Testing Tools (3 tools)
- `self_test_api` - Test API and modules
- `self_test_ssh` - Test SSH on instance
- `self_test_all` - Run all tests

### GPU Support
- **12 GPU types** with full multi-GPU configurations (1x, 2x, 4x, 8x)
- Complete SPOT pricing (~75% savings on all GPUs)
- NVLink GPUs: GB300, B300, B200, H200, H100, A100 80G, A100 40G, V100
- General Compute: RTX PRO 6000, L40S, RTX 6000 Ada, A6000

### Framework Templates
- PyTorch with CheckpointManager
- HuggingFace Trainer with SpotCheckpointCallback
- PyTorch Lightning with TimeBasedCheckpoint
- LLaMA/Mistral fine-tuning with PEFT
- Stable Diffusion with Diffusers

### Documentation
- Epic ASCII art README with badges
- Complete tool inventory with examples
- GPU pricing catalog
- Architecture diagrams
- Spot strategy flowcharts

---

## [1.0.0] - 2026-01-01

### Initial Release

#### Core Features (20 tools)
- `list_instances` - List all GPU instances
- `check_instance_status` - Get instance details
- `deploy_spot_instance` - Deploy spot GPU
- `check_spot_availability` - Check availability
- `monitor_spot_availability` - Poll until available
- `start_instance` - Start stopped instance
- `shutdown_instance` - Stop running instance
- `delete_instance` - Delete instance
- `list_volumes` - List storage volumes
- `create_volume` - Create new volume
- `attach_volume` - Attach to instance
- `detach_volume` - Detach from instance
- `list_scripts` - List startup scripts
- `create_startup_script` - Create new script
- `set_default_script` - Set default script
- `get_instance_startup_script` - Get script content
- `list_ssh_keys` - List SSH keys
- `list_images` - List OS images
- `show_config` - Show configuration

#### Infrastructure
- MCP Protocol integration
- Verda SDK wrapper
- YAML configuration
- Claude Desktop support

---

## Version Comparison

| Version | Tools | Features |
|---------|-------|----------|
| 1.0.0 | 20 | Basic instance management |
| **2.0.0** | **78** | Full training automation, spot optimization, live API |

---

[Unreleased]: https://github.com/Ghenghis/verda-cloud-mcp/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Ghenghis/verda-cloud-mcp/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/Ghenghis/verda-cloud-mcp/releases/tag/v1.0.0

