# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
