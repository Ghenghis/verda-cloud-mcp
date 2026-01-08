# 🚀 Verda Cloud MCP Server (Enhanced Edition)

> **Fork of [sniper35/verda-cloud-mcp](https://github.com/sniper35/verda-cloud-mcp)** with SSH remote access and WatchDog monitoring.

## ✨ New Features

### 🔧 SSH Remote Access Tools
Full remote access to your GPU instances via SSH:

| Tool | Description |
|------|-------------|
| `remote_run_command` | Run any bash command on the instance |
| `remote_gpu_status` | Get nvidia-smi output |
| `remote_training_logs` | View training logs (last N lines) |
| `remote_training_progress` | Full status report (GPU, process, logs, disk) |
| `remote_read_file` | Read any file on the instance |
| `remote_write_file` | Write/edit files on the instance |
| `remote_list_dir` | List directory contents |
| `remote_kill_training` | Stop training process |

### 🐕 WatchDog Automatic Monitoring
Automatic training monitoring with timestamped reports:

| Tool | Description |
|------|-------------|
| `watchdog_enable` | Start automatic monitoring (every N minutes) |
| `watchdog_disable` | Stop monitoring and create summary |
| `watchdog_status` | Check if WatchDog is running |
| `watchdog_latest_report` | View the latest report |
| `watchdog_check_now` | Manual one-time check |

**WatchDog automatically:**
- Checks GPU status (nvidia-smi)
- Monitors training process
- Captures recent logs
- Tracks disk space and memory
- Creates timestamped markdown reports
- Alerts if training stops

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Verda Cloud account with API credentials

### Install

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/verda-cloud-mcp-enhanced.git
cd verda-cloud-mcp-enhanced

# Install with pip
pip install -e .

# Install SSH support (required for remote tools)
pip install paramiko
```

### Configure

1. Copy `config.yaml.example` to `config.yaml`
2. Add your Verda API credentials:

```yaml
credentials:
  client_id: "your_client_id"
  client_secret: "your_client_secret"

defaults:
  gpu_type: "B300"
  gpu_count: 1
  location: "FIN-03"
  image: "ubuntu-24.04-cuda-12.8-open-docker"
```

### Add to Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "verda-cloud": {
      "command": "python",
      "args": ["-m", "verda_mcp"],
      "cwd": "C:\\path\\to\\verda-cloud-mcp-enhanced",
      "disabled": false,
      "env": {
        "VERDA_CLIENT_ID": "your_client_id",
        "VERDA_CLIENT_SECRET": "your_client_secret"
      }
    }
  }
}
```

Restart Windsurf after adding.

---

## 🔧 All Available Tools (33 Total)

### Instance Management (Original)
- `list_instances` - List all instances
- `check_instance_status` - Get instance details
- `deploy_spot_instance` - Deploy new instance
- `start_instance` - Start stopped instance
- `shutdown_instance` - Stop running instance
- `delete_instance` - Delete instance

### Availability & Monitoring (Original)
- `check_spot_availability` - Check GPU availability
- `monitor_spot_availability` - Auto-monitor and deploy

### Storage (Original)
- `list_volumes` - List storage volumes
- `create_volume` - Create new volume
- `attach_volume` - Attach volume to instance
- `detach_volume` - Detach volume

### SSH Keys & Scripts (Original)
- `list_ssh_keys` - List SSH keys
- `list_scripts` - List startup scripts
- `create_startup_script` - Create new script
- `set_default_script` - Set default script
- `get_instance_startup_script` - Get instance's script

### Configuration (Original)
- `show_config` - Show current config
- `list_images` - List available images

### SSH Remote Access (NEW!)
- `remote_run_command` - Run bash commands
- `remote_gpu_status` - Get nvidia-smi
- `remote_training_logs` - View logs
- `remote_training_progress` - Full status report
- `remote_read_file` - Read remote files
- `remote_write_file` - Write remote files
- `remote_list_dir` - List directories
- `remote_kill_training` - Kill training

### WatchDog Monitoring (NEW!)
- `watchdog_enable` - Start auto-monitoring
- `watchdog_disable` - Stop monitoring
- `watchdog_status` - Check status
- `watchdog_latest_report` - Get latest report
- `watchdog_check_now` - Manual check

---

## 🐕 WatchDog Usage

### Enable Automatic Monitoring
```
# In Windsurf chat:
"Enable WatchDog for IP 203.0.113.45 every 15 minutes"
```

### What WatchDog Does
1. Connects to your instance via SSH
2. Runs nvidia-smi, checks training process, reads logs
3. Creates a timestamped markdown report
4. Repeats every N minutes
5. Alerts if training stops

### Report Location
Reports are saved to: `~/verda_watchdog_reports/`

Each report includes:
- GPU status
- Training process status
- Recent logs
- Disk space
- Memory usage
- Timestamp

---

## 🔐 SSH Key Setup

Generate an SSH key for Verda:

```powershell
# Windows (PowerShell with Git)
& "C:\Program Files\Git\usr\bin\ssh-keygen.exe" -t ed25519 -f "$env:USERPROFILE\.ssh\verda_key"
```

```bash
# Linux/Mac
ssh-keygen -t ed25519 -f ~/.ssh/verda_key
```

Add the public key to Verda console and the MCP will auto-detect it.

---

## 📊 Example Workflows

### Deploy and Monitor Training
```
1. "Check B300 availability"
2. "Deploy a B300 instance"
3. "Run command on IP: pip install gdown && gdown --folder GDRIVE_LINK"
4. "Run command on IP: python train.py"
5. "Enable WatchDog for IP every 10 minutes"
6. [Go do something else - WatchDog has your back!]
7. "What's the latest WatchDog report?"
8. "Disable WatchDog"
9. "Delete instance"
```

### Debug Training Issues
```
1. "Get GPU status for IP"
2. "Show training logs for IP (last 100 lines)"
3. "Read file /workspace/train.py on IP"
4. "Write file /workspace/fix.py on IP with content: ..."
5. "Kill training on IP"
6. "Run command: python train_fixed.py"
```

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

MIT License - see original repo for details.

---

## 🙏 Credits

- Original MCP Server: [sniper35/verda-cloud-mcp](https://github.com/sniper35/verda-cloud-mcp)
- SSH & WatchDog enhancements: Windsurf + Cascade AI
