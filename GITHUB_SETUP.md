# 🚀 GitHub Setup Instructions

## Option 1: Fork the Original Repo (Recommended)

### Step 1: Fork on GitHub
1. Go to: https://github.com/sniper35/verda-cloud-mcp
2. Click "Fork" button (top right)
3. Name your fork: `verda-cloud-mcp-enhanced`

### Step 2: Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/verda-cloud-mcp-enhanced.git
cd verda-cloud-mcp-enhanced
```

### Step 3: Copy Enhanced Files
Copy these files from this directory to your fork:
- `src/verda_mcp/ssh_tools.py` (NEW)
- `src/verda_mcp/watchdog.py` (NEW)
- `src/verda_mcp/server.py` (MODIFIED)
- `README_ENHANCED.md` → rename to `README.md`
- `.gitignore` (NEW)

### Step 4: Commit and Push
```bash
git add .
git commit -m "Add SSH remote access and WatchDog monitoring features"
git push origin main
```

---

## Option 2: Create New Repo (Fresh Start)

### Step 1: Create on GitHub
1. Go to: https://github.com/new
2. Name: `verda-cloud-mcp-enhanced`
3. Make it public or private
4. Don't initialize with README

### Step 2: Push This Directory
```bash
cd C:\Users\Admin\Downloads\phase2\verda-cloud-mcp

# Remove old origin
git remote remove origin

# Add your new repo
git remote add origin https://github.com/YOUR_USERNAME/verda-cloud-mcp-enhanced.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Verda MCP with SSH and WatchDog"

# Push
git push -u origin main
```

---

## Option 3: Quick Push to New Repo

Run these commands:

```powershell
cd "C:\Users\Admin\Downloads\phase2\verda-cloud-mcp"

# Remove old remote
git remote remove origin

# Add your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/verda-cloud-mcp-enhanced.git

# Add all enhanced files
git add src/verda_mcp/ssh_tools.py
git add src/verda_mcp/watchdog.py
git add src/verda_mcp/server.py
git add README_ENHANCED.md
git add .gitignore
git add GITHUB_SETUP.md

# Commit
git commit -m "feat: Add SSH remote access and WatchDog monitoring

New features:
- SSH remote access (8 tools): run commands, read/write files, view logs
- WatchDog monitoring (5 tools): auto-check training every N minutes
- Timestamped markdown reports

Tools added:
- remote_run_command, remote_gpu_status, remote_training_logs
- remote_training_progress, remote_read_file, remote_write_file
- remote_list_dir, remote_kill_training
- watchdog_enable, watchdog_disable, watchdog_status
- watchdog_latest_report, watchdog_check_now"

# Push
git push -u origin main
```

---

## After GitHub Setup

### For Other Users to Install:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/verda-cloud-mcp-enhanced.git
cd verda-cloud-mcp-enhanced

# Install
pip install -e .
pip install paramiko

# Copy config
cp config.yaml.example config.yaml
# Edit config.yaml with your credentials

# Add to Windsurf MCP config and restart
```

---

## Files in This Enhanced Version

| File | Description |
|------|-------------|
| `src/verda_mcp/server.py` | Main MCP server (enhanced) |
| `src/verda_mcp/ssh_tools.py` | SSH remote access module |
| `src/verda_mcp/watchdog.py` | WatchDog monitoring module |
| `src/verda_mcp/client.py` | Verda API client (original) |
| `src/verda_mcp/config.py` | Configuration handler (original) |
| `config.yaml.example` | Example config file |
| `README_ENHANCED.md` | Enhanced documentation |
| `.gitignore` | Git ignore rules |

---

## Credits

- Original: https://github.com/sniper35/verda-cloud-mcp by sniper35
- Enhancements: SSH tools + WatchDog by Windsurf/Cascade AI
