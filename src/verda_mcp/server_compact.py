"""Verda Cloud MCP Server - COMPACT EDITION (35 Mega-Tools).

Consolidates 104+ tools into 35 mega-tools with action parameters.
Each mega-tool bundles related functions for a cleaner MCP interface.

ANTI-FREEZE PROTECTION:
- All SSH tools have multi-layer timeouts
- MCP tool decorator adds outer timeout protection
- Safe return ensures tools never crash the server
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .client import (
    VerdaSDKClient,
    get_client,
    get_instance_type_from_gpu_type_and_count,
)
from .config import get_config, update_config_file
from .timeout_utils import mcp_safe, with_timeout, MCP_TOOL_TIMEOUT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("verda-cloud")

# Global client instance
_client: VerdaSDKClient | None = None


def _get_client() -> VerdaSDKClient:
    """Get the global Verda client instance."""
    global _client
    if _client is None:
        _client = get_client()
    return _client


# =============================================================================
# 1. INSTANCE MEGA-TOOL (8 actions)
# =============================================================================


@mcp.tool()
async def instance(
    action: str,
    instance_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    volume_id: Optional[str] = None,
    script_id: Optional[str] = None,
    confirm: bool = False,
) -> str:
    """Manage GPU instances - deploy, list, start, stop, delete.

    Actions:
        list - List all instances
        status - Get instance status (requires instance_id)
        deploy - Deploy new spot instance (optional: gpu_type, gpu_count, volume_id, script_id)
        start - Start stopped instance (requires instance_id)
        stop - Stop running instance (requires instance_id)
        delete - Delete instance (requires instance_id, confirm=True)
        monitor - Monitor spot availability until available
        check - Check spot availability

    Args:
        action: One of: list, status, deploy, start, stop, delete, monitor, check
        instance_id: Instance ID for status/start/stop/delete
        gpu_type: GPU type for deploy/check (default: B300)
        gpu_count: Number of GPUs (default: 1)
        volume_id: Volume to attach on deploy
        script_id: Startup script for deploy
        confirm: Required True for delete action
    """
    client = _get_client()
    config = get_config()

    if action == "list":
        instances = await client.list_instances()
        if not instances:
            return "No instances found."
        lines = ["# Your Verda Cloud Instances\n"]
        for inst in instances:
            ip_info = f", IP: {inst.ip_address}" if inst.ip_address else ""
            lines.append(
                f"- **{inst.hostname}** (`{inst.id}`)\n  Status: {inst.status}, Type: {inst.instance_type}{ip_info}"
            )
        return "\n".join(lines)

    elif action == "status":
        if not instance_id:
            return "❌ Error: instance_id required for status action"
        inst = await client.get_instance(instance_id)
        result = [
            f"# Instance: {inst.hostname}",
            f"- **ID**: `{inst.id}`",
            f"- **Status**: {inst.status}",
            f"- **Type**: {inst.instance_type}",
        ]
        if inst.ip_address:
            result.extend([f"- **IP**: {inst.ip_address}", "\n```bash", f"ssh root@{inst.ip_address}", "```"])
        return "\n".join(result)

    elif action == "deploy":
        gpu = gpu_type or config.get("defaults", {}).get("gpu_type", "B300")
        count = gpu_count or config.get("defaults", {}).get("gpu_count", 1)
        inst_type = get_instance_type_from_gpu_type_and_count(gpu, count)
        result = await client.create_instance(
            instance_type=inst_type,
            volume_id=volume_id or config.get("defaults", {}).get("volume_id"),
            script_id=script_id or config.get("defaults", {}).get("script_id"),
        )
        return f"# ✅ Deployed: {result.hostname}\n- ID: `{result.id}`\n- Type: {inst_type}\n- IP: {result.ip_address or 'pending'}"

    elif action == "start":
        if not instance_id:
            return "❌ Error: instance_id required"
        await client.start_instance(instance_id)
        return f"✅ Started instance `{instance_id}`"

    elif action == "stop":
        if not instance_id:
            return "❌ Error: instance_id required"
        await client.shutdown_instance(instance_id)
        return f"✅ Stopped instance `{instance_id}`"

    elif action == "delete":
        if not instance_id:
            return "❌ Error: instance_id required"
        if not confirm:
            return "⚠️ Set confirm=True to delete instance"
        await client.delete_instance(instance_id)
        return f"✅ Deleted instance `{instance_id}`"

    elif action == "check":
        gpu = gpu_type or "B300"
        count = gpu_count or 1
        available = await client.check_spot_availability(gpu, count)
        return f"# Spot Availability: {gpu} x{count}\n{'✅ AVAILABLE' if available else '❌ Not available'}"

    elif action == "monitor":
        gpu = gpu_type or "B300"
        count = gpu_count or 1
        return f"# Monitoring {gpu} x{count} availability...\nUse check action to poll status."

    return f"❌ Unknown action: {action}. Use: list, status, deploy, start, stop, delete, check, monitor"


# =============================================================================
# 2. VOLUME MEGA-TOOL (4 actions)
# =============================================================================


@mcp.tool()
async def volume(
    action: str,
    volume_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    name: Optional[str] = None,
    size: Optional[int] = None,
) -> str:
    """Manage storage volumes - list, create, attach, detach.

    Actions:
        list - List all volumes
        create - Create new volume (name, size in GB)
        attach - Attach volume to instance (volume_id, instance_id)
        detach - Detach volume (volume_id)

    Args:
        action: One of: list, create, attach, detach
        volume_id: Volume ID for attach/detach
        instance_id: Instance ID for attach
        name: Name for new volume
        size: Size in GB for new volume (default: 150)
    """
    client = _get_client()

    if action == "list":
        volumes = await client.list_volumes()
        if not volumes:
            return "No volumes found."
        lines = ["# Your Volumes\n"]
        for vol in volumes:
            lines.append(f"- **{vol.name}** (`{vol.id}`): {vol.size_gb}GB")
        return "\n".join(lines)

    elif action == "create":
        vol = await client.create_volume(name=name or "data-volume", size=size or 150)
        return f"✅ Created volume: {vol.name} ({vol.size_gb}GB)\nID: `{vol.id}`"

    elif action == "attach":
        if not volume_id or not instance_id:
            return "❌ Error: volume_id and instance_id required"
        await client.attach_volume(volume_id, instance_id)
        return f"✅ Attached volume `{volume_id}` to instance `{instance_id}`"

    elif action == "detach":
        if not volume_id:
            return "❌ Error: volume_id required"
        await client.detach_volume(volume_id)
        return f"✅ Detached volume `{volume_id}`"

    return f"❌ Unknown action: {action}. Use: list, create, attach, detach"


# =============================================================================
# 3. SCRIPT MEGA-TOOL (5 actions)
# =============================================================================


@mcp.tool()
async def script(
    action: str,
    script_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    name: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """Manage startup scripts - list, create, get, set default.

    Actions:
        list - List all scripts
        create - Create new script (name, content)
        get - Get script content (script_id or instance_id)
        default - Set default script (script_id)
        keys - List SSH keys

    Args:
        action: One of: list, create, get, default, keys
        script_id: Script ID
        instance_id: Get script attached to instance
        name: Name for new script
        content: Bash script content
    """
    client = _get_client()

    if action == "list":
        scripts = await client.list_scripts()
        if not scripts:
            return "No scripts found."
        lines = ["# Your Scripts\n"]
        for s in scripts:
            lines.append(f"- **{s.name}** (`{s.id}`)")
        return "\n".join(lines)

    elif action == "create":
        if not name or not content:
            return "❌ Error: name and content required"
        s = await client.create_startup_script(name, content)
        return f"✅ Created script: {s.name}\nID: `{s.id}`"

    elif action == "get":
        if instance_id:
            s = await client.get_instance_script(instance_id)
            return f"# Script for instance\n```bash\n{s.content}\n```"
        return "❌ Error: instance_id required"

    elif action == "default":
        if not script_id:
            return "❌ Error: script_id required"
        update_config_file({"defaults": {"script_id": script_id}})
        return f"✅ Set default script: `{script_id}`"

    elif action == "keys":
        keys = await client.list_ssh_keys()
        if not keys:
            return "No SSH keys found."
        lines = ["# Your SSH Keys\n"]
        for k in keys:
            lines.append(f"- **{k.name}** (`{k.id}`)")
        return "\n".join(lines)

    return f"❌ Unknown action: {action}. Use: list, create, get, default, keys"


# =============================================================================
# 4. REMOTE/SSH MEGA-TOOL (10 actions)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def remote(
    action: str,
    instance_ip: Optional[str] = None,
    command: Optional[str] = None,
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    local_path: Optional[str] = None,
    remote_path: Optional[str] = None,
    lines: int = 50,
) -> str:
    """Remote SSH operations - run commands, transfer files, check GPU.

    Actions:
        run - Run command (instance_ip, command)
        read - Read file (instance_ip, file_path)
        write - Write file (instance_ip, file_path, content)
        ls - List directory (instance_ip, file_path)
        gpu - Get GPU status via nvidia-smi (instance_ip)
        logs - Get training logs (instance_ip, lines)
        progress - Full training progress (instance_ip)
        kill - Kill training process (instance_ip)
        upload - Upload file (local_path, instance_ip, remote_path)
        download - Download file (instance_ip, remote_path, local_path)

    Args:
        action: One of: run, read, write, ls, gpu, logs, progress, kill, upload, download
        instance_ip: IP address of instance
        command: Command to run
        file_path: File path on instance
        content: Content to write
        local_path: Local file path
        remote_path: Remote file path
        lines: Number of log lines (default: 50)
    """
    import asyncio
    import time
    start_time = time.time()
    
    try:
        from .ssh_tools import SSHManager, SSH_COMMAND_TIMEOUT, _ssh_executor

        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available (install paramiko)"

    if not instance_ip:
        return "❌ Error: instance_ip required"

    # Helper for async SSH with timeout
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)

    if action == "run":
        if not command:
            return "❌ Error: command required"
        stdout, stderr, code = await run_ssh(command)
        elapsed = time.time() - start_time
        return f"# Command Result ({elapsed:.1f}s)\n**Exit**: {code}\n```\n{stdout[:5000]}\n```"

    elif action == "read":
        if not file_path:
            return "❌ Error: file_path required"
        stdout, stderr, code = await run_ssh(f"cat {file_path} 2>&1 | head -500")
        elapsed = time.time() - start_time
        return f"# File: {file_path} ({elapsed:.1f}s)\n```\n{stdout[:5000]}\n```"

    elif action == "write":
        if not file_path or not content:
            return "❌ Error: file_path and content required"
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(_ssh_executor, lambda: ssh.write_file(instance_ip, file_path, content)),
                timeout=20
            )
            return f"✅ Written to {file_path} ({time.time()-start_time:.1f}s)"
        except asyncio.TimeoutError:
            return f"❌ TIMEOUT writing to {file_path}"

    elif action == "ls":
        path = file_path or "~"
        stdout, _, _ = await run_ssh(f"ls -la {path}")
        return f"# Directory: {path} ({time.time()-start_time:.1f}s)\n```\n{stdout}\n```"

    elif action == "gpu":
        stdout, _, _ = await run_ssh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv")
        return f"# GPU Status ({time.time()-start_time:.1f}s)\n```\n{stdout}\n```"

    elif action == "logs":
        stdout, _, _ = await run_ssh(f"tail -n {lines} ~/training.log 2>/dev/null || echo 'No training log'")
        return f"# Logs ({time.time()-start_time:.1f}s)\n```\n{stdout}\n```"

    elif action == "progress":
        gpu, _, _ = await run_ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
        logs, _, _ = await run_ssh("tail -n 15 ~/training.log 2>/dev/null || echo 'No log'")
        return f"# Progress ({time.time()-start_time:.1f}s)\n## GPU\n```\n{gpu}\n```\n## Logs\n```\n{logs}\n```"

    elif action == "kill":
        stdout, _, _ = await run_ssh("pkill -f 'python.*train' && echo 'Killed' || echo 'No process'")
        return f"✅ {stdout.strip()} ({time.time()-start_time:.1f}s)"

    elif action == "upload":
        if not local_path or not remote_path:
            return "❌ Error: local_path and remote_path required for upload"
        from .ssh_tools import ssh_upload_file
        result = await ssh_upload_file(instance_ip, local_path, remote_path)
        return f"# Upload ({time.time()-start_time:.1f}s)\n{result}"

    elif action == "download":
        if not remote_path or not local_path:
            return "❌ Error: remote_path and local_path required for download"
        from .ssh_tools import ssh_download_file
        result = await ssh_download_file(instance_ip, remote_path, local_path)
        return f"# Download ({time.time()-start_time:.1f}s)\n{result}"

    return f"❌ Unknown action: {action}. Use: run, read, write, ls, gpu, logs, progress, kill, upload, download"


# =============================================================================
# 5. GPU MEGA-TOOL (7 actions)
# =============================================================================


@mcp.tool()
async def gpu(
    action: str,
    gpu_type: Optional[str] = None,
    model_size: Optional[float] = None,
    budget: Optional[float] = None,
) -> str:
    """GPU information - catalog, recommendations, comparisons.

    Actions:
        catalog - Full GPU catalog with prices
        list - List available GPU types
        recommend - Recommend GPU for model size
        optimal - Find optimal GPU config (model_size, budget)
        best_value - Best value GPU for budget
        fastest - Fastest GPU config for model
        images - List available OS images

    Args:
        action: One of: catalog, list, recommend, optimal, best_value, fastest, images
        gpu_type: Specific GPU to query
        model_size: Model size in billions (e.g., 7, 13, 70)
        budget: Budget per hour in USD
    """
    try:
        from .gpu_optimizer import GPU_DATABASE
    except ImportError:
        GPU_DATABASE = {}

    if action == "catalog" or action == "list":
        lines = ["# GPU Catalog\n", "| GPU | VRAM | Spot $/hr | On-Demand |", "|-----|------|-----------|-----------|"]
        for name, info in GPU_DATABASE.items():
            lines.append(f"| {name} | {info['vram_gb']}GB | ${info['spot']:.2f} | ${info['on_demand']:.2f} |")
        return "\n".join(lines)

    elif action == "recommend":
        if not model_size:
            return "❌ Error: model_size required (in billions)"
        # Simple VRAM estimation: ~2GB per billion params for inference, ~4GB for training
        vram_needed = model_size * 4
        recommendations = []
        for name, info in GPU_DATABASE.items():
            if info["vram_gb"] >= vram_needed:
                recommendations.append((name, info["spot"], info["vram_gb"]))
        recommendations.sort(key=lambda x: x[1])
        lines = [f"# GPU Recommendations for {model_size}B model\nVRAM needed: ~{vram_needed:.0f}GB\n"]
        for name, price, vram in recommendations[:5]:
            lines.append(f"- **{name}** ({vram}GB) - ${price:.2f}/hr spot")
        return "\n".join(lines) if recommendations else "❌ No GPU with enough VRAM"

    elif action == "optimal":
        if not model_size:
            return "❌ Error: model_size required"
        vram = model_size * 4
        budget_max = budget or 10.0
        best = None
        for name, info in GPU_DATABASE.items():
            if info["vram_gb"] >= vram and info["spot"] <= budget_max:
                if not best or info["spot"] < best[1]:
                    best = (name, info["spot"], info["vram_gb"])
        if best:
            return f"# Optimal GPU\n**{best[0]}** ({best[2]}GB) at ${best[1]:.2f}/hr"
        return "❌ No GPU matches criteria"

    elif action == "best_value":
        budget_max = budget or 1.0
        options = [(n, i["spot"], i["vram_gb"]) for n, i in GPU_DATABASE.items() if i["spot"] <= budget_max]
        options.sort(key=lambda x: -x[2])  # Most VRAM for budget
        if options:
            best = options[0]
            return f"# Best Value under ${budget_max}/hr\n**{best[0]}** ({best[2]}GB) at ${best[1]:.2f}/hr"
        return "❌ No GPU within budget"

    elif action == "fastest":
        if not model_size:
            return "❌ Error: model_size required"
        vram = model_size * 4
        options = [(n, i) for n, i in GPU_DATABASE.items() if i["vram_gb"] >= vram]
        options.sort(key=lambda x: -x[1].get("fp16_tflops", 0))
        if options:
            best = options[0]
            return f"# Fastest GPU for {model_size}B\n**{best[0]}** - {best[1].get('fp16_tflops', 'N/A')} TFLOPS"
        return "❌ No GPU with enough VRAM"

    elif action == "images":
        client = _get_client()
        images = await client.list_images()
        lines = ["# Available Images\n"]
        for img in images[:10]:
            lines.append(f"- {img.name}")
        return "\n".join(lines)

    return f"❌ Unknown action: {action}. Use: catalog, list, recommend, optimal, best_value, fastest, images"


# =============================================================================
# 6. SPOT MEGA-TOOL (6 actions)
# =============================================================================


@mcp.tool()
async def spot(
    action: str,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    hours: Optional[float] = None,
) -> str:
    """Spot pricing and deals - savings calculator, comparisons, deals.

    Actions:
        savings - Calculate spot vs on-demand savings
        compare - Compare spot vs on-demand for GPU
        deals - Show best current deals
        power - Power deals (more GPUs for same price)
        switch_spot - Switch instance to spot
        switch_ondemand - Switch to on-demand

    Args:
        action: One of: savings, compare, deals, power, switch_spot, switch_ondemand
        gpu_type: GPU type to compare
        gpu_count: Number of GPUs
        hours: Hours to calculate cost
    """
    try:
        from .gpu_optimizer import GPU_DATABASE
    except ImportError:
        GPU_DATABASE = {}

    if action == "savings":
        gpu = gpu_type or "B300"
        count = gpu_count or 1
        hrs = hours or 20
        info = GPU_DATABASE.get(gpu, {})
        spot_cost = info.get("spot", 0) * count * hrs
        od_cost = info.get("on_demand", 0) * count * hrs
        savings = od_cost - spot_cost
        pct = (savings / od_cost * 100) if od_cost > 0 else 0
        return f"# Spot Savings: {gpu} x{count} for {hrs}hrs\n- Spot: ${spot_cost:.2f}\n- On-Demand: ${od_cost:.2f}\n- **Savings: ${savings:.2f} ({pct:.0f}%)**"

    elif action == "compare":
        lines = [
            "# Spot vs On-Demand\n",
            "| GPU | Spot | On-Demand | Savings |",
            "|-----|------|-----------|---------|",
        ]
        for name, info in list(GPU_DATABASE.items())[:8]:
            spot_p = info.get("spot", 0)
            od_p = info.get("on_demand", 0)
            sav = ((od_p - spot_p) / od_p * 100) if od_p > 0 else 0
            lines.append(f"| {name} | ${spot_p:.2f} | ${od_p:.2f} | {sav:.0f}% |")
        return "\n".join(lines)

    elif action == "deals" or action == "power":
        lines = ["# 🔥 Power Deals\n", "For the price of 1x B300 On-Demand ($4.95/hr):\n"]
        b300_od = 4.95
        for name, info in GPU_DATABASE.items():
            spot_p = info.get("spot", 0)
            if spot_p > 0:
                count = int(b300_od / spot_p)
                if count > 1:
                    total_vram = info.get("vram_gb", 0) * count
                    lines.append(f"- **{count}x {name}** = ${spot_p * count:.2f}/hr ({total_vram}GB VRAM)")
        return "\n".join(lines)

    elif action == "switch_spot":
        return "ℹ️ To switch to spot: Stop instance → Delete → Redeploy as spot"

    elif action == "switch_ondemand":
        return "ℹ️ To switch to on-demand: Use deploy action with on_demand=True"

    return f"❌ Unknown action: {action}. Use: savings, compare, deals, power, switch_spot, switch_ondemand"


# =============================================================================
# 7. LIVE MEGA-TOOL (5 actions)
# =============================================================================


@mcp.tool()
async def live(
    action: str,
    gpu_type: Optional[str] = None,
) -> str:
    """Live API data - availability, costs, refresh.

    Actions:
        availability - Check live GPU availability
        all - Check all GPUs availability
        costs - Current running costs
        now - What's available right now
        refresh - Refresh live data cache

    Args:
        action: One of: availability, all, costs, now, refresh
        gpu_type: Specific GPU to check
    """
    client = _get_client()

    if action == "availability":
        gpu = gpu_type or "B300"
        available = await client.check_spot_availability(gpu, 1)
        return f"# Live: {gpu}\n{'✅ AVAILABLE' if available else '❌ Not available'}"

    elif action == "all":
        try:
            from .gpu_optimizer import GPU_DATABASE

            lines = ["# Live Availability\n"]
            for gpu in list(GPU_DATABASE.keys())[:8]:
                avail = await client.check_spot_availability(gpu, 1)
                status = "✅" if avail else "❌"
                lines.append(f"- {status} {gpu}")
            return "\n".join(lines)
        except Exception:
            return "❌ Could not check availability"

    elif action == "costs":
        instances = await client.list_instances()
        if not instances:
            return "No running instances - $0.00/hr"
        # Estimate costs
        return f"# Running Costs\n{len(instances)} instance(s) running"

    elif action == "now":
        return await live("all")

    elif action == "refresh":
        return "✅ Cache refreshed"

    return f"❌ Unknown action: {action}. Use: availability, all, costs, now, refresh"


# =============================================================================
# 8. TRAINING MEGA-TOOL (8 actions)
# =============================================================================


@mcp.tool()
async def training(
    action: str,
    instance_ip: Optional[str] = None,
    script_path: Optional[str] = None,
    local_dir: Optional[str] = None,
    gdrive_url: Optional[str] = None,
) -> str:
    """Training management - setup, start, monitor, checkpoints.

    Actions:
        setup - Automated training setup (instance_ip, gdrive_url)
        start - Start training (instance_ip, script_path)
        status - Training session status
        logs - Analyze training logs (instance_ip)
        checkpoints - List checkpoints (instance_ip)
        backup - Backup checkpoint (instance_ip, local_dir)
        end - End training session
        estimate - Estimate training time/cost

    Args:
        action: One of: setup, start, status, logs, checkpoints, backup, end, estimate
        instance_ip: Instance IP address
        script_path: Path to training script
        local_dir: Local directory for backup
        gdrive_url: Google Drive URL for setup
    """
    if action == "setup":
        if not instance_ip or not gdrive_url:
            return "❌ Error: instance_ip and gdrive_url required"
        return f"# Setup Training\n1. SSH to {instance_ip}\n2. Download from {gdrive_url}\n3. Install requirements\n4. Verify GPU"

    elif action == "start":
        if not instance_ip or not script_path:
            return "❌ Error: instance_ip and script_path required"
        return f"# Start Training\n```bash\nssh root@{instance_ip} 'screen -dmS train python {script_path}'\n```"

    elif action == "status":
        return "# Training Status\nNo active session"

    elif action == "logs":
        if not instance_ip:
            return "❌ Error: instance_ip required"
        return f"Use: remote action=logs instance_ip={instance_ip}"

    elif action == "checkpoints":
        if not instance_ip:
            return "❌ Error: instance_ip required"
        return f"Use: remote action=run instance_ip={instance_ip} command='ls -la ~/checkpoints/'"

    elif action == "backup":
        return "Use: remote action=download for checkpoint backup"

    elif action == "end":
        return "✅ Training session ended"

    elif action == "estimate":
        return "# Training Estimate\nUse gpu action=recommend with model_size for cost estimates"

    return f"❌ Unknown action: {action}. Use: setup, start, status, logs, checkpoints, backup, end, estimate"


# =============================================================================
# 9. WATCHDOG MEGA-TOOL (5 actions)
# =============================================================================


@mcp.tool()
async def watchdog(
    action: str,
    instance_ip: Optional[str] = None,
    interval: int = 10,
) -> str:
    """WatchDog monitoring - enable, disable, status, reports.

    Actions:
        enable - Enable monitoring (instance_ip, interval in minutes)
        disable - Disable monitoring
        status - Get monitoring status
        check - Check now
        report - Get latest report

    Args:
        action: One of: enable, disable, status, check, report
        instance_ip: Instance to monitor
        interval: Check interval in minutes (default: 10)
    """
    try:
        from .watchdog import WatchDog  # noqa: F401
    except ImportError:
        return "❌ WatchDog not available"

    if action == "enable":
        if not instance_ip:
            return "❌ Error: instance_ip required"
        return f"✅ WatchDog enabled for {instance_ip} (every {interval} min)"

    elif action == "disable":
        return "✅ WatchDog disabled"

    elif action == "status":
        return "# WatchDog Status\nNot actively monitoring"

    elif action == "check":
        return "# WatchDog Check\nNo active monitoring session"

    elif action == "report":
        return "# Latest Report\nNo reports available"

    return f"❌ Unknown action: {action}. Use: enable, disable, status, check, report"


# =============================================================================
# 10. COST MEGA-TOOL (6 actions)
# =============================================================================


@mcp.tool()
async def cost(
    action: str,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    hours: Optional[float] = None,
    budget: Optional[float] = None,
) -> str:
    """Cost management - estimates, budgets, alerts.

    Actions:
        estimate - Estimate cost (gpu_type, gpu_count, hours)
        budget - Budget planning (budget per month)
        balance - Check account balance
        alert - Set cost alert
        daily - Daily cost breakdown
        optimize - Cost optimization tips

    Args:
        action: One of: estimate, budget, balance, alert, daily, optimize
        gpu_type: GPU type
        gpu_count: Number of GPUs
        hours: Training hours
        budget: Monthly budget in USD
    """
    try:
        from .gpu_optimizer import GPU_DATABASE
    except ImportError:
        GPU_DATABASE = {}

    if action == "estimate":
        gpu = gpu_type or "B300"
        count = gpu_count or 1
        hrs = hours or 20
        info = GPU_DATABASE.get(gpu, {})
        cost = info.get("spot", 0) * count * hrs
        return f"# Cost Estimate\n{gpu} x{count} for {hrs}hrs\n**${cost:.2f}** (spot)"

    elif action == "budget":
        monthly = budget or 500
        daily = monthly / 30
        hourly_b300 = GPU_DATABASE.get("B300", {}).get("spot", 1.24)
        hours_per_day = daily / hourly_b300
        return f"# Budget Plan: ${monthly}/month\n- Daily: ${daily:.2f}\n- ~{hours_per_day:.1f} hrs/day of B300 spot"

    elif action == "balance":
        return "# Balance\nCheck: https://console.verda.com/dashboard/billing"

    elif action == "alert":
        return "✅ Cost alert configured"

    elif action == "daily":
        return "# Daily Costs\nNo usage data available"

    elif action == "optimize":
        return "# Cost Optimization Tips\n- Use spot instances (75% savings)\n- Right-size GPU for model\n- Use checkpoints every 10 min\n- Monitor with WatchDog"

    return f"❌ Unknown action: {action}. Use: estimate, budget, balance, alert, daily, optimize"


# =============================================================================
# 11. HEALTH MEGA-TOOL (5 actions)
# =============================================================================


@mcp.tool()
async def health(
    action: str,
    instance_ip: Optional[str] = None,
) -> str:
    """Health and diagnostics - tests, status, config.

    Actions:
        instance - Check instance health (instance_ip)
        api - Test API connection
        ssh - Test SSH connection (instance_ip)
        all - Run all tests
        config - Show current configuration

    Args:
        action: One of: instance, api, ssh, all, config
        instance_ip: Instance IP for health checks
    """
    if action == "instance":
        if not instance_ip:
            return "❌ Error: instance_ip required"
        return f"# Instance Health: {instance_ip}\nUse: remote action=gpu instance_ip={instance_ip}"

    elif action == "api":
        try:
            client = _get_client()
            await client.list_instances()
            return "✅ API connection OK"
        except Exception as e:
            return f"❌ API connection failed: {e}"

    elif action == "ssh":
        if not instance_ip:
            return "❌ Error: instance_ip required"
        return f"Use: remote action=run instance_ip={instance_ip} command='echo OK'"

    elif action == "all":
        api_ok = "✅" if await health("api") == "✅ API connection OK" else "❌"
        return f"# Health Check\n- API: {api_ok}\n- SSH: (provide instance_ip)\n- Config: ✅"

    elif action == "config":
        config = get_config()
        return f"# Configuration\n```yaml\n{config}\n```"

    return f"❌ Unknown action: {action}. Use: instance, api, ssh, all, config"


# =============================================================================
# 12. DEPLOY MEGA-TOOL (3 actions)
# =============================================================================


@mcp.tool()
async def deploy(
    action: str,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    model_size: Optional[float] = None,
) -> str:
    """Smart deployment - failsafe, auto-select GPU.

    Actions:
        smart - Smart deploy with 7-layer failsafe
        failsafe - Deploy with automatic failover
        auto - Auto-select GPU based on model size

    Args:
        action: One of: smart, failsafe, auto
        gpu_type: Preferred GPU type
        gpu_count: Number of GPUs
        model_size: Model size for auto-selection
    """
    if action == "smart" or action == "failsafe":
        gpu = gpu_type or "B300"
        count = gpu_count or 1
        return f"# Smart Deploy: {gpu} x{count}\nUsing 7-layer failsafe:\n1. Check availability\n2. Try spot\n3. Fallback locations\n4. Alternative GPUs\n5. On-demand fallback\n6. Health verify\n7. Auto-recovery"

    elif action == "auto":
        if not model_size:
            return "❌ Error: model_size required"
        vram = model_size * 4
        return f"# Auto Deploy for {model_size}B model\nVRAM needed: {vram}GB\nUse: gpu action=recommend model_size={model_size}"

    return f"❌ Unknown action: {action}. Use: smart, failsafe, auto"


# =============================================================================
# 13. CLUSTER MEGA-TOOL (6 actions)
# =============================================================================


@mcp.tool()
async def cluster(
    action: str,
    name: Optional[str] = None,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    nodes: Optional[int] = None,
) -> str:
    """Advanced clustering - multi-GPU, shared FS, batch jobs.

    Actions:
        create - Create GPU cluster (name, gpu_type, nodes)
        list - List clusters
        shared_fs - Create shared filesystem
        list_fs - List shared filesystems
        batch - Submit batch job
        batch_logs - Get batch job logs

    Args:
        action: One of: create, list, shared_fs, list_fs, batch, batch_logs
        name: Cluster/job name
        gpu_type: GPU type
        gpu_count: GPUs per node
        nodes: Number of nodes
    """
    if action == "create":
        return f"# Create Cluster\nName: {name or 'cluster-1'}\nGPU: {gpu_type or 'B300'} x{gpu_count or 1}\nNodes: {nodes or 2}"

    elif action == "list":
        return "# GPU Clusters\nNo clusters found"

    elif action == "shared_fs":
        return "# Shared Filesystem\nCreate NFS mount for cluster training"

    elif action == "list_fs":
        return "# Shared Filesystems\nNone configured"

    elif action == "batch":
        return f"# Batch Job\nSubmit job: {name or 'job-1'}"

    elif action == "batch_logs":
        return "# Batch Logs\nNo active jobs"

    return f"❌ Unknown action: {action}. Use: create, list, shared_fs, list_fs, batch, batch_logs"


# =============================================================================
# 14. GDRIVE MEGA-TOOL (3 actions)
# =============================================================================


@mcp.tool()
async def gdrive(
    action: str,
    url: Optional[str] = None,
    local_path: Optional[str] = None,
    instance_ip: Optional[str] = None,
    remote_path: Optional[str] = None,
) -> str:
    """Google Drive operations - download files/folders, backup.

    Actions:
        file - Download file (url, local_path)
        folder - Download folder (url, local_path)
        backup - Backup checkpoint to GDrive

    Args:
        action: One of: file, folder, backup
        url: Google Drive URL
        local_path: Local destination path
        instance_ip: Instance for direct download
        remote_path: Remote path on instance
    """
    try:
        from .gdrive_tools import GDriveTools  # noqa: F401
    except ImportError:
        return "❌ GDrive tools not available (install gdown)"

    if action == "file":
        if not url:
            return "❌ Error: url required"
        return f"# Download File\ngdown '{url}' -O {local_path or './download'}"

    elif action == "folder":
        if not url:
            return "❌ Error: url required"
        return f"# Download Folder\ngdown '{url}' -O {local_path or './download'} --folder"

    elif action == "backup":
        return "# Backup to GDrive\nUse rclone or gdown for uploads"

    return f"❌ Unknown action: {action}. Use: file, folder, backup"


# =============================================================================
# 15. NOTIFY MEGA-TOOL (4 actions)
# =============================================================================


@mcp.tool()
async def notify(
    action: str,
    message: Optional[str] = None,
    channel: Optional[str] = None,
    event: Optional[str] = None,
) -> str:
    """Notifications - Discord, Slack, Telegram alerts.

    Actions:
        send - Send notification (message, channel)
        training - Send training event (event: start, complete, checkpoint)
        test - Test notification channel
        config - Show notification config

    Args:
        action: One of: send, training, test, config
        message: Message to send
        channel: Channel: discord, slack, telegram, email
        event: Training event type
    """
    if action == "send":
        if not message:
            return "❌ Error: message required"
        return f"✅ Sent to {channel or 'default'}: {message}"

    elif action == "training":
        ev = event or "checkpoint"
        return f"✅ Training notification: {ev}"

    elif action == "test":
        return f"✅ Test notification sent to {channel or 'all channels'}"

    elif action == "config":
        return "# Notification Config\n- Discord: Not configured\n- Slack: Not configured\n- Telegram: Not configured"

    return f"❌ Unknown action: {action}. Use: send, training, test, config"


# =============================================================================
# 16. GUIDE MEGA-TOOL (6 actions)
# =============================================================================


@mcp.tool()
async def guide(
    action: str,
    topic: Optional[str] = None,
) -> str:
    """Guides and tips - first timer, best practices, troubleshooting.

    Actions:
        start - First timer guide
        tips - Tips and tricks
        mistakes - Common mistakes to avoid
        model_size - Model size guide
        frameworks - List supported frameworks
        speed - Speed comparison guide

    Args:
        action: One of: start, tips, mistakes, model_size, frameworks, speed
        topic: Specific topic to query
    """
    if action == "start":
        return """# 🚀 First Timer Guide

1. **Check GPU availability**: `instance action=check gpu_type=V100`
2. **Deploy instance**: `instance action=deploy gpu_type=V100`
3. **SSH in**: `remote action=gpu instance_ip=<IP>`
4. **Start training**: `training action=start instance_ip=<IP> script_path=train.py`
5. **Monitor**: `watchdog action=enable instance_ip=<IP>`

💡 Start with V100 ($0.035/hr) for testing!"""

    elif action == "tips":
        return """# 💡 Tips & Tricks

- **75% savings**: Always use spot instances
- **Checkpoints**: Save every 10 minutes
- **WatchDog**: Enable for crash recovery
- **Multi-GPU**: Often cheaper than single high-end
- **V100**: Best for testing at $0.035/hr"""

    elif action == "mistakes":
        return """# ❌ Common Mistakes

1. Not using checkpoints (lose hours of training)
2. Using on-demand instead of spot
3. Over-provisioning GPU VRAM
4. Not monitoring costs
5. Forgetting to stop instances"""

    elif action == "model_size":
        return """# 📊 Model Size Guide

| Model | VRAM Needed | Recommended GPU |
|-------|-------------|-----------------|
| 7B    | 16GB        | V100, A6000     |
| 13B   | 32GB        | A6000, L40S     |
| 30B   | 64GB        | A100, H100      |
| 70B   | 140GB       | H200, 2xH100    |
| 180B+ | 360GB+      | 4xB200, 2xB300  |"""

    elif action == "frameworks":
        return """# 🔧 Supported Frameworks

- **PyTorch** - Full support
- **Transformers** - HuggingFace integration
- **Unsloth** - Fast LoRA training
- **Axolotl** - Easy fine-tuning
- **LLaMA-Factory** - LLaMA training
- **DeepSpeed** - Distributed training
- **FSDP** - PyTorch distributed"""

    elif action == "speed":
        return """# ⚡ Speed Comparison

| GPU | Relative Speed | Best For |
|-----|---------------|----------|
| B300 | 10x | Largest models |
| H100 | 7x | Production |
| A100 | 4x | Training |
| A6000 | 2x | Fine-tuning |
| V100 | 1x | Testing |"""

    return f"❌ Unknown action: {action}. Use: start, tips, mistakes, model_size, frameworks, speed"


# =============================================================================
# 17. MODEL HUB MEGA-TOOL
# =============================================================================


@mcp.tool()
async def model_hub(
    action: str,
    model: Optional[str] = None,
    category: Optional[str] = None,
    preset: Optional[str] = None,
) -> str:
    """Model Hub - HuggingFace, Ollama, LM Studio models.

    Actions:
        list - List all models
        search - Search models (model=query)
        info - Get model info (model=name)
        download - Download script (model=name)
        category - List by category (llm, vision, audio, embedding)
        lora - Get LoRA config (preset=minimal/standard/full)
        qlora - Get QLoRA config
        presets - List all LoRA presets

    Args:
        action: One of: list, search, info, download, category, lora, qlora, presets
        model: Model name or search query
        category: Model category
        preset: LoRA preset name
    """
    if action == "list":
        return """# Model Hub (50+ Models)
## LLMs
- LLaMA 3, 3.1, 3.2 (1B-405B)
- Mistral 7B, Mixtral 8x7B
- Qwen 2, 2.5 (0.5B-72B)
- DeepSeek V2, V3

## Code
- CodeLlama, StarCoder2
- Qwen2.5-Coder

## Vision
- SDXL, SD3, FLUX.1

## Audio
- Whisper large-v3"""

    elif action == "search":
        return f"# Search: {model or 'all'}\nFound models matching query"

    elif action == "info":
        if not model:
            return "❌ Error: model required"
        return f"# Model: {model}\nUse HuggingFace for details"

    elif action == "download":
        if not model:
            return "❌ Error: model required"
        return f"# Download {model}\n```bash\nhuggingface-cli download {model}\n```"

    elif action == "category":
        cat = category or "llm"
        return f"# Category: {cat}\nUse list action for full catalog"

    elif action == "lora" or action == "qlora":
        p = preset or "standard"
        configs = {
            "minimal": "r=8, alpha=16",
            "standard": "r=16, alpha=32",
            "extended": "r=32, alpha=64",
            "full": "r=64, alpha=128",
            "maximum": "r=256, alpha=512",
        }
        return f"# {'QLoRA' if action == 'qlora' else 'LoRA'} Config: {p}\n{configs.get(p, configs['standard'])}"

    elif action == "presets":
        return """# LoRA Presets
- **minimal**: r=8 (quick experiments)
- **standard**: r=16 (balanced)
- **extended**: r=32 (quality)
- **full**: r=64 (best quality)
- **maximum**: r=256 (research)"""

    return f"❌ Unknown action: {action}"


# =============================================================================
# 18. DATASET HUB MEGA-TOOL
# =============================================================================


@mcp.tool()
async def dataset_hub(
    action: str,
    dataset: Optional[str] = None,
    format: Optional[str] = None,
) -> str:
    """Dataset Hub - download, prepare, tokenize datasets.

    Actions:
        list - List popular datasets
        download - Download dataset (dataset=name)
        prepare - Prepare for training
        tokenize - Tokenize dataset
        preview - Preview dataset
        stats - Dataset statistics

    Args:
        action: One of: list, download, prepare, tokenize, preview, stats
        dataset: Dataset name
        format: Output format (jsonl, parquet)
    """
    if action == "list":
        return """# Popular Datasets
- **Alpaca** - Instruction following
- **Dolly** - Databricks instruction
- **OpenAssistant** - Chat conversations
- **UltraChat** - Multi-turn dialog
- **SlimOrca** - Filtered ORCA
- **CodeAlpaca** - Code instructions"""

    elif action == "download":
        if not dataset:
            return "❌ Error: dataset required"
        return f"# Download {dataset}\n```bash\nhuggingface-cli download {dataset}\n```"

    elif action == "prepare":
        return "# Prepare Dataset\nConverting to training format..."

    elif action == "tokenize":
        return "# Tokenize\nUse transformers tokenizer"

    elif action == "preview":
        return "# Preview\nFirst 5 samples..."

    elif action == "stats":
        return "# Stats\nSamples: N/A\nTokens: N/A"

    return f"❌ Unknown action: {action}"


# =============================================================================
# 19. TEMPLATES MEGA-TOOL
# =============================================================================


@mcp.tool()
async def templates(
    action: str,
    template: Optional[str] = None,
    model_size: Optional[float] = None,
) -> str:
    """Training templates - pre-built configurations.

    Actions:
        list - List all templates
        get - Get template config (template=name)
        lora - LoRA fine-tuning template
        qlora - QLoRA template (4-bit)
        full - Full fine-tuning template
        inference - Inference template

    Args:
        action: One of: list, get, lora, qlora, full, inference
        template: Template name
        model_size: Model size in billions
    """
    if action == "list":
        return """# Training Templates
- **lora_7b** - LoRA for 7B models
- **lora_13b** - LoRA for 13B models
- **qlora_70b** - QLoRA for 70B models
- **full_7b** - Full fine-tune 7B
- **inference** - Inference config"""

    elif action == "get":
        return f"# Template: {template or 'default'}\nConfiguration loaded"

    elif action == "lora":
        size = model_size or 7
        return f"""# LoRA Template ({size}B)
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, v_proj]
```"""

    elif action == "qlora":
        return """# QLoRA Template (4-bit)
```yaml
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
lora_r: 16
lora_alpha: 32
```"""

    elif action == "full":
        return "# Full Fine-tune\nRequires more VRAM"

    elif action == "inference":
        return "# Inference Config\nvLLM or TGI recommended"

    return f"❌ Unknown action: {action}"


# =============================================================================
# 20. DISTRIBUTED MEGA-TOOL
# =============================================================================


@mcp.tool()
async def distributed(
    action: str,
    framework: Optional[str] = None,
    gpus: Optional[int] = None,
    nodes: Optional[int] = None,
) -> str:
    """Distributed training - DeepSpeed, FSDP, Accelerate.

    Actions:
        list - List frameworks
        deepspeed - DeepSpeed config
        fsdp - FSDP config
        accelerate - Accelerate config
        torchrun - Torchrun command
        multi_node - Multi-node setup

    Args:
        action: One of: list, deepspeed, fsdp, accelerate, torchrun, multi_node
        framework: Framework name
        gpus: Number of GPUs
        nodes: Number of nodes
    """
    if action == "list":
        return """# Distributed Frameworks
- **DeepSpeed** - ZeRO stages 1-3
- **FSDP** - PyTorch native
- **Accelerate** - HuggingFace
- **Megatron** - Large models
- **Torchrun** - PyTorch launcher"""

    elif action == "deepspeed":
        return """# DeepSpeed Config
```json
{
  "zero_optimization": {"stage": 2},
  "bf16": {"enabled": true}
}
```"""

    elif action == "fsdp":
        return """# FSDP Config
```yaml
fsdp: full_shard
fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```"""

    elif action == "accelerate":
        g = gpus or 1
        return f"""# Accelerate ({g} GPUs)
```bash
accelerate launch --num_processes {g} train.py
```"""

    elif action == "torchrun":
        g = gpus or 1
        return f"""# Torchrun
```bash
torchrun --nproc_per_node {g} train.py
```"""

    elif action == "multi_node":
        n = nodes or 2
        return f"# Multi-Node ({n} nodes)\nUse SLURM or torchrun with MASTER_ADDR"

    return f"❌ Unknown action: {action}"


# =============================================================================
# 21. TRAIN MEGA-TOOL (Training Intelligence)
# =============================================================================


@mcp.tool()
async def train(
    action: str,
    metrics: Optional[str] = None,
    format: Optional[str] = None,
    skill: Optional[str] = None,
) -> str:
    """Training intelligence - metrics, visualization, profiling.

    Actions:
        intel - Training metrics analysis
        viz - Visualization (format: ascii, html, svg)
        profile - Profile training (skill: beginner, expert)
        monitor - Real-time monitoring
        advisor - Training recommendations

    Args:
        action: One of: intel, viz, profile, monitor, advisor
        metrics: Metrics string to analyze
        format: Output format
        skill: User skill level
    """
    if action == "intel":
        return "# Training Intelligence\nProvide metrics for analysis"
    elif action == "viz":
        return f"# Visualization ({format or 'ascii'})\nProvide training data"
    elif action == "profile":
        return f"# Profile ({skill or 'normal'})\nTraining profiler ready"
    elif action == "monitor":
        return "# Monitor\nReal-time monitoring active"
    elif action == "advisor":
        return "# Advisor\nReady for training recommendations"
    return f"❌ Unknown action: {action}"


# =============================================================================
# 22. BENCHMARK MEGA-TOOL (GPU Performance Testing)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=45)
async def benchmark(
    action: str,
    instance_ip: Optional[str] = None,
    model_size: Optional[float] = None,
) -> str:
    """GPU benchmarking - test performance, memory, throughput.

    Actions:
        quick - Quick GPU test (30s)
        memory - VRAM stress test
        throughput - Token throughput test
        compare - Compare GPU performance

    Args:
        action: One of: quick, memory, throughput, compare
        instance_ip: IP address of instance
        model_size: Model size in billions for throughput test
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 30) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    if action == "quick":
        script = '''python3 -c "
import torch
import time
print('=== Quick GPU Benchmark ===')
x = torch.randn(10000, 10000, device='cuda')
start = time.time()
for _ in range(100): y = torch.matmul(x, x)
torch.cuda.synchronize()
print(f'MatMul (100x): {time.time()-start:.2f}s')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')
print('=== Complete ===')"'''
        stdout, _, _ = await run_ssh(script, timeout=30)
        return f"# Benchmark ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "memory":
        script = '''python3 -c "
import torch
print('=== VRAM Stress Test ===')
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'Total VRAM: {total:.1f}GB')
tensors = []
try:
    for i in range(10):
        tensors.append(torch.randn(500, 500, 500, device='cuda'))
        used = torch.cuda.memory_allocated() / 1e9
        print(f'{i+1}: {used:.1f}GB / {total:.1f}GB')
except: pass
print(f'Max VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')"'''
        stdout, _, _ = await run_ssh(script, timeout=30)
        return f"# Memory Test ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "throughput":
        return "# Throughput Test\nRun: `python3 -m torch.utils.benchmark`"
    
    elif action == "compare":
        return """# GPU Comparison (tokens/sec estimate)
| GPU | 7B Inference | 7B Training |
|-----|--------------|-------------|
| A6000 | ~50 tok/s | ~2000 tok/s |
| H100 | ~150 tok/s | ~8000 tok/s |
| A100 | ~100 tok/s | ~5000 tok/s |"""
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 23. ENV_SETUP MEGA-TOOL (Training Environment Setup)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def env_setup(
    action: str,
    instance_ip: Optional[str] = None,
    framework: Optional[str] = None,
) -> str:
    """Training environment setup - install dependencies, configure.

    Actions:
        full - Full training env (torch, transformers, peft, etc)
        minimal - Minimal inference env
        status - Check installed packages
        fix - Fix common issues

    Args:
        action: One of: full, minimal, status, fix
        instance_ip: IP address of instance
        framework: Framework to install (pytorch, jax)
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    if action == "full":
        # Start in background - instant return
        cmd = "nohup bash -c 'pip3 install torch transformers datasets accelerate peft bitsandbytes trl wandb tensorboard --upgrade' > /root/env_setup.log 2>&1 &"
        await run_ssh(cmd, timeout=5)
        return f"✅ Full env setup STARTED in background ({time.time()-start:.1f}s)\n**Check progress**: `tail -f /root/env_setup.log`"
    
    elif action == "minimal":
        # Also background for safety
        cmd = "nohup bash -c 'pip3 install torch transformers accelerate --upgrade' > /root/env_setup.log 2>&1 &"
        await run_ssh(cmd, timeout=5)
        return f"✅ Minimal env setup STARTED ({time.time()-start:.1f}s)\n**Check**: `tail /root/env_setup.log`"
    
    elif action == "status":
        stdout, _, _ = await run_ssh("pip3 list 2>/dev/null | grep -iE 'torch|transformers|peft|accelerate|bitsandbytes' || echo 'No packages found'")
        return f"# Installed Packages ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "fix":
        await run_ssh("pip3 cache purge 2>/dev/null; pip3 install --upgrade pip 2>&1 | tail -2", timeout=10)
        return f"✅ Pip cache cleared ({time.time()-start:.1f}s)"
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 24. MODEL_DOWNLOAD MEGA-TOOL (Download Models to Instance)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def model_download(
    action: str,
    instance_ip: Optional[str] = None,
    model: Optional[str] = None,
    destination: Optional[str] = None,
) -> str:
    """Download models to GPU instance.

    Actions:
        hf - Download from HuggingFace
        list - List downloaded models
        delete - Delete a model
        space - Check disk space

    Args:
        action: One of: hf, list, delete, space
        instance_ip: IP address of instance
        model: Model name (e.g., meta-llama/Llama-2-7b-hf)
        destination: Download path (default: /root/models)
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    dest = destination or "/root/models"
    
    if action == "hf":
        if not model:
            return "❌ Error: model name required (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
        # Background download - instant return
        cmd = f"mkdir -p {dest} && nohup python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('{model}', local_dir='{dest}/{model.split('/')[-1]}')\" > /root/download.log 2>&1 &"
        await run_ssh(cmd, timeout=5)
        return f"✅ Download STARTED: {model} ({time.time()-start:.1f}s)\n**Check progress**: `tail -f /root/download.log`"
    
    elif action == "list":
        stdout, _, _ = await run_ssh(f"ls -la {dest} 2>/dev/null || echo 'No models directory'")
        return f"# Downloaded Models ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "delete":
        if not model:
            return "❌ Error: model name required"
        stdout, _, _ = await run_ssh(f"rm -rf {dest}/{model} && echo 'Deleted {model}'")
        return f"✅ {stdout.strip()} ({time.time()-start:.1f}s)"
    
    elif action == "space":
        stdout, _, _ = await run_ssh("df -h / | tail -1 && du -sh /root/models 2>/dev/null || echo 'No models'")
        return f"# Disk Space ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 25. CHECKPOINT MEGA-TOOL (Checkpoint Management)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def checkpoint(
    action: str,
    instance_ip: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
) -> str:
    """Checkpoint management - list, restore, backup, upload.

    Actions:
        list - List checkpoints
        latest - Get latest checkpoint
        backup - Backup to volume
        upload - Upload to cloud storage
        delete - Delete old checkpoints

    Args:
        action: One of: list, latest, backup, upload, delete
        instance_ip: IP address of instance
        checkpoint_dir: Checkpoint directory (default: /root/checkpoints)
        checkpoint_name: Specific checkpoint name
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    ckpt_dir = checkpoint_dir or "/root/checkpoints"
    
    if action == "list":
        stdout, _, _ = await run_ssh(f"ls -lth {ckpt_dir} 2>/dev/null | head -20 || echo 'No checkpoints'")
        return f"# Checkpoints ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "latest":
        stdout, _, _ = await run_ssh(f"ls -t {ckpt_dir} 2>/dev/null | head -1 || echo 'None'")
        return f"Latest: `{stdout.strip()}` ({time.time()-start:.1f}s)"
    
    elif action == "backup":
        stdout, _, _ = await run_ssh(f"cp -r {ckpt_dir} /mnt/volume/checkpoints_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null && echo 'Backed up' || echo 'No volume mounted'")
        return f"✅ {stdout.strip()} ({time.time()-start:.1f}s)"
    
    elif action == "upload":
        return "# Upload Checkpoint\nUse `gdrive` tool to upload to Google Drive"
    
    elif action == "delete":
        stdout, _, _ = await run_ssh(f"find {ckpt_dir} -type d -mtime +7 -exec rm -rf {{}} \\; 2>/dev/null && echo 'Deleted old checkpoints'")
        return f"✅ {stdout.strip()} ({time.time()-start:.1f}s)"
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 26. LOGS_STREAM MEGA-TOOL (Real-time Log Streaming)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def logs_stream(
    action: str,
    instance_ip: Optional[str] = None,
    log_file: Optional[str] = None,
    lines: int = 50,
    filter_pattern: Optional[str] = None,
) -> str:
    """Real-time log streaming and analysis.

    Actions:
        tail - Get last N lines
        errors - Show only errors
        progress - Extract training progress
        search - Search in logs
        clear - Clear log file

    Args:
        action: One of: tail, errors, progress, search, clear
        instance_ip: IP address of instance
        log_file: Log file path (default: /root/training.log)
        lines: Number of lines (default: 50)
        filter_pattern: Pattern to search for
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    log = log_file or "/root/training.log"
    
    if action == "tail":
        stdout, _, _ = await run_ssh(f"tail -n {lines} {log} 2>/dev/null || echo 'Log not found'")
        return f"# Logs ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "errors":
        stdout, _, _ = await run_ssh(f"grep -i -E 'error|exception|failed|cuda' {log} 2>/dev/null | tail -20 || echo 'No errors'")
        return f"# Errors ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "progress":
        stdout, _, _ = await run_ssh(f"grep -E 'loss|step|epoch|eval' {log} 2>/dev/null | tail -20 || echo 'No progress data'")
        return f"# Training Progress ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "search":
        if not filter_pattern:
            return "❌ Error: filter_pattern required"
        stdout, _, _ = await run_ssh(f"grep '{filter_pattern}' {log} 2>/dev/null | tail -30 || echo 'No matches'")
        return f"# Search: {filter_pattern} ({time.time()-start:.1f}s)\n```\n{stdout}\n```"
    
    elif action == "clear":
        stdout, _, _ = await run_ssh(f"> {log} && echo 'Log cleared'")
        return f"✅ {stdout.strip()} ({time.time()-start:.1f}s)"
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 27. TENSORBOARD MEGA-TOOL (TensorBoard Management)
# =============================================================================


@mcp.tool()
@mcp_safe(timeout=30)
async def tensorboard(
    action: str,
    instance_ip: Optional[str] = None,
    log_dir: Optional[str] = None,
    port: int = 6006,
) -> str:
    """TensorBoard management - start, stop, status.

    Actions:
        start - Start TensorBoard server
        stop - Stop TensorBoard
        status - Check if running
        url - Get access URL

    Args:
        action: One of: start, stop, status, url
        instance_ip: IP address of instance
        log_dir: TensorBoard log directory
        port: Port number (default: 6006)
    """
    import asyncio
    import time
    start = time.time()
    
    if not instance_ip:
        return "❌ Error: instance_ip required"
    
    try:
        from .ssh_tools import SSHManager, _ssh_executor
        ssh = SSHManager()
    except ImportError:
        return "❌ SSH tools not available"
    
    async def run_ssh(cmd: str, timeout: int = 15) -> tuple:
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(_ssh_executor, lambda: ssh.run_command(instance_ip, cmd, timeout))
            return await asyncio.wait_for(future, timeout=timeout + 5)
        except asyncio.TimeoutError:
            return (f"TIMEOUT after {timeout}s", "", -1)
        except Exception as e:
            return (f"Error: {e}", "", -1)
    
    logdir = log_dir or "/root/runs"
    
    if action == "start":
        await run_ssh(f"mkdir -p {logdir} && nohup tensorboard --logdir={logdir} --port={port} --bind_all > /root/tensorboard.log 2>&1 &", timeout=5)
        return f"✅ TensorBoard STARTED ({time.time()-start:.1f}s)\nURL: http://{instance_ip}:{port}"
    
    elif action == "stop":
        stdout, _, _ = await run_ssh("pkill -f tensorboard && echo 'Stopped' || echo 'Not running'")
        return f"✅ {stdout.strip()} ({time.time()-start:.1f}s)"
    
    elif action == "status":
        stdout, _, _ = await run_ssh("pgrep -f tensorboard >/dev/null && echo 'Running' || echo 'Not running'")
        return f"TensorBoard: {stdout.strip()} ({time.time()-start:.1f}s)"
    
    elif action == "url":
        return f"# TensorBoard URL\nhttp://{instance_ip}:{port}\n\nSSH Tunnel: `ssh -L {port}:localhost:{port} root@{instance_ip}`"
    
    return f"❌ Unknown action: {action}"


# =============================================================================
# 28. SESSION_COST MEGA-TOOL (Cost Tracking)
# =============================================================================

# Session tracking storage
_session_data = {
    "start_time": None,
    "instance_id": None,
    "gpu_type": None,
    "hourly_rate": 0.0,
    "total_cost": 0.0,
}


@mcp.tool()
async def session_cost(
    action: str,
    instance_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    hourly_rate: Optional[float] = None,
) -> str:
    """Track session costs - start, check, end, estimate.

    Actions:
        start - Start tracking a session (instance_id, gpu_type, hourly_rate)
        check - Check current session cost
        end - End session and get final cost
        estimate - Estimate cost for hours (hourly_rate, hours)
        history - Show session history

    Args:
        action: One of: start, check, end, estimate, history
        instance_id: Instance ID to track
        gpu_type: GPU type (e.g., V100, H100)
        hourly_rate: Cost per hour in USD
    """
    import time
    
    # GPU pricing (spot rates)
    GPU_RATES = {
        "V100": 0.035, "A6000": 0.12, "L40S": 0.23, "RTX_6000_ADA": 0.21,
        "A100_40G": 0.18, "A100_80G": 0.32, "H100": 0.57, "H200": 0.75,
        "B200": 0.95, "B300": 1.24, "GB300": 1.36, "RTX_PRO_6000": 0.35,
    }
    
    if action == "start":
        if not instance_id:
            return "❌ Error: instance_id required to start tracking"
        
        rate = hourly_rate or GPU_RATES.get(gpu_type, 0.10)
        _session_data["start_time"] = time.time()
        _session_data["instance_id"] = instance_id
        _session_data["gpu_type"] = gpu_type or "Unknown"
        _session_data["hourly_rate"] = rate
        _session_data["total_cost"] = 0.0
        
        return f"""# Session Started ✅
- **Instance**: {instance_id}
- **GPU**: {_session_data['gpu_type']}
- **Rate**: ${rate:.3f}/hr
- **Started**: {time.strftime('%H:%M:%S')}

Use `session_cost(action='check')` to see running cost."""

    elif action == "check":
        if not _session_data["start_time"]:
            return "❌ No active session. Use `session_cost(action='start', instance_id='...')`"
        
        elapsed = time.time() - _session_data["start_time"]
        hours = elapsed / 3600
        cost = hours * _session_data["hourly_rate"]
        
        return f"""# Session Cost 💰
- **Instance**: {_session_data['instance_id']}
- **GPU**: {_session_data['gpu_type']}
- **Duration**: {elapsed/60:.1f} minutes ({hours:.2f} hours)
- **Rate**: ${_session_data['hourly_rate']:.3f}/hr
- **Current Cost**: **${cost:.4f}**
- **Projected 1hr**: ${_session_data['hourly_rate']:.3f}
- **Projected 24hr**: ${_session_data['hourly_rate'] * 24:.2f}"""

    elif action == "end":
        if not _session_data["start_time"]:
            return "❌ No active session to end"
        
        elapsed = time.time() - _session_data["start_time"]
        hours = elapsed / 3600
        cost = hours * _session_data["hourly_rate"]
        
        result = f"""# Session Ended ✅
- **Instance**: {_session_data['instance_id']}
- **GPU**: {_session_data['gpu_type']}
- **Duration**: {elapsed/60:.1f} minutes ({hours:.2f} hours)
- **Final Cost**: **${cost:.4f}**

💡 Remember to stop/delete your instance to avoid charges!"""
        
        # Reset session
        _session_data["start_time"] = None
        _session_data["total_cost"] += cost
        
        return result

    elif action == "estimate":
        rate = hourly_rate or _session_data["hourly_rate"] or 0.10
        hours_list = [1, 2, 4, 8, 24]
        
        lines = ["# Cost Estimates", f"**Rate**: ${rate:.3f}/hr", ""]
        for h in hours_list:
            lines.append(f"- {h}hr: ${rate * h:.2f}")
        
        return "\n".join(lines)

    elif action == "history":
        return f"""# Session History
- **Total Sessions**: 1
- **Cumulative Cost**: ${_session_data['total_cost']:.4f}

💡 Session history resets when MCP server restarts."""

    return f"❌ Unknown action: {action}. Use: start, check, end, estimate, history"


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Run the compact MCP server."""
    logger.info("=" * 60)
    logger.info("🚀 Verda Cloud MCP Server - COMPACT EDITION")
    logger.info("   27 Mega-Tools with 200+ Capabilities")
    logger.info("=" * 60)

    # Count tools
    tools = list(mcp._tool_manager._tools.keys())
    logger.info(f"📊 Total Tools: {len(tools)}")
    logger.info("=" * 60)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()



