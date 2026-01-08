"""Verda Cloud MCP Server - GPU instance management for Claude."""

import asyncio
import logging
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from .client import (
    VerdaSDKClient,
    get_client,
    get_instance_type_from_gpu_type_and_count,
)
from .config import get_config, update_config_file

# Configure logging to stderr (required for MCP servers)
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
# Instance Management Tools
# =============================================================================


@mcp.tool()
async def list_instances() -> str:
    """List all your Verda Cloud instances with their status.

    Returns:
        A formatted list of all instances with ID, hostname, status, type, and IP.
    """
    client = _get_client()
    instances = await client.list_instances()

    if not instances:
        return "No instances found."

    lines = ["# Your Verda Cloud Instances\n"]
    for inst in instances:
        ip_info = f", IP: {inst.ip_address}" if inst.ip_address else ""
        lines.append(
            f"- **{inst.hostname}** (`{inst.id}`)\n"
            f"  Status: {inst.status}, Type: {inst.instance_type}{ip_info}"
        )

    return "\n".join(lines)


@mcp.tool()
async def check_instance_status(instance_id: str) -> str:
    """Check the status of a specific instance.

    Args:
        instance_id: The ID of the instance to check.

    Returns:
        Instance status details including SSH connection info if running.
    """
    client = _get_client()
    instance = await client.get_instance(instance_id)

    result = [
        f"# Instance: {instance.hostname}",
        f"- **ID**: `{instance.id}`",
        f"- **Status**: {instance.status}",
        f"- **Type**: {instance.instance_type}",
    ]

    if instance.ip_address:
        result.append(f"- **IP Address**: {instance.ip_address}")
        result.append("\n## SSH Connection")
        result.append("```bash")
        result.append(f"ssh root@{instance.ip_address}")
        result.append("```")

    return "\n".join(result)


# =============================================================================
# Spot Availability Tools
# =============================================================================


@mcp.tool()
async def check_spot_availability(
    gpu_type: str | None = None,
    gpu_count: int | None = None,
) -> str:
    """Check if spot GPU instances are available.

    Uses the official Verda SDK is_available() method to check across all locations.

    Args:
        gpu_type: GPU type to check (default from config, e.g., "B300", "B200").
        gpu_count: Number of GPUs (default from config, e.g., 1, 2, 4, 8).

    Returns:
        Availability status with location if available.
    """
    client = _get_client()
    config = get_config()

    gpu_type = gpu_type or config.defaults.gpu_type
    gpu_count = gpu_count or config.defaults.gpu_count

    result = await client.check_spot_availability(gpu_type, gpu_count)

    instance_type = get_instance_type_from_gpu_type_and_count(gpu_type, gpu_count)

    lines = [
        "# Spot Availability Check",
        "",
        f"**GPU Type**: {gpu_type}",
        f"**GPU Count**: {gpu_count}",
        f"**Instance Type**: {instance_type or 'Unknown'}",
        "",
    ]

    if result.available:
        lines.append("## ✓ AVAILABLE")
        lines.append("")
        lines.append(f"**Location**: {result.location}")
        lines.append("")
        lines.append(
            "Ready to deploy! Use `deploy_spot_instance` to create an instance."
        )
    else:
        lines.append("## ✗ NOT AVAILABLE")
        lines.append("")
        lines.append(
            "No spot instances available across all locations "
            "(FIN-01, FIN-02, FIN-03)."
        )
        lines.append("")
        lines.append("Options:")
        lines.append("- Use `monitor_spot_availability` to wait for availability")
        lines.append("- Try a different GPU type or count")

    return "\n".join(lines)


@mcp.tool()
async def monitor_spot_availability(
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    check_interval: int = 30,
    max_checks: int = 60,
    auto_deploy: bool = False,
    volume_id: str | None = None,
    script_id: str | None = None,
) -> str:
    """Monitor for spot GPU availability and optionally auto-deploy when available.

    Polls using the official Verda SDK is_available() method until a spot
    becomes available.

    Args:
        gpu_type: GPU type to monitor (default from config).
        gpu_count: Number of GPUs (default from config).
        check_interval: Seconds between checks (default: 30).
        max_checks: Maximum number of checks before giving up (default: 60 = 30 min).
        auto_deploy: If True, automatically deploy when available (default: False).
        volume_id: Volume to attach if auto-deploying (default from config).
        script_id: Startup script if auto-deploying (default from config).

    Returns:
        Status updates and deployment info if auto_deploy is enabled.
    """
    client = _get_client()
    config = get_config()

    gpu_type = gpu_type or config.defaults.gpu_type
    gpu_count = gpu_count or config.defaults.gpu_count

    instance_type = get_instance_type_from_gpu_type_and_count(gpu_type, gpu_count)

    results = [
        f"# Monitoring {gpu_type} x{gpu_count} Spot Availability",
        "",
        f"Instance type: {instance_type}",
        f"Checking every {check_interval}s, max {max_checks} checks "
        f"({max_checks * check_interval // 60} min)",
        "",
    ]

    for check_num in range(1, max_checks + 1):
        availability = await client.check_spot_availability(gpu_type, gpu_count)

        if availability.available:
            results.append(f"## ✓ SPOT AVAILABLE! (Check #{check_num})")
            results.append("")
            results.append(f"**Location**: {availability.location}")
            results.append(f"**Instance Type**: {availability.instance_type}")
            results.append("")

            if auto_deploy:
                results.append("### Auto-deploying...")

                try:
                    final_volume_id = volume_id or config.defaults.volume_id or None
                    final_script_id = script_id or config.defaults.script_id or None
                    volume_ids = [final_volume_id] if final_volume_id else None

                    instance = await client.create_instance(
                        gpu_type=gpu_type,
                        gpu_count=gpu_count,
                        location=availability.location,
                        volume_ids=volume_ids,
                        script_id=final_script_id,
                    )

                    results.append("")
                    results.append(f"**Instance Created**: `{instance.id}`")
                    results.append(f"**Hostname**: {instance.hostname}")
                    results.append("")
                    results.append("Waiting for instance to be ready...")

                    instance = await client.wait_for_ready(instance.id)

                    results.append("")
                    results.append("## Instance Ready!")
                    results.append("")
                    results.append(f"**IP**: {instance.ip_address}")
                    results.append("")
                    results.append("```bash")
                    results.append(f"ssh root@{instance.ip_address}")
                    results.append("```")

                except Exception as e:
                    results.append("")
                    results.append(f"**Error**: {e}")
            else:
                results.append(
                    "Use `deploy_spot_instance` to deploy, "
                    "or re-run with `auto_deploy=True`"
                )

            return "\n".join(results)

        # Not available yet
        logger.info(f"Check #{check_num}: No {gpu_type} x{gpu_count} spots available")

        if check_num < max_checks:
            await asyncio.sleep(check_interval)

    # Timed out
    results.append("## ✗ Timed Out")
    results.append("")
    results.append(f"No spots became available after {max_checks} checks.")
    results.append("Try again later or consider on-demand instances.")

    return "\n".join(results)


# =============================================================================
# Deployment Tools
# =============================================================================


@mcp.tool()
async def deploy_spot_instance(
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    volume_id: str | None = None,
    script_id: str | None = None,
    hostname: str | None = None,
    image: str | None = None,
    wait_for_ready: bool = True,
) -> str:
    """Deploy a new spot GPU instance.

    Args:
        gpu_type: GPU type (default from config, e.g., "B300").
        gpu_count: Number of GPUs (default from config, e.g., 1, 2, 4, 8).
        volume_id: Block volume ID to attach (default from config).
        script_id: Startup script ID (default from config).
        hostname: Instance hostname (auto-generated if not provided).
        image: OS image (default from config).
        wait_for_ready: If True, wait for instance to be ready (default: True).

    Returns:
        Instance details and SSH connection info when ready.
    """
    client = _get_client()
    config = get_config()

    gpu_type = gpu_type or config.defaults.gpu_type
    gpu_count = gpu_count or config.defaults.gpu_count

    # Check availability first
    availability = await client.check_spot_availability(gpu_type, gpu_count)

    if not availability.available:
        return (
            f"# Deployment Failed\n\n"
            f"No spot instances available for {gpu_type} x{gpu_count}.\n\n"
            f"Use `monitor_spot_availability` to wait for availability."
        )

    # Prepare volume IDs
    final_volume_id = volume_id or config.defaults.volume_id
    volume_ids = [final_volume_id] if final_volume_id else None

    # Create instance
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_hostname = hostname or f"{config.defaults.hostname_prefix}-{ts}"

    instance = await client.create_instance(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        location=availability.location,
        image=image,
        hostname=final_hostname,
        volume_ids=volume_ids,
        script_id=script_id or config.defaults.script_id or None,
    )

    result = [
        "# Instance Created Successfully!",
        "",
        f"- **ID**: `{instance.id}`",
        f"- **Hostname**: {instance.hostname}",
        f"- **Type**: {instance.instance_type}",
        f"- **Location**: {availability.location}",
        f"- **Status**: {instance.status}",
    ]

    if volume_ids:
        result.append(f"- **Volume**: {volume_ids[0]}")

    if wait_for_ready:
        result.append("")
        timeout = config.deployment.ready_timeout
        result.append(f"Waiting for instance to be ready (timeout: {timeout}s)...")

        try:
            instance = await client.wait_for_ready(instance.id)
            result.append("")
            result.append("## Instance is Ready!")
            result.append("")
            result.append(f"- **Status**: {instance.status}")
            result.append(f"- **IP Address**: {instance.ip_address}")
            result.append("")
            result.append("## Connect via SSH")
            result.append("```bash")
            result.append(f"ssh root@{instance.ip_address}")
            result.append("```")
        except TimeoutError as e:
            result.append("")
            result.append(f"**Warning**: {e}")
            result.append("Use `check_instance_status` to monitor progress.")
        except RuntimeError as e:
            result.append("")
            result.append(f"**Error**: {e}")

    return "\n".join(result)


# =============================================================================
# Instance Control Tools
# =============================================================================


@mcp.tool()
async def delete_instance(instance_id: str, confirm: bool = False) -> str:
    """Delete an instance.

    Args:
        instance_id: The ID of the instance to delete.
        confirm: Must be True to actually delete (safety check).

    Returns:
        Confirmation of deletion.
    """
    if not confirm:
        return (
            "**Safety Check**: To delete an instance, you must set confirm=True.\n"
            f"Are you sure you want to delete instance `{instance_id}`?"
        )

    client = _get_client()
    await client.delete_instance(instance_id)
    return f"Instance `{instance_id}` has been deleted."


@mcp.tool()
async def shutdown_instance(instance_id: str) -> str:
    """Shutdown a running instance (can be restarted later).

    Args:
        instance_id: The ID of the instance to shutdown.

    Returns:
        Confirmation of shutdown.
    """
    client = _get_client()
    await client.instance_action(instance_id, "shutdown")
    return f"Instance `{instance_id}` shutdown initiated."


@mcp.tool()
async def start_instance(instance_id: str) -> str:
    """Start a stopped instance.

    Args:
        instance_id: The ID of the instance to start.

    Returns:
        Confirmation that start was initiated.
    """
    client = _get_client()
    await client.instance_action(instance_id, "boot")
    return (
        f"Instance `{instance_id}` start initiated. "
        "Use `check_instance_status` to monitor."
    )


# =============================================================================
# Resource Listing Tools
# =============================================================================


@mcp.tool()
async def list_volumes() -> str:
    """List your block storage volumes.

    Returns:
        A list of volumes with ID, name, size, and attachment status.
    """
    client = _get_client()
    volumes = await client.list_volumes()

    if not volumes:
        return "No volumes found."

    lines = ["# Your Block Volumes\n"]
    for vol in volumes:
        if vol.attached_to:
            attached = f"Attached to: {vol.attached_to}"
        else:
            attached = "Not attached"
        lines.append(
            f"- **{vol.name}** (`{vol.id}`)\n"
            f"  Size: {vol.size_gb} GB, Status: {vol.status}, {attached}"
        )

    return "\n".join(lines)


@mcp.tool()
async def list_scripts() -> str:
    """List your startup scripts.

    Returns:
        A list of scripts with ID and name.
    """
    client = _get_client()
    scripts = await client.list_scripts()

    if not scripts:
        return "No startup scripts found."

    lines = ["# Your Startup Scripts\n"]
    for script in scripts:
        lines.append(f"- **{script.name}** (ID: `{script.id}`)")

    return "\n".join(lines)


@mcp.tool()
async def list_ssh_keys() -> str:
    """List your SSH keys.

    Returns:
        A list of SSH keys with ID and name.
    """
    client = _get_client()
    keys = await client.list_ssh_keys()

    if not keys:
        return "No SSH keys found. Please add an SSH key in the Verda console."

    lines = ["# Your SSH Keys\n"]
    for key in keys:
        lines.append(f"- **{key.name}** (ID: `{key.id}`)")

    return "\n".join(lines)


@mcp.tool()
async def list_images() -> str:
    """List available OS images.

    Returns:
        A list of available OS images.
    """
    client = _get_client()
    images = await client.list_images()

    if not images:
        return "No images found."

    # Filter for Ubuntu images
    ubuntu_images = [img for img in images if "ubuntu" in img.name.lower()]

    lines = ["# Available Ubuntu Images\n"]
    for img in ubuntu_images[:10]:  # Show first 10
        lines.append(f"- **{img.name}** (`{img.image_type}`)")

    if len(ubuntu_images) > 10:
        lines.append(f"\n... and {len(ubuntu_images) - 10} more")

    return "\n".join(lines)


# =============================================================================
# Volume Management Tools
# =============================================================================


@mcp.tool()
async def attach_volume(volume_id: str, instance_id: str) -> str:
    """Attach a volume to an instance.

    Note: The instance must be shut down first.

    Args:
        volume_id: The ID of the volume to attach.
        instance_id: The ID of the instance to attach to.

    Returns:
        Confirmation of attachment.
    """
    client = _get_client()
    await client.attach_volume(volume_id, instance_id)
    return f"Volume `{volume_id}` attached to instance `{instance_id}`."


@mcp.tool()
async def detach_volume(volume_id: str) -> str:
    """Detach a volume from its current instance.

    Note: The instance must be shut down first.

    Args:
        volume_id: The ID of the volume to detach.

    Returns:
        Confirmation of detachment.
    """
    client = _get_client()
    await client.detach_volume(volume_id)
    return f"Volume `{volume_id}` detached."


@mcp.tool()
async def create_volume(
    name: str,
    size: int | None = None,
    volume_type: str = "NVMe",
) -> str:
    """Create a new block storage volume.

    Args:
        name: Name for the volume (e.g., "my-data-volume").
        size: Volume size in GB (default: 150GB from config).
        volume_type: Volume type (default: "NVMe").

    Returns:
        Created volume details with ID.
    """
    client = _get_client()
    config = get_config()

    size = size or config.defaults.volume_size

    volume = await client.create_volume(
        name=name,
        size=size,
        volume_type=volume_type,
    )

    return (
        f"# Volume Created Successfully!\n\n"
        f"- **Name**: {volume.name}\n"
        f"- **ID**: `{volume.id}`\n"
        f"- **Size**: {volume.size_gb} GB\n"
        f"- **Type**: {volume_type}\n"
        f"- **Status**: {volume.status}\n\n"
        f"Add this to your config.yaml to use as default:\n"
        f"```yaml\n"
        f"defaults:\n"
        f'  volume_id: "{volume.id}"\n'
        f"```"
    )


# =============================================================================
# Script Management Tools
# =============================================================================

@mcp.tool()
async def get_instance_startup_script(instance_id: str) -> str:
    """Get the startup script attached to a specific Verda Cloud instance.

    Args:
        instance_id: The ID of the instance.

    Returns:
        The script name, ID, and content, or a message if no script is attached.
    """
    client = _get_client()
    script = await client.get_current_script(instance_id)

    if script is None:
        return "No startup script attached to this instance."

    return f"""# Startup Script: {script.name}
**ID**: `{script.id}`

## Content
```bash
{script.content}
```"""


@mcp.tool()
async def create_startup_script(name: str, content: str) -> str:
    """Create a new startup script.

    Args:
        name: Name for the script.
        content: Bash script content.

    Returns:
        Created script ID.
    """
    client = _get_client()
    script = await client.create_script(name, content)
    return f"Script created: **{script.name}** (ID: `{script.id}`)"


@mcp.tool()
async def create_and_set_default_script(name: str, content: str) -> str:
    """Create a new startup script and set it as the default for new instances.

    This creates a new script and updates config.yaml to use it as the default
    script_id for future instance deployments.

    Args:
        name: Name for the new script.
        content: Bash script content.

    Returns:
        Confirmation with script ID and updated config status.
    """
    client = _get_client()

    # Create the new script
    script = await client.create_script(name, content)

    # Update the config file to set this as default
    update_config_file({"defaults": {"script_id": script.id}})

    return f"""# Script Created and Set as Default

**Name**: {script.name}
**ID**: `{script.id}`

The config.yaml has been updated. All new instances will use this script by default."""


@mcp.tool()
async def set_default_script(script_id: str) -> str:
    """Set an existing script as the default for new Verda Cloud instances.

    Args:
        script_id: The ID of an existing script to set as default.

    Returns:
        Confirmation of the updated default.
    """
    client = _get_client()

    # Verify the script exists
    script = await client.get_script_by_id(script_id)

    # Update the config file
    update_config_file({"defaults": {"script_id": script.id}})

    return f"""# Default Script Updated

**Name**: {script.name}
**ID**: `{script.id}`

All new instances will now use this script by default."""


# =============================================================================
# Configuration Info Tool
# =============================================================================


@mcp.tool()
async def show_config() -> str:
    """Show the current MCP server configuration (without secrets).

    Returns:
        Current configuration settings.
    """
    config = get_config()

    instance_type = get_instance_type_from_gpu_type_and_count(
        config.defaults.gpu_type,
        config.defaults.gpu_count,
    )

    lines = [
        "# Current Configuration",
        "",
        "## GPU Defaults",
        f"- **GPU Type**: {config.defaults.gpu_type}",
        f"- **GPU Count**: {config.defaults.gpu_count}",
        f"- **Instance Type**: {instance_type}",
        f"- **Location**: {config.defaults.location}",
        "",
        "## Deployment Defaults",
        f"- **Image**: {config.defaults.image}",
        f"- **Hostname Prefix**: {config.defaults.hostname_prefix}",
        f"- **Volume ID**: {config.defaults.volume_id or '(not set)'}",
        f"- **Script ID**: {config.defaults.script_id or '(not set)'}",
        "",
        "## Deployment Settings",
        f"- **Ready Timeout**: {config.deployment.ready_timeout}s",
        f"- **Poll Interval**: {config.deployment.poll_interval}s",
        f"- **Use Spot**: {config.deployment.use_spot}",
    ]

    return "\n".join(lines)


# =============================================================================
# SSH Remote Access Tools (NEW!)
# =============================================================================

try:
    from .ssh_tools import (
        ssh_run_command,
        ssh_get_gpu_status,
        ssh_get_training_logs,
        ssh_get_training_progress,
        ssh_read_file,
        ssh_write_file,
        ssh_list_dir,
        ssh_kill_training,
        PARAMIKO_AVAILABLE,
    )
    SSH_TOOLS_AVAILABLE = PARAMIKO_AVAILABLE
except ImportError:
    SSH_TOOLS_AVAILABLE = False


@mcp.tool()
async def remote_run_command(instance_ip: str, command: str) -> str:
    """Run a command on a remote Verda instance via SSH.

    Args:
        instance_ip: The IP address of the instance.
        command: The bash command to run.

    Returns:
        Command output (stdout, stderr, exit code).
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_run_command(instance_ip, command)


@mcp.tool()
async def remote_gpu_status(instance_ip: str) -> str:
    """Get GPU status (nvidia-smi) from a remote instance.

    Args:
        instance_ip: The IP address of the instance.

    Returns:
        nvidia-smi output showing GPU utilization, memory, etc.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_get_gpu_status(instance_ip)


@mcp.tool()
async def remote_training_logs(instance_ip: str, lines: int = 50) -> str:
    """Get recent training logs from a remote instance.

    Args:
        instance_ip: The IP address of the instance.
        lines: Number of log lines to retrieve (default: 50).

    Returns:
        Recent training log output.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_get_training_logs(instance_ip, lines)


@mcp.tool()
async def remote_training_progress(instance_ip: str) -> str:
    """Get comprehensive training progress from a remote instance.

    Includes GPU status, training process status, recent logs, and disk space.

    Args:
        instance_ip: The IP address of the instance.

    Returns:
        Full training progress report.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_get_training_progress(instance_ip)


@mcp.tool()
async def remote_read_file(instance_ip: str, file_path: str, max_lines: int = 0) -> str:
    """Read a file from a remote instance.

    Args:
        instance_ip: The IP address of the instance.
        file_path: Path to the file on the remote instance.
        max_lines: If > 0, only read last N lines (like tail).

    Returns:
        File contents.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_read_file(instance_ip, file_path, max_lines if max_lines > 0 else None)


@mcp.tool()
async def remote_write_file(instance_ip: str, file_path: str, content: str) -> str:
    """Write content to a file on a remote instance.

    Args:
        instance_ip: The IP address of the instance.
        file_path: Path to the file on the remote instance.
        content: Content to write to the file.

    Returns:
        Success or failure message.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_write_file(instance_ip, file_path, content)


@mcp.tool()
async def remote_list_dir(instance_ip: str, dir_path: str) -> str:
    """List files in a directory on a remote instance.

    Args:
        instance_ip: The IP address of the instance.
        dir_path: Path to the directory on the remote instance.

    Returns:
        Directory listing (ls -la output).
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_list_dir(instance_ip, dir_path)


@mcp.tool()
async def remote_kill_training(instance_ip: str) -> str:
    """Kill any running training processes on a remote instance.

    Args:
        instance_ip: The IP address of the instance.

    Returns:
        Result of the kill command.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await ssh_kill_training(instance_ip)


# =============================================================================
# Extended Tools (Cost, Health, Logs, Checkpoints)
# =============================================================================

try:
    from .extended_tools import (
        estimate_training_cost,
        parse_training_logs,
        health_check,
        list_instance_checkpoints,
        backup_latest_checkpoint,
        list_all_gpus,
        recommend_gpu_for_model,
    )
    EXTENDED_TOOLS_AVAILABLE = True
except ImportError:
    EXTENDED_TOOLS_AVAILABLE = False


@mcp.tool()
async def cost_estimate(
    gpu_type: str = "B300",
    gpu_count: int = 1,
    hours: float = 20,
    is_spot: bool = True,
) -> str:
    """Estimate training cost before deployment.

    Args:
        gpu_type: GPU type (B300, B200, H200, etc.).
        gpu_count: Number of GPUs.
        hours: Expected training hours.
        is_spot: Whether using spot pricing.

    Returns:
        Cost breakdown and comparison.
    """
    if not EXTENDED_TOOLS_AVAILABLE:
        return "❌ Extended tools not available."
    return await estimate_training_cost(gpu_type, gpu_count, hours, is_spot)


@mcp.tool()
async def analyze_training_logs(instance_ip: str, log_lines: int = 200) -> str:
    """Parse and analyze training logs to extract metrics.

    Extracts steps, losses, learning rates, checkpoints, errors, and warnings.

    Args:
        instance_ip: IP address of the instance.
        log_lines: Number of log lines to analyze.

    Returns:
        Parsed metrics summary.
    """
    if not EXTENDED_TOOLS_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ Required tools not available."
    return await parse_training_logs(instance_ip, log_lines)


@mcp.tool()
async def instance_health_check(instance_ip: str) -> str:
    """Perform comprehensive health check on an instance.

    Checks GPU, disk, memory, training process, and network health.

    Args:
        instance_ip: IP address of the instance.

    Returns:
        Health check report with status for each component.
    """
    if not EXTENDED_TOOLS_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ Required tools not available."
    return await health_check(instance_ip)


@mcp.tool()
async def list_checkpoints(instance_ip: str) -> str:
    """List available training checkpoints on an instance.

    Args:
        instance_ip: IP address of the instance.

    Returns:
        List of checkpoints with names, sizes, and dates.
    """
    if not EXTENDED_TOOLS_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ Required tools not available."
    return await list_instance_checkpoints(instance_ip)


@mcp.tool()
async def backup_checkpoint(instance_ip: str, local_dir: str) -> str:
    """Backup the latest checkpoint from instance to local machine.

    Args:
        instance_ip: IP address of the instance.
        local_dir: Local directory to save checkpoint.

    Returns:
        Backup status.
    """
    if not EXTENDED_TOOLS_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ Required tools not available."
    return await backup_latest_checkpoint(instance_ip, local_dir)


@mcp.tool()
async def list_available_gpus() -> str:
    """List ALL available Verda GPUs with pricing, VRAM, and multi-GPU options.

    Shows complete GPU catalog including:
    - NVLink GPUs: GB300, B300, B200, H200, H100, A100, V100
    - General Compute: RTX PRO 6000, L40S, RTX 6000 Ada, RTX A6000

    Returns:
        Table of all GPUs sorted by price with specs.
    """
    if not EXTENDED_TOOLS_AVAILABLE:
        return "❌ Extended tools not available."
    return await list_all_gpus()


@mcp.tool()
async def recommend_gpu(
    model_size_billions: float,
    budget_per_hour: float = 0,
) -> str:
    """Recommend GPU configuration based on model size and budget.

    Analyzes model VRAM requirements and suggests optimal GPU configs
    including single and multi-GPU options (1, 2, 4, 8 GPUs).

    Args:
        model_size_billions: Model size in billions of parameters (e.g., 7, 13, 70).
        budget_per_hour: Optional max budget per hour in USD. 0 = no limit.

    Returns:
        Recommended configurations sorted by cost.
    """
    if not EXTENDED_TOOLS_AVAILABLE:
        return "❌ Extended tools not available."
    budget = budget_per_hour if budget_per_hour > 0 else None
    return await recommend_gpu_for_model(model_size_billions, budget)


# =============================================================================
# Google Drive & File Transfer Tools (NEW!)
# =============================================================================

try:
    from .gdrive_tools import (
        gdrive_download_to_local,
        gdrive_download_folder_to_local,
        transfer_local_to_verda,
        transfer_verda_to_local,
        auto_setup_training,
        auto_start_training,
        GDOWN_AVAILABLE,
    )
    GDRIVE_TOOLS_AVAILABLE = GDOWN_AVAILABLE
except ImportError:
    GDRIVE_TOOLS_AVAILABLE = False


@mcp.tool()
async def gdrive_download_file(gdrive_url: str, local_path: str = "") -> str:
    """Download a file from Google Drive to local machine.

    Args:
        gdrive_url: Google Drive URL or file ID.
        local_path: Optional local destination path.

    Returns:
        Download status and file path.
    """
    if not GDRIVE_TOOLS_AVAILABLE:
        return "❌ Google Drive tools not available. Install gdown: pip install gdown"
    return await gdrive_download_to_local(gdrive_url, local_path if local_path else None)


@mcp.tool()
async def gdrive_download_folder(gdrive_url: str, local_dir: str = "") -> str:
    """Download a folder from Google Drive to local machine.

    Args:
        gdrive_url: Google Drive folder URL or ID.
        local_dir: Optional local destination directory.

    Returns:
        Download status and folder path.
    """
    if not GDRIVE_TOOLS_AVAILABLE:
        return "❌ Google Drive tools not available. Install gdown: pip install gdown"
    return await gdrive_download_folder_to_local(gdrive_url, local_dir if local_dir else None)


@mcp.tool()
async def upload_to_verda(local_path: str, instance_ip: str, remote_path: str) -> str:
    """Upload a local file to a Verda instance via SCP.

    Args:
        local_path: Path to local file.
        instance_ip: IP address of the Verda instance.
        remote_path: Destination path on the instance.

    Returns:
        Upload status.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await transfer_local_to_verda(local_path, instance_ip, remote_path)


@mcp.tool()
async def download_from_verda(instance_ip: str, remote_path: str, local_path: str) -> str:
    """Download a file from a Verda instance to local machine via SCP.

    Args:
        instance_ip: IP address of the Verda instance.
        remote_path: Path to file on the instance.
        local_path: Local destination path.

    Returns:
        Download status.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await transfer_verda_to_local(instance_ip, remote_path, local_path)


@mcp.tool()
async def automated_setup(instance_ip: str, gdrive_url: str) -> str:
    """Fully automated training environment setup on Verda instance.

    Performs:
    1. Installs gdown on instance
    2. Downloads training package from Google Drive
    3. Extracts zip files
    4. Installs requirements
    5. Verifies GPU

    Args:
        instance_ip: IP address of the Verda instance.
        gdrive_url: Google Drive folder/file URL with training package.

    Returns:
        Detailed setup status report.
    """
    if not SSH_TOOLS_AVAILABLE or not GDRIVE_TOOLS_AVAILABLE:
        return "❌ Required tools not available. Install paramiko and gdown."
    return await auto_setup_training(instance_ip, gdrive_url)


@mcp.tool()
async def automated_start_training(instance_ip: str, script_path: str) -> str:
    """Start training on a Verda instance with screen session.

    Starts the training script in a detached screen session for persistence.

    Args:
        instance_ip: IP address of the Verda instance.
        script_path: Path to training script on the instance.

    Returns:
        Training start status and commands for monitoring.
    """
    if not SSH_TOOLS_AVAILABLE:
        return "❌ SSH tools not available. Install paramiko: pip install paramiko"
    return await auto_start_training(instance_ip, script_path)


# =============================================================================
# WatchDog Monitoring Tools (NEW!)
# =============================================================================

try:
    from .watchdog import (
        start_watchdog,
        stop_watchdog,
        get_watchdog_status,
        get_latest_watchdog_report,
        manual_watchdog_check,
    )
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


@mcp.tool()
async def watchdog_enable(instance_ip: str, interval_minutes: int = 10) -> str:
    """Enable WatchDog automatic monitoring for a Verda instance.

    When enabled, WatchDog will check training status every N minutes and
    create timestamped markdown reports automatically.

    Args:
        instance_ip: IP address of the instance to monitor.
        interval_minutes: Check interval in minutes (default: 10).

    Returns:
        Status message confirming WatchDog is enabled.
    """
    if not WATCHDOG_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ WatchDog not available. Ensure SSH tools are installed."
    return await start_watchdog(instance_ip, interval_minutes)


@mcp.tool()
async def watchdog_disable() -> str:
    """Disable WatchDog automatic monitoring.

    Stops the automatic monitoring and creates a summary report.

    Returns:
        Status message and summary report location.
    """
    if not WATCHDOG_AVAILABLE:
        return "❌ WatchDog not available."
    return await stop_watchdog()


@mcp.tool()
async def watchdog_status() -> str:
    """Get the current WatchDog monitoring status.

    Returns:
        Current status including monitoring target and report locations.
    """
    if not WATCHDOG_AVAILABLE:
        return "❌ WatchDog not available."
    return await get_watchdog_status()


@mcp.tool()
async def watchdog_latest_report() -> str:
    """Get the latest WatchDog report content.

    Returns:
        The full content of the most recent monitoring report.
    """
    if not WATCHDOG_AVAILABLE:
        return "❌ WatchDog not available."
    return await get_latest_watchdog_report()


@mcp.tool()
async def watchdog_check_now(instance_ip: str) -> str:
    """Perform a manual WatchDog check (one-time, without enabling continuous monitoring).

    Creates a full status report with GPU, training, logs, and disk info.

    Args:
        instance_ip: IP address of the instance to check.

    Returns:
        Full status report and saved report path.
    """
    if not WATCHDOG_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ WatchDog not available. Ensure SSH tools are installed."
    return await manual_watchdog_check(instance_ip)


# =============================================================================
# Spot Instance Manager (Auto-Switch, Failover, Savings!)
# =============================================================================

try:
    from .spot_manager import (
        compare_spot_vs_ondemand,
        smart_deploy_instance,
        switch_instance_mode,
        get_session_status,
        stop_session_monitoring,
        GPU_PRICING_FULL,
    )
    SPOT_MANAGER_AVAILABLE = True
except ImportError:
    SPOT_MANAGER_AVAILABLE = False


@mcp.tool()
async def spot_savings_calculator(
    gpu_type: str = "B300",
    gpu_count: int = 1,
    hours: float = 20,
) -> str:
    """Compare SPOT vs On-Demand pricing and calculate savings.

    Shows how much you save using spot instances (typically 75% savings!).
    ALWAYS recommends SPOT with 10-minute checkpoints for training.

    Args:
        gpu_type: GPU type (B300, A6000, H100, etc.).
        gpu_count: Number of GPUs (1, 2, 4, 8).
        hours: Expected training hours.

    Returns:
        Detailed comparison with savings calculation.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await compare_spot_vs_ondemand(gpu_type, gpu_count, hours)


@mcp.tool()
async def smart_deploy(
    gpu_type: str = "B300",
    gpu_count: int = 1,
    prefer_spot: bool = True,
    auto_failover: bool = True,
    checkpoint_minutes: int = 10,
    volume_id: str = "",
    script_id: str = "",
) -> str:
    """Smart deploy with SPOT preference and auto-failover to On-Demand.

    RECOMMENDED deployment method! Automatically:
    1. Tries SPOT first (75% savings!)
    2. Falls back to On-Demand if spot unavailable
    3. Monitors for eviction and auto-recovers
    4. Reminds about checkpoint requirements

    Args:
        gpu_type: GPU type (B300, A6000, etc.).
        gpu_count: Number of GPUs (1, 2, 4, 8).
        prefer_spot: Try spot first (default: True - RECOMMENDED!).
        auto_failover: Auto-switch to on-demand if spot fails (default: True).
        checkpoint_minutes: Checkpoint interval (default: 10 - CRITICAL for spot!).
        volume_id: Volume ID to attach (optional).
        script_id: Startup script ID (optional).

    Returns:
        Deployment result with instance info and savings.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await smart_deploy_instance(
        gpu_type, gpu_count, prefer_spot, auto_failover,
        checkpoint_minutes, volume_id, script_id
    )


@mcp.tool()
async def switch_to_spot() -> str:
    """Switch current training from On-Demand to SPOT to save money.

    Creates new SPOT instance and provides instructions to resume training.
    Use this when spot becomes available during an on-demand session.

    Returns:
        New instance info and resume instructions.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await switch_instance_mode(to_spot=True)


@mcp.tool()
async def switch_to_ondemand() -> str:
    """Switch current training from SPOT to On-Demand for stability.

    Creates new On-Demand instance and provides instructions to resume training.
    Use this if you need guaranteed uptime and don't want eviction risk.

    Returns:
        New instance info and resume instructions.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await switch_instance_mode(to_spot=False)


@mcp.tool()
async def training_session_status() -> str:
    """Get current training session status.

    Shows instance mode (spot/on-demand), duration, cost, checkpoints,
    and eviction history.

    Returns:
        Detailed session status report.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await get_session_status()


@mcp.tool()
async def end_training_session() -> str:
    """End current training session and stop monitoring.

    Call this when training is complete to stop eviction monitoring
    and finalize the session.

    Returns:
        Confirmation message.
    """
    if not SPOT_MANAGER_AVAILABLE:
        return "❌ Spot manager not available."
    return await stop_session_monitoring()


# =============================================================================
# Training Tools (Checkpoints, Scripts, Alerts, Notifications)
# =============================================================================

try:
    from .training_tools import (
        check_account_balance,
        generate_checkpoint_script,
        generate_startup_script,
        set_cost_alert,
        send_training_notification,
        upload_checkpoint_to_gdrive,
        list_available_frameworks,
    )
    TRAINING_TOOLS_AVAILABLE = True
except ImportError:
    TRAINING_TOOLS_AVAILABLE = False


@mcp.tool()
async def check_balance() -> str:
    """Check Verda account balance before deploying.

    Shows current balance, usage, and whether you have sufficient funds.
    ALWAYS check balance before deploying expensive multi-GPU instances!

    Returns:
        Account balance and usage information.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await check_account_balance()


@mcp.tool()
async def create_checkpoint_script(
    framework: str = "huggingface",
    checkpoint_dir: str = "/workspace/checkpoints",
    checkpoint_minutes: int = 10,
    model_name: str = "my_model",
) -> str:
    """Generate a training script with automatic 10-minute checkpoints.

    CRITICAL for spot instances! This script:
    - Saves checkpoints every 10 minutes (configurable)
    - Auto-resumes from the latest checkpoint
    - Handles spot eviction gracefully

    Args:
        framework: Training framework (huggingface, pytorch, lightning, llama, stable_diffusion).
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_minutes: Checkpoint interval in minutes (default: 10).
        model_name: Name for your model.

    Returns:
        Complete Python training script with checkpoint support.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await generate_checkpoint_script(framework, checkpoint_dir, checkpoint_minutes, model_name)


@mcp.tool()
async def create_startup_script(framework: str = "huggingface") -> str:
    """Generate a Verda startup script for your framework.

    Creates a bash script that:
    - Installs all required packages
    - Sets up the workspace
    - Verifies GPU access

    Args:
        framework: Framework to set up (huggingface, pytorch, lightning, llama, stable_diffusion).

    Returns:
        Bash startup script ready to use with Verda.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await generate_startup_script(framework)


@mcp.tool()
async def list_frameworks() -> str:
    """List all available framework templates.

    Shows all supported frameworks with descriptions.
    Use these with create_checkpoint_script and create_startup_script.

    Returns:
        List of frameworks and their descriptions.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await list_available_frameworks()


@mcp.tool()
async def set_training_cost_alert(
    threshold_usd: float,
    webhook_url: str = "",
) -> str:
    """Set a cost alert to warn when training costs exceed a threshold.

    Get notified when your training costs go over the specified amount.

    Args:
        threshold_usd: Alert when cost exceeds this amount in USD.
        webhook_url: Optional webhook URL for notifications.

    Returns:
        Confirmation of alert setup.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await set_cost_alert(threshold_usd, webhook_url)


@mcp.tool()
async def notify_training_event(
    webhook_url: str,
    message: str,
    event_type: str = "training_update",
) -> str:
    """Send a training notification via webhook.

    Use this to notify external services about training events
    (start, complete, checkpoint saved, error, etc.).

    Args:
        webhook_url: Webhook URL to send notification to.
        message: Notification message.
        event_type: Type of event (training_start, checkpoint, training_complete, error).

    Returns:
        Notification status.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await send_training_notification(webhook_url, message, event_type)


@mcp.tool()
async def backup_checkpoint_to_gdrive(
    instance_ip: str,
    checkpoint_path: str = "/workspace/checkpoints",
    gdrive_folder: str = "verda-checkpoints",
) -> str:
    """Upload checkpoints from instance to Google Drive.

    Backup your training checkpoints to Google Drive for safety.
    Requires rclone configured on the instance.

    Args:
        instance_ip: IP address of the instance.
        checkpoint_path: Path to checkpoint on instance.
        gdrive_folder: Google Drive folder name.

    Returns:
        Upload status and instructions.
    """
    if not TRAINING_TOOLS_AVAILABLE:
        return "❌ Training tools not available."
    return await upload_checkpoint_to_gdrive(instance_ip, checkpoint_path, gdrive_folder)


# =============================================================================
# Smart Deployer (Fail-Safes, Best Deals, Auto-Recovery)
# =============================================================================

try:
    from .smart_deployer import (
        find_best_deals_now,
        find_power_deals_now,
        deploy_with_all_failsafes,
        show_available_now,
    )
    SMART_DEPLOYER_AVAILABLE = True
except ImportError:
    SMART_DEPLOYER_AVAILABLE = False


@mcp.tool()
async def best_deals_now(budget: float = 5.0, min_vram: int = 48) -> str:
    """Find BEST DEALS available RIGHT NOW with real-time availability check.

    Scans all GPUs across all locations and ranks by value (power/cost).
    Shows multi-GPU spot options that beat single on-demand!

    Args:
        budget: Maximum hourly budget in USD.
        min_vram: Minimum VRAM required in GB.

    Returns:
        Ranked list of best available deals.
    """
    if not SMART_DEPLOYER_AVAILABLE:
        return "❌ Smart deployer not available."
    return await find_best_deals_now(budget, min_vram)


@mcp.tool()
async def power_deals_now(reference_gpu: str = "B300", reference_count: int = 1) -> str:
    """Find configs with MORE POWER for SAME or LESS cost.

    Compares multi-GPU spot to single on-demand.
    Example: 4x B200 spot might give 3x power for same price as 1x B300!

    Args:
        reference_gpu: GPU to compare against.
        reference_count: Number of reference GPUs.

    Returns:
        List of better power deals.
    """
    if not SMART_DEPLOYER_AVAILABLE:
        return "❌ Smart deployer not available."
    return await find_power_deals_now(reference_gpu, reference_count)


@mcp.tool()
async def deploy_failsafe(
    gpu_type: str = "A6000",
    gpu_count: int = 1,
    prefer_spot: bool = True,
    volume_id: str = "",
    script_id: str = "",
) -> str:
    """Deploy with COMPREHENSIVE FAIL-SAFES and backup plans.

    FAIL-SAFE LAYERS:
    1. Check availability before deploying
    2. Try primary location, fall back to others
    3. If spot fails, auto-fallback to on-demand
    4. Retry failed deployments (3 attempts)
    5. Verify deployment matches request
    6. Start health monitoring
    7. Eviction detection with auto-recovery

    Args:
        gpu_type: GPU type (A6000, B300, H100, etc.).
        gpu_count: Number of GPUs (1, 2, 4, 8).
        prefer_spot: Try spot first (default: True).
        volume_id: Volume to attach (optional).
        script_id: Startup script (optional).

    Returns:
        Deployment result with instance info.
    """
    if not SMART_DEPLOYER_AVAILABLE:
        return "❌ Smart deployer not available."
    return await deploy_with_all_failsafes(gpu_type, gpu_count, prefer_spot, volume_id, script_id)


@mcp.tool()
async def available_now() -> str:
    """Show ALL GPUs available RIGHT NOW across all locations.

    Real-time scan of spot and on-demand availability.

    Returns:
        Complete availability matrix.
    """
    if not SMART_DEPLOYER_AVAILABLE:
        return "❌ Smart deployer not available."
    return await show_available_now()


# =============================================================================

# =============================================================================
# Training Intelligence (Mega-Tools with 55+ bundled functions)
# =============================================================================

try:
    from .training_intelligence import (
        training_intel,
        training_viz,
        training_profile,
        training_monitor,
        model_advisor,
    )
    TRAINING_INTEL_AVAILABLE = True
except ImportError:
    TRAINING_INTEL_AVAILABLE = False


@mcp.tool()
async def train_intel(
    action: str = "status",
    instance_ip: str = "",
    skill_level: str = "normal",
) -> str:
    """MEGA-TOOL: Training Intelligence Hub (15+ sub-commands).
    
    Analyzes training in real-time and converts metrics to simple English.
    Uses 10-stage rating system (1-10) to show progress.
    
    Actions:
    - status: Full training status with analysis
    - stage: Current stage (1-10) with explanation
    - health: Health score (0-100) and issues
    - metrics: Raw training metrics
    - summary: Simple English summary
    - detailed: Technical detailed summary
    - trends: Loss/accuracy trends
    - issues: Detected problems
    - recommendations: Actionable suggestions
    - predict: Estimated completion time
    - explain: What metrics mean
    
    Args:
        action: Sub-command to run.
        instance_ip: Instance to analyze (optional).
        skill_level: beginner/casual/normal/advanced/expert/elite/hacker
    
    Returns:
        Training analysis adapted to skill level.
    """
    if not TRAINING_INTEL_AVAILABLE:
        return "Training Intelligence not available."
    return await training_intel(action=action, instance_ip=instance_ip, skill_level=skill_level)


@mcp.tool()
async def train_viz(
    format: str = "ascii",
    chart_type: str = "progress",
    instance_ip: str = "",
) -> str:
    """MEGA-TOOL: Training Visualization Hub (7 output formats).
    
    Generates training visualizations in multiple formats.
    
    Formats:
    - ascii: ASCII art charts (terminal-friendly)
    - markdown: GitHub-flavored markdown
    - html: Modern dashboard with CSS
    - svg: Scalable vector graphics
    - json: Raw JSON data
    - terminal: ANSI colored output
    - minimal: Just the numbers
    
    Chart Types:
    - progress: Progress bar/gauge
    - loss: Loss curve
    - dashboard: Full dashboard
    - gauge: Speedometer style
    
    Args:
        format: Output format type.
        chart_type: Type of visualization.
        instance_ip: Instance to visualize.
    
    Returns:
        Visualization in requested format.
    """
    if not TRAINING_INTEL_AVAILABLE:
        return "Training Intelligence not available."
    return await training_viz(format=format, chart_type=chart_type, instance_ip=instance_ip)


@mcp.tool()
async def train_profile(
    action: str = "set",
    level: str = "normal",
) -> str:
    """MEGA-TOOL: User Profile Manager (7 skill levels).
    
    Adapts all output to your experience level.
    
    Levels:
    - beginner: Simple, friendly explanations
    - casual: Easy with some technical terms
    - normal: Balanced (default)
    - advanced: Full technical details
    - expert: Raw data + analysis
    - elite: Everything + academic context
    - hacker: Minimal CLI-style
    
    Actions:
    - set: Set your level
    - get: Get current level
    - list: List all levels
    - describe: Explain each level
    - sample: Preview output style
    
    Args:
        action: What to do.
        level: Skill level to set.
    
    Returns:
        Profile update confirmation.
    """
    if not TRAINING_INTEL_AVAILABLE:
        return "Training Intelligence not available."
    return await training_profile(action=action, level=level)


@mcp.tool()
async def train_monitor(
    action: str = "check",
    instance_ip: str = "",
    interval: int = 60,
) -> str:
    """MEGA-TOOL: Real-time Training Monitor (12 functions).
    
    Continuous monitoring with alerts and logging.
    
    Actions:
    - start: Begin monitoring
    - stop: Stop monitoring
    - status: Monitor status
    - check: Single check now
    - alerts_on/alerts_off: Toggle alerts
    - get_logs: Recent logs
    - get_alerts: Triggered alerts
    - export: Export data
    
    Args:
        action: Monitor command.
        instance_ip: Instance to monitor.
        interval: Check interval in seconds.
    
    Returns:
        Monitor status/results.
    """
    if not TRAINING_INTEL_AVAILABLE:
        return "Training Intelligence not available."
    return await training_monitor(action=action, instance_ip=instance_ip, interval=interval)


@mcp.tool()
async def train_advisor(
    action: str = "recommend",
    model_size: str = "7B",
    budget: float = 5.0,
) -> str:
    """MEGA-TOOL: Model & Training Advisor (10 functions).
    
    Get recommendations for GPU, settings, and optimization.
    
    Actions:
    - recommend: GPU for model size
    - compare: Compare GPU options
    - estimate_time: Training duration
    - estimate_cost: Total cost
    - optimize: Optimization tips
    - batch_size: Optimal batch size
    - learning_rate: LR recommendation
    - checkpointing: Checkpoint strategy
    - multi_gpu: Multi-GPU advice
    - frameworks: Framework recommendations
    
    Args:
        action: Advice type.
        model_size: Model size (7B, 13B, 70B, etc).
        budget: Hourly budget in USD.
    
    Returns:
        Expert recommendations.
    """
    if not TRAINING_INTEL_AVAILABLE:
        return "Training Intelligence not available."
    return await model_advisor(action=action, model_size=model_size, budget=budget)

# GPU Optimizer (Multi-GPU Spot Comparisons, Training Time Estimates)
# =============================================================================

try:
    from .gpu_optimizer import (
        compare_multi_gpu_spot,
        find_best_gpu_config,
        estimate_training_time,
        list_all_gpus_detailed,
    )
    GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZER_AVAILABLE = False


@mcp.tool()
async def spot_vs_ondemand_comparison(gpu_type: str = "B300", hours: float = 24) -> str:
    """Compare multi-GPU SPOT vs single GPU On-Demand pricing.

    KEY INSIGHT: With 75% spot savings, you can often get 3-4x GPUs
    for the same price as 1x on-demand!

    Args:
        gpu_type: GPU type to compare (B300, H100, A6000, etc.).
        hours: Training duration in hours.

    Returns:
        Detailed comparison showing multi-GPU spot value.
    """
    if not GPU_OPTIMIZER_AVAILABLE:
        return "❌ GPU optimizer not available."
    return await compare_multi_gpu_spot(gpu_type, hours)


@mcp.tool()
async def find_optimal_gpu(
    model_size_billions: float,
    budget_per_hour: float = 5.0,
    prefer_spot: bool = True,
) -> str:
    """Find the best GPU configuration for your model and budget.

    Analyzes all GPU options and recommends the best value configs.

    Args:
        model_size_billions: Model size in billions (7, 13, 30, 70, etc.).
        budget_per_hour: Maximum budget per hour in USD.
        prefer_spot: Prefer spot instances (default: True for savings!).

    Returns:
        Ranked list of GPU configurations by value.
    """
    if not GPU_OPTIMIZER_AVAILABLE:
        return "❌ GPU optimizer not available."
    return await find_best_gpu_config(model_size_billions, budget_per_hour, prefer_spot)


@mcp.tool()
async def estimate_training(
    model_size_billions: float,
    dataset_tokens_billions: float,
    gpu_type: str = "A6000",
    gpu_count: int = 1,
) -> str:
    """Estimate training time and cost for your configuration.

    Args:
        model_size_billions: Model size in billions of parameters.
        dataset_tokens_billions: Dataset size in billions of tokens.
        gpu_type: GPU type to use.
        gpu_count: Number of GPUs (1, 2, 4, 8).

    Returns:
        Time and cost estimates for spot and on-demand.
    """
    if not GPU_OPTIMIZER_AVAILABLE:
        return "❌ GPU optimizer not available."
    return await estimate_training_time(model_size_billions, dataset_tokens_billions, gpu_type, gpu_count)


@mcp.tool()
async def gpu_catalog() -> str:
    """Show complete GPU catalog with specs, pricing, and recommendations.

    Lists all 12 GPU types with:
    - VRAM, architecture, TFLOPs
    - On-demand and SPOT pricing
    - Multi-GPU options
    - Best use cases

    Returns:
        Complete GPU reference table.
    """
    if not GPU_OPTIMIZER_AVAILABLE:
        return "❌ GPU optimizer not available."
    return await list_all_gpus_detailed()


# =============================================================================
# Live Data (Auto-Updated from API)
# =============================================================================

try:
    from .live_data import (
        check_live_availability,
        check_all_availability,
        get_current_costs,
        get_api_info,
        refresh_data,
    )
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False


@mcp.tool()
async def live_gpu_availability(gpu_type: str = "A6000", gpu_count: int = 1) -> str:
    """Check LIVE availability from Verda API.

    Queries the API in real-time to check spot and on-demand availability
    across all locations (FIN-01, FIN-02, FIN-03).

    Args:
        gpu_type: GPU type to check.
        gpu_count: Number of GPUs (1, 2, 4, 8).

    Returns:
        Live availability status by location.
    """
    if not LIVE_DATA_AVAILABLE:
        return "❌ Live data not available."
    return await check_live_availability(gpu_type, gpu_count)


@mcp.tool()
async def live_all_gpus_availability() -> str:
    """Check LIVE availability for ALL GPU types.

    Scans all GPU types across all locations to show what's
    currently available for spot and on-demand.

    Returns:
        Availability matrix for all GPUs.
    """
    if not LIVE_DATA_AVAILABLE:
        return "❌ Live data not available."
    return await check_all_availability()


@mcp.tool()
async def current_running_costs() -> str:
    """Get cost of currently running instances.

    Checks your running instances and calculates current hourly/daily costs.

    Returns:
        Running instance costs breakdown.
    """
    if not LIVE_DATA_AVAILABLE:
        return "❌ Live data not available."
    return await get_current_costs()


@mcp.tool()
async def api_capabilities() -> str:
    """Show what the Verda API can and cannot provide.

    Explains which data is auto-updated from API vs manually cached.

    Returns:
        API capabilities and data freshness info.
    """
    if not LIVE_DATA_AVAILABLE:
        return "❌ Live data not available."
    return await get_api_info()


@mcp.tool()
async def refresh_live_data() -> str:
    """Refresh all cached data from Verda API.

    Forces a refresh of:
    - GPU availability (all types/locations)
    - Instance statuses
    - Volume statuses

    Returns:
        Refresh status and summary.
    """
    if not LIVE_DATA_AVAILABLE:
        return "❌ Live data not available."
    return await refresh_data()


# =============================================================================
# Advanced Tools (Shared FS, Clusters, Batch Jobs - Beta)
# =============================================================================

try:
    from .advanced_tools import (
        list_shared_filesystems,
        create_shared_filesystem,
        list_clusters,
        create_cluster,
        list_batch_jobs,
        create_batch_job,
        get_batch_job_logs,
    )
    ADVANCED_TOOLS_AVAILABLE = True
except ImportError:
    ADVANCED_TOOLS_AVAILABLE = False


@mcp.tool()
async def shared_filesystems() -> str:
    """List all shared filesystems (Beta).

    Shared filesystems can be mounted to multiple instances simultaneously.
    Ideal for distributed training with shared datasets.

    Returns:
        List of shared filesystems with details.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await list_shared_filesystems()


@mcp.tool()
async def create_shared_fs(
    name: str,
    size_gb: int = 100,
    location: str = "FIN-01",
) -> str:
    """Create a new shared filesystem (Beta).

    Creates a filesystem that can be shared between multiple instances.

    Args:
        name: Name for the filesystem.
        size_gb: Size in GB (default: 100).
        location: Data center location (FIN-01, FIN-02, FIN-03).

    Returns:
        Creation result with filesystem details.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await create_shared_filesystem(name, size_gb, location)


@mcp.tool()
async def gpu_clusters() -> str:
    """List all GPU clusters (Beta).

    Clusters allow multi-node distributed training across multiple servers.

    Returns:
        List of clusters with GPU configurations.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await list_clusters()


@mcp.tool()
async def create_gpu_cluster(
    name: str,
    gpu_type: str = "H100",
    gpus_per_node: int = 8,
    num_nodes: int = 2,
    location: str = "FIN-01",
) -> str:
    """Create a new GPU cluster for distributed training (Beta).

    Sets up a multi-node cluster for large-scale training.

    Args:
        name: Cluster name.
        gpu_type: GPU type (H100, A100, etc.).
        gpus_per_node: GPUs per node (1, 2, 4, 8).
        num_nodes: Number of nodes in cluster.
        location: Data center location.

    Returns:
        Cluster creation result.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await create_cluster(name, gpu_type, gpus_per_node, num_nodes, location)


@mcp.tool()
async def batch_jobs() -> str:
    """List all batch jobs (Beta).

    Batch jobs run training automatically without manual instance management.

    Returns:
        List of batch jobs with status.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await list_batch_jobs()


@mcp.tool()
async def create_training_job(
    name: str,
    command: str,
    gpu_type: str = "A6000",
    gpu_count: int = 1,
    timeout_hours: int = 24,
) -> str:
    """Create a batch training job (Beta).

    Submits a training job that runs automatically.

    Args:
        name: Job name.
        command: Training command to run (e.g., "python train.py").
        gpu_type: GPU type (A6000, H100, etc.).
        gpu_count: Number of GPUs (1, 2, 4, 8).
        timeout_hours: Maximum runtime in hours.

    Returns:
        Job creation result.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await create_batch_job(name, command, gpu_type, gpu_count, timeout_hours)


@mcp.tool()
async def batch_job_logs(job_id: str) -> str:
    """Get logs from a batch job (Beta).

    Retrieve output logs from a running or completed batch job.

    Args:
        job_id: The batch job ID.

    Returns:
        Job logs.
    """
    if not ADVANCED_TOOLS_AVAILABLE:
        return "❌ Advanced tools not available."
    return await get_batch_job_logs(job_id)


# =============================================================================
# Testing & Diagnostics Tools (NEW!)
# =============================================================================

try:
    from .testing_tools import (
        run_api_tests,
        run_ssh_tests,
        run_all_tests,
    )
    TESTING_TOOLS_AVAILABLE = True
except ImportError:
    TESTING_TOOLS_AVAILABLE = False


@mcp.tool()
async def self_test_api() -> str:
    """Run self-diagnostic tests on API and module functionality.

    Tests config, GPU mappings, API calls, cost estimator, log parser,
    and module availability. No instance required.

    Returns:
        Detailed test results with pass/fail status for each test.
    """
    if not TESTING_TOOLS_AVAILABLE:
        return "❌ Testing tools not available."
    return await run_api_tests()


@mcp.tool()
async def self_test_ssh(instance_ip: str) -> str:
    """Run self-diagnostic tests on SSH/remote functionality.

    Tests SSH connection, GPU status, command execution, file operations,
    directory listing, and health checks on a running instance.

    Args:
        instance_ip: IP address of a running instance to test against.

    Returns:
        Detailed test results with pass/fail status for each test.
    """
    if not TESTING_TOOLS_AVAILABLE or not SSH_TOOLS_AVAILABLE:
        return "❌ Testing or SSH tools not available."
    return await run_ssh_tests(instance_ip)


@mcp.tool()
async def self_test_all(instance_ip: str = "") -> str:
    """Run ALL self-diagnostic tests (API + SSH if instance provided).

    Comprehensive test of all MCP functionality including:
    - Configuration loading
    - GPU type mappings
    - API calls (instances, volumes, keys, images)
    - Availability checking
    - Cost estimation
    - Log parsing
    - Module availability (GDrive, WatchDog, SSH)
    - SSH operations (if instance_ip provided)

    Args:
        instance_ip: Optional IP of running instance for SSH tests.

    Returns:
        Complete test report with timing and detailed results.
    """
    if not TESTING_TOOLS_AVAILABLE:
        return "❌ Testing tools not available."
    ip = instance_ip if instance_ip else None
    return await run_all_tests(ip)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    logger.info("=" * 60)
    logger.info("🚀 Verda Cloud MCP Server (Enhanced Edition)")
    logger.info("   Full GPU Training Automation Suite")
    logger.info("=" * 60)
    
    # Feature status
    features = []
    if SSH_TOOLS_AVAILABLE:
        features.append("✅ SSH Remote Access (8 tools)")
    else:
        features.append("❌ SSH tools - install paramiko")
    
    if GDRIVE_TOOLS_AVAILABLE:
        features.append("✅ Google Drive Integration (6 tools)")
    else:
        features.append("❌ Google Drive - install gdown")
    
    if WATCHDOG_AVAILABLE:
        features.append("✅ WatchDog Monitoring (5 tools)")
    else:
        features.append("❌ WatchDog not available")
    
    if EXTENDED_TOOLS_AVAILABLE:
        features.append("✅ Extended Tools (7 tools): Cost, Health, Logs, Checkpoints, GPU List, Recommendations")
    else:
        features.append("❌ Extended tools not available")
    
    if SPOT_MANAGER_AVAILABLE:
        features.append("✅ Spot Manager (6 tools): Auto-Switch, Failover, 75% Savings!")
    else:
        features.append("❌ Spot manager not available")
    
    if TRAINING_TOOLS_AVAILABLE:
        features.append("✅ Training Tools (7 tools): Checkpoints, Scripts, Alerts, Notifications")
    else:
        features.append("❌ Training tools not available")
    
    if SMART_DEPLOYER_AVAILABLE:
        features.append("✅ Smart Deployer (4 tools): Fail-Safes, Best Deals, Auto-Recovery")
    else:
        features.append("❌ Smart deployer not available")
    
    if GPU_OPTIMIZER_AVAILABLE:
        features.append("✅ GPU Optimizer (4 tools): Multi-GPU Spot Comparisons, Training Estimates")
    else:
        features.append("❌ GPU optimizer not available")
    
    if LIVE_DATA_AVAILABLE:
        features.append("✅ Live Data (5 tools): API Auto-Update, Availability, Costs")
    else:
        features.append("❌ Live data not available")
    
    if ADVANCED_TOOLS_AVAILABLE:
        features.append("✅ Advanced Tools (7 tools): Shared FS, Clusters, Batch Jobs (Beta)")
    else:
        features.append("❌ Advanced tools not available")
    
    if TESTING_TOOLS_AVAILABLE:
        features.append("✅ Testing Tools (3 tools): Self-diagnostics")
    else:
        features.append("❌ Testing tools not available")
    
    for f in features:
        logger.info(f)
    
    # Count total tools
    base_tools = 20  # Original tools
    ssh_tools = 8 if SSH_TOOLS_AVAILABLE else 0
    gdrive_tools = 6 if GDRIVE_TOOLS_AVAILABLE else 0
    watchdog_tools = 5 if WATCHDOG_AVAILABLE else 0
    extended_tools = 7 if EXTENDED_TOOLS_AVAILABLE else 0
    spot_tools = 6 if SPOT_MANAGER_AVAILABLE else 0
    training_tools = 7 if TRAINING_TOOLS_AVAILABLE else 0
    smart_deployer_tools = 4 if SMART_DEPLOYER_AVAILABLE else 0  # best_deals, power_deals, deploy_failsafe, available_now
    training_intel_tools = 5 if TRAINING_INTEL_AVAILABLE else 0  # 5 mega-tools with 55+ bundled functions
    gpu_optimizer_tools = 4 if GPU_OPTIMIZER_AVAILABLE else 0
    live_data_tools = 5 if LIVE_DATA_AVAILABLE else 0
    advanced_tools = 7 if ADVANCED_TOOLS_AVAILABLE else 0
    testing_tools = 3 if TESTING_TOOLS_AVAILABLE else 0
    total = base_tools + ssh_tools + gdrive_tools + watchdog_tools + extended_tools + spot_tools + training_tools + smart_deployer_tools + gpu_optimizer_tools + live_data_tools + advanced_tools + testing_tools
    
    logger.info(f"📊 Total Tools Available: {total}")
    logger.info("=" * 60)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

