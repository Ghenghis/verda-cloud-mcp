"""SSH Tools for Verda MCP Server - Remote instance management."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import paramiko for SSH
try:
    import paramiko

    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko not installed. SSH tools will be limited.")


class SSHManager:
    """Manages SSH connections to Verda instances."""

    def __init__(self, ssh_key_path: Optional[str] = None):
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()
        self._connections: dict[str, paramiko.SSHClient] = {}

    def _find_ssh_key(self) -> str:
        """Find the Verda SSH key."""
        possible_paths = [
            Path.home() / ".ssh" / "verda_key",
            Path.home() / ".ssh" / "id_ed25519",
            Path.home() / ".ssh" / "id_rsa",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError("No SSH key found. Please specify ssh_key_path.")

    def connect(self, host: str, username: str = "root", port: int = 22) -> paramiko.SSHClient:
        """Establish SSH connection to a host."""
        if not PARAMIKO_AVAILABLE:
            raise RuntimeError("paramiko is not installed. Run: pip install paramiko")

        if host in self._connections:
            # Check if connection is still alive
            try:
                self._connections[host].exec_command("echo test", timeout=5)
                return self._connections[host]
            except Exception:
                self._connections[host].close()
                del self._connections[host]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                key_filename=self.ssh_key_path,
                timeout=30,
            )
            self._connections[host] = client
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {host}: {e}")

    def disconnect(self, host: str):
        """Close SSH connection to a host."""
        if host in self._connections:
            self._connections[host].close()
            del self._connections[host]

    def disconnect_all(self):
        """Close all SSH connections."""
        for host in list(self._connections.keys()):
            self.disconnect(host)

    def run_command(self, host: str, command: str, timeout: int = 300) -> tuple[str, str, int]:
        """Run a command on the remote host.

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        client = self.connect(host)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")

        return stdout_text, stderr_text, exit_code

    def read_file(self, host: str, remote_path: str, max_lines: Optional[int] = None) -> str:
        """Read a file from the remote host."""
        if max_lines:
            command = f"tail -n {max_lines} {remote_path}"
        else:
            command = f"cat {remote_path}"

        stdout, stderr, exit_code = self.run_command(host, command)
        if exit_code != 0:
            raise FileNotFoundError(f"Failed to read {remote_path}: {stderr}")
        return stdout

    def write_file(self, host: str, remote_path: str, content: str) -> bool:
        """Write content to a file on the remote host."""
        client = self.connect(host)
        sftp = client.open_sftp()

        try:
            with sftp.file(remote_path, "w") as f:
                f.write(content)
            return True
        finally:
            sftp.close()

    def append_file(self, host: str, remote_path: str, content: str) -> bool:
        """Append content to a file on the remote host."""
        client = self.connect(host)
        sftp = client.open_sftp()

        try:
            with sftp.file(remote_path, "a") as f:
                f.write(content)
            return True
        finally:
            sftp.close()

    def list_dir(self, host: str, remote_path: str) -> list[str]:
        """List files in a directory on the remote host."""
        stdout, stderr, exit_code = self.run_command(host, f"ls -la {remote_path}")
        if exit_code != 0:
            raise FileNotFoundError(f"Failed to list {remote_path}: {stderr}")
        return stdout

    def file_exists(self, host: str, remote_path: str) -> bool:
        """Check if a file exists on the remote host."""
        _, _, exit_code = self.run_command(host, f"test -e {remote_path}")
        return exit_code == 0

    def get_gpu_status(self, host: str) -> str:
        """Get nvidia-smi output from the remote host."""
        stdout, stderr, exit_code = self.run_command(host, "nvidia-smi")
        if exit_code != 0:
            return f"Failed to get GPU status: {stderr}"
        return stdout

    def get_training_logs(
        self,
        host: str,
        log_path: str = "/workspace/outputs/training.log",
        lines: int = 50,
    ) -> str:
        """Get recent training logs from the remote host."""
        stdout, stderr, exit_code = self.run_command(host, f"tail -n {lines} {log_path}")
        if exit_code != 0:
            # Try alternative paths
            alt_paths = [
                "/workspace/training.log",
                "/workspace/outputs/*.log",
                "/workspace/nohup.out",
            ]
            for alt in alt_paths:
                stdout, stderr, exit_code = self.run_command(host, f"tail -n {lines} {alt} 2>/dev/null")
                if exit_code == 0 and stdout.strip():
                    return stdout
            return f"No training logs found. Tried: {log_path}, {alt_paths}"
        return stdout

    def get_training_progress(self, host: str) -> str:
        """Get comprehensive training progress info."""
        results = []

        # GPU status
        gpu_status = self.get_gpu_status(host)
        results.append("## GPU Status")
        results.append("```")
        results.append(gpu_status[:2000])  # Limit output
        results.append("```")

        # Check if training is running
        stdout, _, _ = self.run_command(host, "ps aux | grep -E 'python.*train' | grep -v grep")
        if stdout.strip():
            results.append("\n## Training Process: ✅ RUNNING")
            results.append("```")
            results.append(stdout[:500])
            results.append("```")
        else:
            results.append("\n## Training Process: ❌ NOT RUNNING")

        # Recent logs
        logs = self.get_training_logs(host, lines=30)
        results.append("\n## Recent Logs (last 30 lines)")
        results.append("```")
        results.append(logs[:3000])  # Limit output
        results.append("```")

        # Disk space
        stdout, _, _ = self.run_command(host, "df -h /workspace")
        results.append("\n## Disk Space")
        results.append("```")
        results.append(stdout)
        results.append("```")

        return "\n".join(results)

    def kill_training(self, host: str) -> str:
        """Kill any running training processes."""
        stdout, stderr, exit_code = self.run_command(host, "pkill -f 'python.*train'")
        if exit_code == 0:
            return "Training process killed successfully."
        elif exit_code == 1:
            return "No training process found to kill."
        else:
            return f"Error killing process: {stderr}"


# Global SSH manager instance
_ssh_manager: Optional[SSHManager] = None


def get_ssh_manager() -> SSHManager:
    """Get the global SSH manager instance."""
    global _ssh_manager
    if _ssh_manager is None:
        _ssh_manager = SSHManager()
    return _ssh_manager


# =============================================================================
# Async wrappers for MCP tools
# =============================================================================


async def ssh_run_command(host: str, command: str, timeout: int = 300) -> str:
    """Run a command on the remote instance."""
    manager = get_ssh_manager()

    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    stdout, stderr, exit_code = await loop.run_in_executor(None, lambda: manager.run_command(host, command, timeout))

    result = [f"# Command Output\n**Command**: `{command}`\n**Exit Code**: {exit_code}\n"]

    if stdout.strip():
        result.append("## stdout")
        result.append("```")
        result.append(stdout[:5000])  # Limit output
        result.append("```")

    if stderr.strip():
        result.append("\n## stderr")
        result.append("```")
        result.append(stderr[:2000])
        result.append("```")

    return "\n".join(result)


async def ssh_get_gpu_status(host: str) -> str:
    """Get GPU status from nvidia-smi."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.get_gpu_status(host))
    return f"# GPU Status\n```\n{result}\n```"


async def ssh_get_training_logs(host: str, lines: int = 50) -> str:
    """Get recent training logs."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.get_training_logs(host, lines=lines))
    return f"# Training Logs (last {lines} lines)\n```\n{result}\n```"


async def ssh_get_training_progress(host: str) -> str:
    """Get comprehensive training progress."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.get_training_progress(host))
    return f"# Training Progress Report\n{result}"


async def ssh_read_file(host: str, path: str, max_lines: Optional[int] = None) -> str:
    """Read a file from the remote instance."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.read_file(host, path, max_lines))
    return f"# File: {path}\n```\n{result}\n```"


async def ssh_write_file(host: str, path: str, content: str) -> str:
    """Write a file to the remote instance."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, lambda: manager.write_file(host, path, content))
    if success:
        return f"✅ Successfully wrote to {path}"
    else:
        return f"❌ Failed to write to {path}"


async def ssh_list_dir(host: str, path: str) -> str:
    """List directory contents on the remote instance."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.list_dir(host, path))
    return f"# Directory: {path}\n```\n{result}\n```"


async def ssh_kill_training(host: str) -> str:
    """Kill training process on the remote instance."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.kill_training(host))
    return result
