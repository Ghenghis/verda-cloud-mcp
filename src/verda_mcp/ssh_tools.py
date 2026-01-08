"""SSH Tools for Verda MCP Server - Remote instance management with anti-lockup safeguards."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Global thread pool for SSH operations - prevents blocking
_ssh_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ssh_")

# Default timeouts (seconds)
SSH_CONNECT_TIMEOUT = 10
SSH_COMMAND_TIMEOUT = 15
SSH_ASYNC_TIMEOUT = 20

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

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
        """Establish SSH connection to a host with timeout safeguards."""
        if not PARAMIKO_AVAILABLE:
            raise RuntimeError("paramiko is not installed. Run: pip install paramiko")

        logger.info(f"[SSH] Connecting to {host}...")
        start = time.time()
        
        # Check existing connection
        if host in self._connections:
            try:
                transport = self._connections[host].get_transport()
                if transport and transport.is_active():
                    logger.info(f"[SSH] Reusing connection to {host}")
                    return self._connections[host]
            except Exception:
                pass
            # Close stale connection
            try:
                self._connections[host].close()
            except Exception:
                pass
            del self._connections[host]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                key_filename=self.ssh_key_path,
                timeout=SSH_CONNECT_TIMEOUT,
                auth_timeout=SSH_CONNECT_TIMEOUT,
                banner_timeout=SSH_CONNECT_TIMEOUT,
            )
            self._connections[host] = client
            elapsed = time.time() - start
            logger.info(f"[SSH] Connected to {host} in {elapsed:.1f}s")
            return client
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[SSH] Connection to {host} FAILED after {elapsed:.1f}s: {e}")
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

    def run_command(self, host: str, command: str, timeout: int = SSH_COMMAND_TIMEOUT) -> tuple[str, str, int]:
        """Run a command on the remote host with anti-lockup safeguards.

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        start = time.time()
        cmd_short = command[:50] + "..." if len(command) > 50 else command
        logger.info(f"[SSH] Running on {host}: {cmd_short}")
        
        try:
            client = self.connect(host)
            
            # Execute with timeout
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Set channel timeouts to prevent blocking reads
            channel = stdout.channel
            channel.settimeout(timeout)
            channel.setblocking(0)  # Non-blocking
            
            # Wait for command with timeout
            start_wait = time.time()
            while not channel.exit_status_ready():
                if time.time() - start_wait > timeout:
                    logger.warning(f"[SSH] Command timeout after {timeout}s on {host}")
                    channel.close()
                    return f"TIMEOUT after {timeout}s", "", -1
                time.sleep(0.1)
            
            exit_code = channel.recv_exit_status()
            
            # Read output with size limit
            stdout_text = ""
            stderr_text = ""
            try:
                channel.setblocking(1)
                channel.settimeout(2)  # Short timeout for reading
                stdout_text = stdout.read(50000).decode("utf-8", errors="replace")  # 50KB limit
                stderr_text = stderr.read(10000).decode("utf-8", errors="replace")  # 10KB limit
            except Exception:
                pass
            
            elapsed = time.time() - start
            logger.info(f"[SSH] Command completed in {elapsed:.1f}s, exit={exit_code}")
            return stdout_text, stderr_text, exit_code
            
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[SSH] Command FAILED after {elapsed:.1f}s on {host}: {e}")
            return f"Error: {str(e)}", "", -1

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

    def upload_file(self, host: str, local_path: str, remote_path: str, 
                    progress_callback: Optional[Callable[[int, int], None]] = None) -> dict:
        """Upload a file to the remote host with progress tracking.
        
        Args:
            host: Remote host IP
            local_path: Local file path
            remote_path: Remote destination path
            progress_callback: Optional callback(bytes_transferred, total_bytes)
            
        Returns:
            dict with status, bytes_transferred, elapsed_time
        """
        start = time.time()
        logger.info(f"[SSH] Uploading {local_path} to {host}:{remote_path}")
        
        if not os.path.exists(local_path):
            return {"status": "error", "message": f"Local file not found: {local_path}"}
        
        file_size = os.path.getsize(local_path)
        bytes_transferred = [0]  # Use list for closure
        
        def _progress(transferred, total):
            bytes_transferred[0] = transferred
            if progress_callback:
                progress_callback(transferred, total)
        
        try:
            client = self.connect(host)
            sftp = client.open_sftp()
            
            try:
                sftp.put(local_path, remote_path, callback=_progress)
                elapsed = time.time() - start
                speed = file_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
                
                logger.info(f"[SSH] Upload complete: {file_size} bytes in {elapsed:.1f}s ({speed:.1f} MB/s)")
                return {
                    "status": "success",
                    "bytes_transferred": file_size,
                    "elapsed_time": elapsed,
                    "speed_mbps": speed
                }
            finally:
                sftp.close()
                
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[SSH] Upload FAILED after {elapsed:.1f}s: {e}")
            return {
                "status": "error",
                "message": str(e),
                "bytes_transferred": bytes_transferred[0],
                "elapsed_time": elapsed
            }

    def download_file(self, host: str, remote_path: str, local_path: str,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> dict:
        """Download a file from the remote host with progress tracking.
        
        Args:
            host: Remote host IP
            remote_path: Remote file path
            local_path: Local destination path
            progress_callback: Optional callback(bytes_transferred, total_bytes)
            
        Returns:
            dict with status, bytes_transferred, elapsed_time
        """
        start = time.time()
        logger.info(f"[SSH] Downloading {host}:{remote_path} to {local_path}")
        
        bytes_transferred = [0]
        
        def _progress(transferred, total):
            bytes_transferred[0] = transferred
            if progress_callback:
                progress_callback(transferred, total)
        
        try:
            client = self.connect(host)
            sftp = client.open_sftp()
            
            try:
                # Get remote file size first
                file_stat = sftp.stat(remote_path)
                file_size = file_stat.st_size
                
                # Ensure local directory exists
                os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
                
                sftp.get(remote_path, local_path, callback=_progress)
                elapsed = time.time() - start
                speed = file_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
                
                logger.info(f"[SSH] Download complete: {file_size} bytes in {elapsed:.1f}s ({speed:.1f} MB/s)")
                return {
                    "status": "success",
                    "bytes_transferred": file_size,
                    "elapsed_time": elapsed,
                    "speed_mbps": speed
                }
            finally:
                sftp.close()
                
        except FileNotFoundError:
            return {"status": "error", "message": f"Remote file not found: {remote_path}"}
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[SSH] Download FAILED after {elapsed:.1f}s: {e}")
            return {
                "status": "error",
                "message": str(e),
                "bytes_transferred": bytes_transferred[0],
                "elapsed_time": elapsed
            }

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


async def ssh_run_command(host: str, command: str, timeout: int = SSH_COMMAND_TIMEOUT) -> str:
    """Run a command on the remote instance with async timeout protection."""
    manager = get_ssh_manager()
    logger.info(f"[ASYNC-SSH] Starting command on {host}")
    
    try:
        # Run in dedicated thread pool with asyncio timeout wrapper
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            _ssh_executor,
            lambda: manager.run_command(host, command, timeout)
        )
        
        # Double timeout protection: asyncio + internal
        stdout, stderr, exit_code = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        
        result = [f"# Command Output\n**Exit Code**: {exit_code}\n"]
        
        if stdout.strip():
            result.append("```")
            result.append(stdout[:5000])  # Limit output
            result.append("```")
        
        if stderr.strip() and exit_code != 0:
            result.append("\n## stderr")
            result.append("```")
            result.append(stderr[:2000])
            result.append("```")
        
        logger.info(f"[ASYNC-SSH] Command completed on {host}")
        return "\n".join(result)
        
    except asyncio.TimeoutError:
        logger.error(f"[ASYNC-SSH] TIMEOUT on {host} after {SSH_ASYNC_TIMEOUT}s")
        return f"❌ TIMEOUT: Command did not complete within {SSH_ASYNC_TIMEOUT}s\nHost: {host}\nUse shorter commands or run in background with `nohup ... &`"
    except Exception as e:
        logger.error(f"[ASYNC-SSH] ERROR on {host}: {e}")
        return f"❌ ERROR: {str(e)}\nHost: {host}"


async def ssh_get_gpu_status(host: str) -> str:
    """Get GPU status from nvidia-smi with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.get_gpu_status(host))
        result = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        return f"# GPU Status\n```\n{result}\n```"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT getting GPU status from {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_get_training_logs(host: str, lines: int = 50) -> str:
    """Get recent training logs with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.get_training_logs(host, lines=lines))
        result = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        return f"# Training Logs (last {lines} lines)\n```\n{result}\n```"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT getting logs from {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_get_training_progress(host: str) -> str:
    """Get comprehensive training progress with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.get_training_progress(host))
        result = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT * 2)  # Longer for progress
        return f"# Training Progress Report\n{result}"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT getting progress from {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_read_file(host: str, path: str, max_lines: Optional[int] = None) -> str:
    """Read a file from the remote instance with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.read_file(host, path, max_lines))
        result = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        return f"# File: {path}\n```\n{result}\n```"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT reading {path} from {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_write_file(host: str, path: str, content: str) -> str:
    """Write a file to the remote instance with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.write_file(host, path, content))
        success = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        if success:
            return f"✅ Successfully wrote to {path}"
        else:
            return f"❌ Failed to write to {path}"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT writing to {path} on {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_list_dir(host: str, path: str) -> str:
    """List directory contents on the remote instance with timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_ssh_executor, lambda: manager.list_dir(host, path))
        result = await asyncio.wait_for(future, timeout=SSH_ASYNC_TIMEOUT)
        return f"# Directory: {path}\n```\n{result}\n```"
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT listing {path} on {host}"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_kill_training(host: str) -> str:
    """Kill training process on the remote instance."""
    manager = get_ssh_manager()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: manager.kill_training(host))
    return result


async def ssh_upload_file(host: str, local_path: str, remote_path: str) -> str:
    """Upload a file with progress and timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(
            _ssh_executor,
            lambda: manager.upload_file(host, local_path, remote_path)
        )
        result = await asyncio.wait_for(future, timeout=300)  # 5 min for large files
        
        if result["status"] == "success":
            return f"✅ Upload complete: {result['bytes_transferred']} bytes in {result['elapsed_time']:.1f}s ({result['speed_mbps']:.1f} MB/s)"
        else:
            return f"❌ Upload failed: {result.get('message', 'Unknown error')}"
            
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT: Upload did not complete within 5 minutes"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_download_file(host: str, remote_path: str, local_path: str) -> str:
    """Download a file with progress and timeout protection."""
    try:
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(
            _ssh_executor,
            lambda: manager.download_file(host, remote_path, local_path)
        )
        result = await asyncio.wait_for(future, timeout=300)  # 5 min for large files
        
        if result["status"] == "success":
            return f"✅ Download complete: {result['bytes_transferred']} bytes in {result['elapsed_time']:.1f}s ({result['speed_mbps']:.1f} MB/s)\nSaved to: {local_path}"
        else:
            return f"❌ Download failed: {result.get('message', 'Unknown error')}"
            
    except asyncio.TimeoutError:
        return f"❌ TIMEOUT: Download did not complete within 5 minutes"
    except Exception as e:
        return f"❌ ERROR: {e}"


async def ssh_run_with_retry(host: str, command: str, timeout: int = SSH_COMMAND_TIMEOUT, 
                              max_retries: int = MAX_RETRIES) -> tuple[str, str, int]:
    """Run a command with automatic retry on timeout/failure.
    
    Returns:
        Tuple of (stdout, stderr, exit_code)
    """
    manager = get_ssh_manager()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _ssh_executor,
                lambda: manager.run_command(host, command, timeout)
            )
            result = await asyncio.wait_for(future, timeout=timeout + 5)
            
            stdout, stderr, exit_code = result
            
            # Check if it's a timeout response
            if "TIMEOUT" in stdout:
                logger.warning(f"[SSH] Attempt {attempt + 1}/{max_retries} timed out")
                last_error = stdout
                if attempt < max_retries - 1:
                    await asyncio.sleep(RETRY_DELAY)
                continue
            
            # Success
            if attempt > 0:
                logger.info(f"[SSH] Command succeeded on attempt {attempt + 1}")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"[SSH] Attempt {attempt + 1}/{max_retries} async timeout")
            last_error = f"Async timeout after {timeout + 5}s"
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning(f"[SSH] Attempt {attempt + 1}/{max_retries} error: {e}")
            last_error = str(e)
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY)
    
    # All retries failed
    logger.error(f"[SSH] All {max_retries} attempts failed: {last_error}")
    return f"FAILED after {max_retries} attempts: {last_error}", "", -1
