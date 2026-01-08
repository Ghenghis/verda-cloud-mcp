"""Google Drive Tools for Verda MCP Server.

Enables file transfer between local machine, Google Drive, and Verda instances.
"""

import asyncio
import logging
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Check for gdown availability
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


class GoogleDriveManager:
    """Manages Google Drive operations via gdown and gdrive CLI."""
    
    def __init__(self):
        self.downloads_dir = Path.home() / "verda_gdrive_downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_file_id(self, url_or_id: str) -> str:
        """Extract file/folder ID from Google Drive URL or return ID directly."""
        if "drive.google.com" in url_or_id:
            # Extract ID from URL
            if "/folders/" in url_or_id:
                # Folder URL
                parts = url_or_id.split("/folders/")
                if len(parts) > 1:
                    return parts[1].split("?")[0].split("/")[0]
            elif "/file/d/" in url_or_id:
                # File URL
                parts = url_or_id.split("/file/d/")
                if len(parts) > 1:
                    return parts[1].split("/")[0]
            elif "id=" in url_or_id:
                # URL with id parameter
                parts = url_or_id.split("id=")
                if len(parts) > 1:
                    return parts[1].split("&")[0]
        return url_or_id  # Assume it's already an ID
    
    def download_file(self, url_or_id: str, output_path: Optional[str] = None) -> Path:
        """Download a file from Google Drive.
        
        Args:
            url_or_id: Google Drive URL or file ID.
            output_path: Optional output path. If None, downloads to default dir.
        
        Returns:
            Path to downloaded file.
        """
        if not GDOWN_AVAILABLE:
            raise RuntimeError("gdown not installed. Run: pip install gdown")
        
        file_id = self._extract_file_id(url_or_id)
        url = f"https://drive.google.com/uc?id={file_id}"
        
        if output_path:
            output = Path(output_path)
        else:
            output = self.downloads_dir / f"gdrive_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Downloading from Google Drive: {file_id}")
        downloaded = gdown.download(url, str(output), quiet=False)
        
        if downloaded:
            return Path(downloaded)
        else:
            raise RuntimeError(f"Failed to download file: {url_or_id}")
    
    def download_folder(self, url_or_id: str, output_dir: Optional[str] = None) -> Path:
        """Download a folder from Google Drive.
        
        Args:
            url_or_id: Google Drive folder URL or ID.
            output_dir: Optional output directory.
        
        Returns:
            Path to downloaded folder.
        """
        if not GDOWN_AVAILABLE:
            raise RuntimeError("gdown not installed. Run: pip install gdown")
        
        folder_id = self._extract_file_id(url_or_id)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        if output_dir:
            output = Path(output_dir)
        else:
            output = self.downloads_dir / f"gdrive_folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading folder from Google Drive: {folder_id}")
        gdown.download_folder(url, output=str(output), quiet=False)
        
        return output
    
    def list_local_downloads(self) -> List[dict]:
        """List files in the local downloads directory."""
        files = []
        for item in self.downloads_dir.iterdir():
            stat = item.stat()
            files.append({
                "name": item.name,
                "path": str(item),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_dir": item.is_dir(),
            })
        return files


class VerdaFileTransfer:
    """Handles file transfers between local machine and Verda instances."""
    
    def __init__(self, ssh_key_path: Optional[str] = None):
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()
    
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
        raise FileNotFoundError("No SSH key found.")
    
    def upload_file(self, local_path: str, instance_ip: str, remote_path: str) -> bool:
        """Upload a file to a Verda instance via SCP.
        
        Args:
            local_path: Path to local file.
            instance_ip: IP address of the Verda instance.
            remote_path: Destination path on the instance.
        
        Returns:
            True if successful.
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Use scp command
        cmd = [
            "scp",
            "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            str(local),
            f"root@{instance_ip}:{remote_path}"
        ]
        
        logger.info(f"Uploading {local_path} to {instance_ip}:{remote_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SCP upload failed: {result.stderr}")
        
        return True
    
    def download_file(self, instance_ip: str, remote_path: str, local_path: str) -> bool:
        """Download a file from a Verda instance via SCP.
        
        Args:
            instance_ip: IP address of the Verda instance.
            remote_path: Path to file on the instance.
            local_path: Destination path locally.
        
        Returns:
            True if successful.
        """
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "scp",
            "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"root@{instance_ip}:{remote_path}",
            str(local)
        ]
        
        logger.info(f"Downloading {instance_ip}:{remote_path} to {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SCP download failed: {result.stderr}")
        
        return True
    
    def upload_directory(self, local_dir: str, instance_ip: str, remote_dir: str) -> bool:
        """Upload a directory to a Verda instance via SCP.
        
        Args:
            local_dir: Path to local directory.
            instance_ip: IP address of the Verda instance.
            remote_dir: Destination directory on the instance.
        
        Returns:
            True if successful.
        """
        local = Path(local_dir)
        if not local.is_dir():
            raise NotADirectoryError(f"Not a directory: {local_dir}")
        
        cmd = [
            "scp",
            "-r",
            "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            str(local),
            f"root@{instance_ip}:{remote_dir}"
        ]
        
        logger.info(f"Uploading directory {local_dir} to {instance_ip}:{remote_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SCP upload failed: {result.stderr}")
        
        return True


class AutomatedDeployment:
    """Automated deployment and setup for Verda training."""
    
    def __init__(self):
        self.gdrive = GoogleDriveManager()
        self.transfer = VerdaFileTransfer()
    
    async def setup_training_environment(
        self,
        instance_ip: str,
        gdrive_url: str,
        workspace_dir: str = "/workspace",
    ) -> str:
        """Fully automated training environment setup.
        
        1. Installs gdown on instance
        2. Downloads training package from Google Drive
        3. Extracts if zip file
        4. Installs requirements
        5. Returns status
        
        Args:
            instance_ip: IP of the Verda instance.
            gdrive_url: Google Drive folder/file URL.
            workspace_dir: Working directory on instance.
        
        Returns:
            Status report.
        """
        from .ssh_tools import get_ssh_manager
        
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        results = []
        
        # Step 1: Install gdown
        results.append("## Step 1: Installing gdown...")
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, "pip install gdown -q")
        )
        results.append(f"Exit code: {code}")
        
        # Step 2: Download from Google Drive
        results.append("\n## Step 2: Downloading from Google Drive...")
        folder_id = self.gdrive._extract_file_id(gdrive_url)
        
        if "/folders/" in gdrive_url:
            # Folder download
            cmd = f"cd {workspace_dir} && gdown --folder 'https://drive.google.com/drive/folders/{folder_id}'"
        else:
            # File download
            cmd = f"cd {workspace_dir} && gdown 'https://drive.google.com/uc?id={folder_id}'"
        
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, cmd, timeout=600)
        )
        results.append(f"Exit code: {code}")
        if stdout:
            results.append(f"Output: {stdout[:500]}")
        
        # Step 3: Extract zip files
        results.append("\n## Step 3: Extracting zip files...")
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, f"cd {workspace_dir} && unzip -o *.zip 2>/dev/null || echo 'No zip files or already extracted'")
        )
        results.append(stdout[:300] if stdout else "No output")
        
        # Step 4: Install requirements
        results.append("\n## Step 4: Installing requirements...")
        req_paths = [
            f"{workspace_dir}/requirements.txt",
            f"{workspace_dir}/scripts/requirements.txt",
        ]
        for req in req_paths:
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, f"test -f {req} && pip install -r {req} -q")
            )
            if code == 0:
                results.append(f"Installed from: {req}")
                break
        
        # Step 5: List workspace
        results.append("\n## Step 5: Workspace contents...")
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, f"ls -la {workspace_dir}")
        )
        results.append(f"```\n{stdout[:1000]}\n```")
        
        # Step 6: Verify GPU
        results.append("\n## Step 6: GPU verification...")
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, "nvidia-smi --query-gpu=name,memory.total --format=csv")
        )
        results.append(f"```\n{stdout}\n```")
        
        return "\n".join(results)
    
    async def start_training(
        self,
        instance_ip: str,
        script_path: str,
        use_screen: bool = True,
    ) -> str:
        """Start training on a Verda instance.
        
        Args:
            instance_ip: IP of the Verda instance.
            script_path: Path to training script on instance.
            use_screen: Whether to use screen for persistence.
        
        Returns:
            Status message.
        """
        from .ssh_tools import get_ssh_manager
        
        manager = get_ssh_manager()
        loop = asyncio.get_event_loop()
        
        if use_screen:
            # Start in a screen session
            cmd = f"screen -dmS training bash -c 'python {script_path} 2>&1 | tee /workspace/training.log'"
        else:
            # Start with nohup
            cmd = f"nohup python {script_path} > /workspace/training.log 2>&1 &"
        
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, cmd)
        )
        
        # Verify it started
        await asyncio.sleep(2)
        stdout, stderr, code = await loop.run_in_executor(
            None, lambda: manager.run_command(instance_ip, "ps aux | grep -E 'python.*train' | grep -v grep")
        )
        
        if stdout.strip():
            return f"""# ✅ Training Started!

**Script**: {script_path}
**Screen Session**: training (if using screen)
**Log File**: /workspace/training.log

## Process Running:
```
{stdout[:500]}
```

## Commands:
- View logs: `tail -f /workspace/training.log`
- Attach screen: `screen -r training`
- Detach screen: Ctrl+A, D"""
        else:
            return f"""# ❌ Training May Not Have Started

Check the log file:
```
cat /workspace/training.log
```

Or check for errors:
```
{stderr[:500] if stderr else 'No stderr output'}
```"""


# Global instances
_gdrive_manager: Optional[GoogleDriveManager] = None
_transfer_manager: Optional[VerdaFileTransfer] = None
_deployment_manager: Optional[AutomatedDeployment] = None


def get_gdrive_manager() -> GoogleDriveManager:
    global _gdrive_manager
    if _gdrive_manager is None:
        _gdrive_manager = GoogleDriveManager()
    return _gdrive_manager


def get_transfer_manager() -> VerdaFileTransfer:
    global _transfer_manager
    if _transfer_manager is None:
        _transfer_manager = VerdaFileTransfer()
    return _transfer_manager


def get_deployment_manager() -> AutomatedDeployment:
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = AutomatedDeployment()
    return _deployment_manager


# =============================================================================
# Async wrappers for MCP tools
# =============================================================================

async def gdrive_download_to_local(url_or_id: str, output_path: Optional[str] = None) -> str:
    """Download file from Google Drive to local machine."""
    manager = get_gdrive_manager()
    loop = asyncio.get_event_loop()
    
    try:
        path = await loop.run_in_executor(
            None, lambda: manager.download_file(url_or_id, output_path)
        )
        return f"✅ Downloaded to: {path}"
    except Exception as e:
        return f"❌ Download failed: {e}"


async def gdrive_download_folder_to_local(url_or_id: str, output_dir: Optional[str] = None) -> str:
    """Download folder from Google Drive to local machine."""
    manager = get_gdrive_manager()
    loop = asyncio.get_event_loop()
    
    try:
        path = await loop.run_in_executor(
            None, lambda: manager.download_folder(url_or_id, output_dir)
        )
        return f"✅ Downloaded folder to: {path}"
    except Exception as e:
        return f"❌ Download failed: {e}"


async def transfer_local_to_verda(local_path: str, instance_ip: str, remote_path: str) -> str:
    """Upload local file to Verda instance."""
    manager = get_transfer_manager()
    loop = asyncio.get_event_loop()
    
    try:
        await loop.run_in_executor(
            None, lambda: manager.upload_file(local_path, instance_ip, remote_path)
        )
        return f"✅ Uploaded {local_path} to {instance_ip}:{remote_path}"
    except Exception as e:
        return f"❌ Upload failed: {e}"


async def transfer_verda_to_local(instance_ip: str, remote_path: str, local_path: str) -> str:
    """Download file from Verda instance to local machine."""
    manager = get_transfer_manager()
    loop = asyncio.get_event_loop()
    
    try:
        await loop.run_in_executor(
            None, lambda: manager.download_file(instance_ip, remote_path, local_path)
        )
        return f"✅ Downloaded {instance_ip}:{remote_path} to {local_path}"
    except Exception as e:
        return f"❌ Download failed: {e}"


async def auto_setup_training(instance_ip: str, gdrive_url: str) -> str:
    """Fully automated training environment setup."""
    manager = get_deployment_manager()
    return await manager.setup_training_environment(instance_ip, gdrive_url)


async def auto_start_training(instance_ip: str, script_path: str) -> str:
    """Start training on Verda instance."""
    manager = get_deployment_manager()
    return await manager.start_training(instance_ip, script_path)
