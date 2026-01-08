"""Advanced Tools for Verda MCP Server.

Beta features including:
- Shared Filesystems (multi-instance storage)
- Clusters (multi-node training)
- Batch Jobs (scheduled training)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED FILESYSTEM TOOLS
# =============================================================================


@dataclass
class SharedFilesystem:
    """Shared filesystem configuration."""

    id: str
    name: str
    size_gb: int
    location: str
    status: str
    mount_path: str = "/mnt/shared"
    attached_instances: List[str] = None

    def __post_init__(self):
        if self.attached_instances is None:
            self.attached_instances = []


class SharedFilesystemManager:
    """Manage shared filesystems for multi-instance training."""

    async def list_shared_filesystems(self) -> List[Dict[str, Any]]:
        """List all shared filesystems."""
        try:
            from .client import get_client

            client = get_client()

            # Note: This depends on Verda API having shared filesystem endpoints
            try:
                filesystems = await client.list_shared_filesystems()
                return [
                    {
                        "id": fs.id,
                        "name": fs.name,
                        "size_gb": fs.size_gb,
                        "location": fs.location,
                        "status": fs.status,
                        "instances": getattr(fs, "attached_instances", []),
                    }
                    for fs in filesystems
                ]
            except AttributeError:
                return []
        except Exception as e:
            logger.error(f"Error listing shared filesystems: {e}")
            return []

    async def create_shared_filesystem(
        self,
        name: str,
        size_gb: int = 100,
        location: str = "FIN-01",
    ) -> Dict[str, Any]:
        """Create a new shared filesystem."""
        try:
            from .client import get_client

            client = get_client()

            try:
                fs = await client.create_shared_filesystem(
                    name=name,
                    size_gb=size_gb,
                    location=location,
                )
                return {
                    "success": True,
                    "id": fs.id,
                    "name": name,
                    "size_gb": size_gb,
                    "location": location,
                    "mount_path": "/mnt/shared",
                }
            except AttributeError:
                return {
                    "success": False,
                    "error": "Shared filesystem API not available. Use Verda console.",
                    "console_url": "https://console.verda.com/storage/shared-filesystems",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def attach_to_instance(
        self,
        filesystem_id: str,
        instance_id: str,
    ) -> Dict[str, Any]:
        """Attach shared filesystem to an instance."""
        try:
            from .client import get_client

            client = get_client()

            try:
                await client.attach_shared_filesystem(filesystem_id, instance_id)
                return {
                    "success": True,
                    "filesystem_id": filesystem_id,
                    "instance_id": instance_id,
                    "mount_path": "/mnt/shared",
                }
            except AttributeError:
                return {
                    "success": False,
                    "error": "Use Verda console to attach shared filesystems.",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format_filesystems(self, filesystems: List[Dict[str, Any]]) -> str:
        """Format filesystem list for display."""
        if not filesystems:
            return """# 📁 Shared Filesystems

No shared filesystems found.

## Create One
Use `create_shared_filesystem` or visit:
https://console.verda.com/storage/shared-filesystems

## Benefits
- Share data between multiple instances
- Persist data across instance restarts
- Ideal for distributed training
"""

        lines = [
            "# 📁 Shared Filesystems",
            "",
            "| Name | Size | Location | Status | Instances |",
            "|------|------|----------|--------|-----------|",
        ]

        for fs in filesystems:
            instances = ", ".join(fs.get("instances", [])) or "None"
            lines.append(f"| {fs['name']} | {fs['size_gb']}GB | {fs['location']} | {fs['status']} | {instances} |")

        lines.extend(
            [
                "",
                "## Mount Path",
                "Shared filesystems are mounted at `/mnt/shared`",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# CLUSTER TOOLS (BETA)
# =============================================================================


@dataclass
class ClusterConfig:
    """Cluster configuration for multi-node training."""

    name: str
    gpu_type: str
    gpu_count_per_node: int
    num_nodes: int
    location: str = "FIN-01"
    shared_filesystem_id: Optional[str] = None


class ClusterManager:
    """Manage GPU clusters for distributed training."""

    async def list_clusters(self) -> List[Dict[str, Any]]:
        """List all clusters."""
        try:
            from .client import get_client

            client = get_client()

            try:
                clusters = await client.list_clusters()
                return [
                    {
                        "id": c.id,
                        "name": c.name,
                        "gpu_type": c.gpu_type,
                        "nodes": c.num_nodes,
                        "gpus_per_node": c.gpu_count,
                        "total_gpus": c.num_nodes * c.gpu_count,
                        "status": c.status,
                    }
                    for c in clusters
                ]
            except AttributeError:
                return []
        except Exception as e:
            logger.error(f"Error listing clusters: {e}")
            return []

    async def create_cluster(self, config: ClusterConfig) -> Dict[str, Any]:
        """Create a new cluster."""
        try:
            from .client import get_client

            client = get_client()

            try:
                cluster = await client.create_cluster(
                    name=config.name,
                    gpu_type=config.gpu_type,
                    gpu_count=config.gpu_count_per_node,
                    num_nodes=config.num_nodes,
                    location=config.location,
                    shared_filesystem_id=config.shared_filesystem_id,
                )
                return {
                    "success": True,
                    "id": cluster.id,
                    "name": config.name,
                    "total_gpus": config.gpu_count_per_node * config.num_nodes,
                }
            except AttributeError:
                return {
                    "success": False,
                    "error": "Cluster API is in beta. Use Verda console.",
                    "console_url": "https://console.verda.com/clusters",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Delete a cluster."""
        try:
            from .client import get_client

            client = get_client()

            try:
                await client.delete_cluster(cluster_id)
                return {"success": True, "cluster_id": cluster_id}
            except AttributeError:
                return {
                    "success": False,
                    "error": "Use Verda console to delete clusters.",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format_clusters(self, clusters: List[Dict[str, Any]]) -> str:
        """Format cluster list for display."""
        if not clusters:
            return """# 🔗 GPU Clusters (Beta)

No clusters found.

## What are Clusters?
Clusters allow multi-node distributed training across multiple GPU servers.

## Benefits
- Train larger models with data parallelism
- Faster training with more GPUs
- NVLink interconnect for fast communication

## Create a Cluster
Visit: https://console.verda.com/clusters

## Example Use Cases
- 70B+ model training across 8 nodes
- Data parallel training for faster convergence
- Model parallel training for huge models
"""

        lines = [
            "# 🔗 GPU Clusters",
            "",
            "| Name | GPU Type | Nodes | GPUs/Node | Total GPUs | Status |",
            "|------|----------|-------|-----------|------------|--------|",
        ]

        for c in clusters:
            lines.append(
                f"| {c['name']} | {c['gpu_type']} | {c['nodes']} | {c['gpus_per_node']} | {c['total_gpus']} | {c['status']} |"
            )

        return "\n".join(lines)


# =============================================================================
# BATCH JOBS TOOLS (BETA)
# =============================================================================


@dataclass
class BatchJobConfig:
    """Batch job configuration."""

    name: str
    gpu_type: str
    gpu_count: int
    command: str
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    timeout_hours: int = 24
    volume_id: Optional[str] = None
    environment: Dict[str, str] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = {}


class BatchJobManager:
    """Manage batch jobs for automated training."""

    async def list_batch_jobs(self) -> List[Dict[str, Any]]:
        """List all batch jobs."""
        try:
            from .client import get_client

            client = get_client()

            try:
                jobs = await client.list_batch_jobs()
                return [
                    {
                        "id": j.id,
                        "name": j.name,
                        "status": j.status,
                        "gpu_type": j.gpu_type,
                        "gpu_count": j.gpu_count,
                        "created_at": j.created_at,
                        "started_at": getattr(j, "started_at", None),
                        "completed_at": getattr(j, "completed_at", None),
                    }
                    for j in jobs
                ]
            except AttributeError:
                return []
        except Exception as e:
            logger.error(f"Error listing batch jobs: {e}")
            return []

    async def create_batch_job(self, config: BatchJobConfig) -> Dict[str, Any]:
        """Create a new batch job."""
        try:
            from .client import get_client

            client = get_client()

            try:
                job = await client.create_batch_job(
                    name=config.name,
                    gpu_type=config.gpu_type,
                    gpu_count=config.gpu_count,
                    command=config.command,
                    docker_image=config.docker_image,
                    timeout_hours=config.timeout_hours,
                    volume_id=config.volume_id,
                    environment=config.environment,
                )
                return {
                    "success": True,
                    "id": job.id,
                    "name": config.name,
                    "status": "queued",
                }
            except AttributeError:
                return {
                    "success": False,
                    "error": "Batch Jobs API is in beta. Use Verda console.",
                    "console_url": "https://console.verda.com/batch-jobs",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cancel_batch_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a batch job."""
        try:
            from .client import get_client

            client = get_client()

            try:
                await client.cancel_batch_job(job_id)
                return {"success": True, "job_id": job_id, "status": "cancelled"}
            except AttributeError:
                return {
                    "success": False,
                    "error": "Use Verda console to cancel batch jobs.",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_job_logs(self, job_id: str) -> str:
        """Get logs from a batch job."""
        try:
            from .client import get_client

            client = get_client()

            try:
                logs = await client.get_batch_job_logs(job_id)
                return logs
            except AttributeError:
                return "Batch job logs not available via API. Check Verda console."
        except Exception as e:
            return f"Error getting logs: {e}"

    def format_jobs(self, jobs: List[Dict[str, Any]]) -> str:
        """Format batch jobs list for display."""
        if not jobs:
            return """# 📋 Batch Jobs (Beta)

No batch jobs found.

## What are Batch Jobs?
Batch jobs run training scripts automatically without manual instance management.

## Benefits
- Set it and forget it training
- Automatic resource cleanup
- Cost-effective for long training runs
- Queue multiple jobs

## Create a Batch Job
Visit: https://console.verda.com/batch-jobs

## Example
```bash
python train.py --model llama-7b --epochs 3
```
"""

        lines = [
            "# 📋 Batch Jobs",
            "",
            "| Name | GPU | Status | Created | Started | Completed |",
            "|------|-----|--------|---------|---------|-----------|",
        ]

        for j in jobs:
            gpu = f"{j['gpu_count']}x {j['gpu_type']}"
            created = j.get("created_at", "N/A")[:10] if j.get("created_at") else "N/A"
            started = j.get("started_at", "-")[:10] if j.get("started_at") else "-"
            completed = j.get("completed_at", "-")[:10] if j.get("completed_at") else "-"
            lines.append(f"| {j['name']} | {gpu} | {j['status']} | {created} | {started} | {completed} |")

        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_fs_manager: Optional[SharedFilesystemManager] = None
_cluster_manager: Optional[ClusterManager] = None
_batch_manager: Optional[BatchJobManager] = None


def get_filesystem_manager() -> SharedFilesystemManager:
    global _fs_manager
    if _fs_manager is None:
        _fs_manager = SharedFilesystemManager()
    return _fs_manager


def get_cluster_manager() -> ClusterManager:
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


def get_batch_manager() -> BatchJobManager:
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchJobManager()
    return _batch_manager


# =============================================================================
# ASYNC WRAPPER FUNCTIONS FOR MCP TOOLS
# =============================================================================


async def list_shared_filesystems() -> str:
    """List all shared filesystems."""
    manager = get_filesystem_manager()
    filesystems = await manager.list_shared_filesystems()
    return manager.format_filesystems(filesystems)


async def create_shared_filesystem(
    name: str,
    size_gb: int = 100,
    location: str = "FIN-01",
) -> str:
    """Create a new shared filesystem."""
    manager = get_filesystem_manager()
    result = await manager.create_shared_filesystem(name, size_gb, location)

    if result.get("success"):
        return f"""# ✅ Shared Filesystem Created

**ID**: {result["id"]}
**Name**: {result["name"]}
**Size**: {result["size_gb"]} GB
**Location**: {result["location"]}
**Mount Path**: {result["mount_path"]}

## Next Steps
1. Attach to your instances
2. Access data at `/mnt/shared`
"""
    else:
        return f"""# ❌ Creation Failed

{result.get("error", "Unknown error")}

## Alternative
{result.get("console_url", "Visit Verda console")}
"""


async def list_clusters() -> str:
    """List all clusters."""
    manager = get_cluster_manager()
    clusters = await manager.list_clusters()
    return manager.format_clusters(clusters)


async def create_cluster(
    name: str,
    gpu_type: str = "H100",
    gpu_count_per_node: int = 8,
    num_nodes: int = 2,
    location: str = "FIN-01",
) -> str:
    """Create a new cluster."""
    manager = get_cluster_manager()
    config = ClusterConfig(
        name=name,
        gpu_type=gpu_type,
        gpu_count_per_node=gpu_count_per_node,
        num_nodes=num_nodes,
        location=location,
    )
    result = await manager.create_cluster(config)

    if result.get("success"):
        return f"""# ✅ Cluster Created

**ID**: {result["id"]}
**Name**: {result["name"]}
**Total GPUs**: {result["total_gpus"]}

## Distributed Training Ready!
Your cluster is being provisioned.
"""
    else:
        return f"""# ❌ Creation Failed

{result.get("error", "Unknown error")}

## Alternative
{result.get("console_url", "Visit Verda console")}
"""


async def list_batch_jobs() -> str:
    """List all batch jobs."""
    manager = get_batch_manager()
    jobs = await manager.list_batch_jobs()
    return manager.format_jobs(jobs)


async def create_batch_job(
    name: str,
    command: str,
    gpu_type: str = "A6000",
    gpu_count: int = 1,
    timeout_hours: int = 24,
) -> str:
    """Create a new batch job."""
    manager = get_batch_manager()
    config = BatchJobConfig(
        name=name,
        command=command,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        timeout_hours=timeout_hours,
    )
    result = await manager.create_batch_job(config)

    if result.get("success"):
        return f"""# ✅ Batch Job Created

**ID**: {result["id"]}
**Name**: {result["name"]}
**Status**: {result["status"]}

Your job is queued and will start when resources are available.
"""
    else:
        return f"""# ❌ Creation Failed

{result.get("error", "Unknown error")}

## Alternative
{result.get("console_url", "Visit Verda console")}
"""


async def get_batch_job_logs(job_id: str) -> str:
    """Get logs from a batch job."""
    manager = get_batch_manager()
    logs = await manager.get_job_logs(job_id)
    return f"""# 📜 Batch Job Logs

**Job ID**: {job_id}

```
{logs}
```
"""
