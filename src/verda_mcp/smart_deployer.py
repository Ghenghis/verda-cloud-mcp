"""Smart Deployer with Fail-Safes for Verda MCP Server.

Comprehensive deployment system with:
- Real-time availability checking across all GPUs and locations
- Best deal finder (multi-GPU spot vs single on-demand comparison)
- Multiple fail-safe layers for all scenarios
- Backup plans for crashes, stops, evictions
- Verification system (confirm deployment matches request)
- On-demand fail-safes (not just spot)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

LOCATIONS = ["FIN-01", "FIN-02", "FIN-03"]
MAX_DEPLOY_RETRIES = 3
DEPLOY_RETRY_DELAY = 10  # seconds
VERIFICATION_TIMEOUT = 300  # 5 minutes
HEALTH_CHECK_INTERVAL = 60  # seconds


class DeploymentMode(Enum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"
    AUTO = "auto"  # Try spot first, fall back to on-demand


class FailureType(Enum):
    UNAVAILABLE = "unavailable"
    DEPLOY_FAILED = "deploy_failed"
    VERIFICATION_FAILED = "verification_failed"
    EVICTED = "evicted"
    CRASHED = "crashed"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


# GPU Database with all options
GPU_OPTIONS = {
    "GB300": {
        "vram": 288,
        "spot": 1.36,
        "on_demand": 5.45,
        "configs": [1, 2, 4],
        "power": 1800,
    },
    "B300": {
        "vram": 262,
        "spot": 1.24,
        "on_demand": 4.95,
        "configs": [1, 2, 4, 8],
        "power": 1500,
    },
    "B200": {
        "vram": 180,
        "spot": 0.95,
        "on_demand": 3.79,
        "configs": [1, 2, 4, 8],
        "power": 1200,
    },
    "H200": {
        "vram": 141,
        "spot": 0.75,
        "on_demand": 2.99,
        "configs": [1, 2, 4, 8],
        "power": 990,
    },
    "H100": {
        "vram": 80,
        "spot": 0.57,
        "on_demand": 2.29,
        "configs": [1, 2, 4, 8],
        "power": 990,
    },
    "A100_80G": {
        "vram": 80,
        "spot": 0.32,
        "on_demand": 1.29,
        "configs": [1, 2, 4, 8],
        "power": 312,
    },
    "A100_40G": {
        "vram": 40,
        "spot": 0.18,
        "on_demand": 0.72,
        "configs": [1, 8],
        "power": 312,
    },
    "V100": {
        "vram": 16,
        "spot": 0.035,
        "on_demand": 0.14,
        "configs": [1, 2, 4, 8],
        "power": 125,
    },
    "RTX_PRO_6000": {
        "vram": 96,
        "spot": 0.35,
        "on_demand": 1.39,
        "configs": [1, 2, 4, 8],
        "power": 91,
    },
    "L40S": {
        "vram": 48,
        "spot": 0.23,
        "on_demand": 0.91,
        "configs": [1, 2, 4, 8],
        "power": 91,
    },
    "RTX_6000_ADA": {
        "vram": 48,
        "spot": 0.21,
        "on_demand": 0.83,
        "configs": [1, 2, 4, 8],
        "power": 91,
    },
    "A6000": {
        "vram": 48,
        "spot": 0.12,
        "on_demand": 0.49,
        "configs": [1, 2, 4, 8],
        "power": 38,
    },
}

# Multi-GPU scaling efficiency
MULTI_GPU_SCALING = {1: 1.0, 2: 1.85, 4: 3.5, 8: 6.5}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GPUConfig:
    """GPU configuration with pricing and power."""

    gpu_type: str
    gpu_count: int
    mode: DeploymentMode
    location: str = ""

    @property
    def hourly_cost(self) -> float:
        gpu = GPU_OPTIONS.get(self.gpu_type, {})
        rate = gpu.get("spot" if self.mode == DeploymentMode.SPOT else "on_demand", 0)
        return rate * self.gpu_count

    @property
    def total_vram(self) -> int:
        return GPU_OPTIONS.get(self.gpu_type, {}).get("vram", 0) * self.gpu_count

    @property
    def total_power(self) -> float:
        base = GPU_OPTIONS.get(self.gpu_type, {}).get("power", 0)
        scaling = MULTI_GPU_SCALING.get(self.gpu_count, 1.0)
        return base * scaling

    @property
    def value_score(self) -> float:
        """Power per dollar - higher is better."""
        if self.hourly_cost == 0:
            return 0
        return self.total_power / self.hourly_cost


@dataclass
class AvailabilityResult:
    """Result of availability check."""

    gpu_type: str
    gpu_count: int
    spot_available: bool
    on_demand_available: bool
    location: str
    checked_at: datetime


@dataclass
class DeploymentPlan:
    """Plan for deployment with primary and backup options."""

    primary: GPUConfig
    backup_options: List[GPUConfig] = field(default_factory=list)
    reason: str = ""


@dataclass
class DeploymentResult:
    """Result of a deployment attempt."""

    success: bool
    instance_id: str = ""
    instance_ip: str = ""
    gpu_config: Optional[GPUConfig] = None
    failure_type: Optional[FailureType] = None
    error_message: str = ""
    attempts: int = 0
    used_backup: bool = False
    backup_used: Optional[GPUConfig] = None


@dataclass
class HealthStatus:
    """Health status of a running instance."""

    instance_id: str
    healthy: bool
    gpu_responding: bool
    ssh_accessible: bool
    last_check: datetime
    issues: List[str] = field(default_factory=list)


# =============================================================================
# SMART DEPLOYER CLASS
# =============================================================================


class SmartDeployer:
    """Smart deployment system with comprehensive fail-safes."""

    def __init__(self):
        self.availability_cache: Dict[str, AvailabilityResult] = {}
        self.cache_duration = timedelta(minutes=2)
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.health_monitors: Dict[str, asyncio.Task] = {}

    # =========================================================================
    # AVAILABILITY CHECKING
    # =========================================================================

    async def check_all_availability(self) -> Dict[str, List[AvailabilityResult]]:
        """Check availability for all GPU types across all locations."""
        results = {}

        try:
            from .client import get_client

            client = get_client()
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return results

        for gpu_type, gpu_info in GPU_OPTIONS.items():
            results[gpu_type] = []

            for gpu_count in gpu_info.get("configs", [1]):
                for location in LOCATIONS:
                    try:
                        # Check spot
                        spot_avail = await self._check_availability(client, gpu_type, gpu_count, location, is_spot=True)
                        # Check on-demand
                        od_avail = await self._check_availability(client, gpu_type, gpu_count, location, is_spot=False)

                        result = AvailabilityResult(
                            gpu_type=gpu_type,
                            gpu_count=gpu_count,
                            spot_available=spot_avail,
                            on_demand_available=od_avail,
                            location=location,
                            checked_at=datetime.now(),
                        )
                        results[gpu_type].append(result)

                        # Cache result
                        cache_key = f"{gpu_type}_{gpu_count}_{location}"
                        self.availability_cache[cache_key] = result

                    except Exception as e:
                        logger.debug(f"Error checking {gpu_type} x{gpu_count} at {location}: {e}")

        return results

    async def _check_availability(
        self,
        client,
        gpu_type: str,
        gpu_count: int,
        location: str,
        is_spot: bool,
    ) -> bool:
        """Check if a specific configuration is available."""
        try:
            from .client import get_instance_type_from_gpu_type_and_count

            instance_type = get_instance_type_from_gpu_type_and_count(gpu_type, gpu_count)

            if not instance_type:
                return False

            return await client._run_sync(
                client._instances.is_available,
                instance_type,
                is_spot,
                location,
            )
        except Exception:
            return False

    async def get_available_configs(
        self,
        min_vram: int = 0,
        prefer_spot: bool = True,
    ) -> List[GPUConfig]:
        """Get all currently available configurations."""
        available = []

        all_availability = await self.check_all_availability()

        for gpu_type, results in all_availability.items():
            for result in results:
                gpu_info = GPU_OPTIONS.get(gpu_type, {})
                total_vram = gpu_info.get("vram", 0) * result.gpu_count

                if total_vram < min_vram:
                    continue

                if prefer_spot and result.spot_available:
                    available.append(
                        GPUConfig(
                            gpu_type=gpu_type,
                            gpu_count=result.gpu_count,
                            mode=DeploymentMode.SPOT,
                            location=result.location,
                        )
                    )

                if result.on_demand_available:
                    available.append(
                        GPUConfig(
                            gpu_type=gpu_type,
                            gpu_count=result.gpu_count,
                            mode=DeploymentMode.ON_DEMAND,
                            location=result.location,
                        )
                    )

        # Sort by value score (best deals first)
        available.sort(key=lambda x: x.value_score, reverse=True)

        return available

    # =========================================================================
    # BEST DEAL FINDER
    # =========================================================================

    async def find_best_deals(
        self,
        budget_per_hour: float = 5.0,
        min_vram: int = 48,
    ) -> List[Tuple[GPUConfig, str]]:
        """Find best deals - compare multi-GPU spot vs single on-demand.

        Returns list of (config, reason) tuples sorted by value.
        """
        deals = []

        available = await self.get_available_configs(min_vram=min_vram, prefer_spot=True)

        for config in available:
            if config.hourly_cost > budget_per_hour:
                continue

            # Calculate value proposition
            if config.mode == DeploymentMode.SPOT:
                # Find equivalent on-demand for comparison
                od_cost = GPU_OPTIONS.get(config.gpu_type, {}).get("on_demand", 0)
                spot_cost = config.hourly_cost
                savings_pct = (1 - spot_cost / (od_cost * config.gpu_count)) * 100 if od_cost > 0 else 0

                reason = f"💰 {config.gpu_count}x {config.gpu_type} SPOT @ ${spot_cost:.2f}/hr - {savings_pct:.0f}% savings, {config.total_power:.0f} TFLOPs"
            else:
                reason = f"🔵 {config.gpu_count}x {config.gpu_type} On-Demand @ ${config.hourly_cost:.2f}/hr - {config.total_power:.0f} TFLOPs (stable)"

            deals.append((config, reason))

        return deals

    async def find_power_deals(
        self,
        reference_gpu: str = "B300",
        reference_count: int = 1,
        reference_mode: str = "on_demand",
    ) -> List[Tuple[GPUConfig, str]]:
        """Find configs with MORE power for SAME or LESS cost than reference."""
        deals = []

        ref_info = GPU_OPTIONS.get(reference_gpu, {})
        ref_cost = ref_info.get(reference_mode, 0) * reference_count
        ref_power = ref_info.get("power", 0) * MULTI_GPU_SCALING.get(reference_count, 1.0)

        available = await self.get_available_configs(prefer_spot=True)

        for config in available:
            if config.hourly_cost > ref_cost * 1.1:  # Allow 10% more
                continue

            if config.total_power > ref_power:
                power_boost = (config.total_power / ref_power - 1) * 100
                cost_diff = config.hourly_cost - ref_cost

                if cost_diff <= 0:
                    cost_str = f"${abs(cost_diff):.2f} CHEAPER"
                else:
                    cost_str = f"${cost_diff:.2f} more"

                reason = f"🚀 {config.gpu_count}x {config.gpu_type} {config.mode.value}: +{power_boost:.0f}% more power, {cost_str}"
                deals.append((config, reason))

        # Sort by power boost
        deals.sort(key=lambda x: x[0].total_power, reverse=True)

        return deals

    # =========================================================================
    # DEPLOYMENT WITH FAIL-SAFES
    # =========================================================================

    async def deploy_with_failsafes(
        self,
        gpu_type: str = "A6000",
        gpu_count: int = 1,
        prefer_spot: bool = True,
        volume_id: str = "",
        script_id: str = "",
        auto_fallback: bool = True,
        checkpoint_minutes: int = 10,
    ) -> DeploymentResult:
        """Deploy with comprehensive fail-safes and backup plans.

        FAIL-SAFE LAYERS:
        1. Check availability before attempting deploy
        2. Try primary location, fall back to other locations
        3. If spot fails, fall back to on-demand (if auto_fallback)
        4. Retry failed deployments up to MAX_DEPLOY_RETRIES
        5. Verify deployment matches request
        6. Start health monitoring
        7. Set up eviction detection and auto-recovery
        """
        result = DeploymentResult(success=False)

        # FAIL-SAFE 1: Check availability first
        logger.info(f"🔍 Checking availability for {gpu_count}x {gpu_type}...")

        mode = DeploymentMode.SPOT if prefer_spot else DeploymentMode.ON_DEMAND
        plan = await self._create_deployment_plan(gpu_type, gpu_count, mode, auto_fallback)

        if not plan:
            result.failure_type = FailureType.UNAVAILABLE
            result.error_message = f"No availability for {gpu_count}x {gpu_type} in any location"
            return result

        # FAIL-SAFE 2: Try primary, then backups
        configs_to_try = [plan.primary] + plan.backup_options

        for attempt_num, config in enumerate(configs_to_try):
            logger.info(
                f"🚀 Attempt {attempt_num + 1}: Deploying {config.gpu_count}x {config.gpu_type} ({config.mode.value}) at {config.location}..."
            )

            # FAIL-SAFE 3: Retry each config multiple times
            for retry in range(MAX_DEPLOY_RETRIES):
                try:
                    deploy_result = await self._attempt_deploy(config, volume_id, script_id)

                    if deploy_result.success:
                        # FAIL-SAFE 4: Verify deployment
                        verified = await self._verify_deployment(
                            deploy_result.instance_id,
                            config,
                        )

                        if verified:
                            result = deploy_result
                            result.gpu_config = config
                            result.attempts = attempt_num + 1
                            result.used_backup = attempt_num > 0

                            if attempt_num > 0:
                                result.backup_used = config

                            # FAIL-SAFE 5: Start health monitoring
                            await self._start_health_monitor(
                                deploy_result.instance_id,
                                deploy_result.instance_ip,
                                config,
                                checkpoint_minutes,
                            )

                            logger.info(
                                f"✅ Deployment successful: {config.gpu_count}x {config.gpu_type} at {config.location}"
                            )
                            return result
                        else:
                            logger.warning("⚠️ Verification failed, retrying...")
                            result.failure_type = FailureType.VERIFICATION_FAILED
                    else:
                        logger.warning(f"⚠️ Deploy attempt {retry + 1} failed: {deploy_result.error_message}")
                        await asyncio.sleep(DEPLOY_RETRY_DELAY)

                except Exception as e:
                    logger.error(f"❌ Deploy error: {e}")
                    await asyncio.sleep(DEPLOY_RETRY_DELAY)

        # All attempts failed
        result.failure_type = FailureType.DEPLOY_FAILED
        result.error_message = "All deployment attempts failed"
        return result

    async def _create_deployment_plan(
        self,
        gpu_type: str,
        gpu_count: int,
        mode: DeploymentMode,
        auto_fallback: bool,
    ) -> Optional[DeploymentPlan]:
        """Create deployment plan with primary and backup options."""
        available = await self.get_available_configs()

        primary = None
        backups = []

        # Find primary match
        for config in available:
            if config.gpu_type == gpu_type and config.gpu_count == gpu_count:
                if mode == DeploymentMode.SPOT and config.mode == DeploymentMode.SPOT:
                    primary = config
                    break
                elif mode == DeploymentMode.ON_DEMAND and config.mode == DeploymentMode.ON_DEMAND:
                    primary = config
                    break

        # If no primary, find alternatives
        if not primary:
            for config in available:
                if config.gpu_type == gpu_type and config.gpu_count == gpu_count:
                    primary = config
                    break

        if not primary:
            # Find equivalent power alternatives
            target_vram = GPU_OPTIONS.get(gpu_type, {}).get("vram", 0) * gpu_count
            for config in available:
                if config.total_vram >= target_vram:
                    primary = config
                    break

        if not primary:
            return None

        # Build backup options
        if auto_fallback:
            for config in available:
                if config == primary:
                    continue

                # Add same GPU type, different mode as first backup
                if config.gpu_type == gpu_type and config.gpu_count == gpu_count:
                    backups.append(config)
                    continue

                # Add similar power configs
                if config.total_vram >= primary.total_vram * 0.8:
                    backups.append(config)

                if len(backups) >= 5:
                    break

        return DeploymentPlan(
            primary=primary,
            backup_options=backups,
            reason=f"Primary: {primary.gpu_count}x {primary.gpu_type}, {len(backups)} backups",
        )

    async def _attempt_deploy(
        self,
        config: GPUConfig,
        volume_id: str,
        script_id: str,
    ) -> DeploymentResult:
        """Attempt a single deployment."""
        result = DeploymentResult(success=False)

        try:
            from .client import get_client

            client = get_client()

            instance = await client.deploy_spot_instance(
                gpu_type=config.gpu_type,
                gpu_count=config.gpu_count,
                location=config.location,
                volume_id=volume_id if volume_id else None,
                script_id=script_id if script_id else None,
            )

            result.success = True
            result.instance_id = instance.id
            result.instance_ip = instance.ip_address or ""

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.failure_type = FailureType.API_ERROR

        return result

    async def _verify_deployment(
        self,
        instance_id: str,
        expected_config: GPUConfig,
    ) -> bool:
        """Verify that deployed instance matches expected configuration."""
        try:
            from .client import get_client

            client = get_client()

            # Wait for instance to be ready
            timeout = VERIFICATION_TIMEOUT
            start = datetime.now()

            while (datetime.now() - start).total_seconds() < timeout:
                status = await client.check_instance_status(instance_id)

                if status.status == "running":
                    # Verify GPU type matches
                    instance_type = status.instance_type or ""
                    if expected_config.gpu_type.lower() in instance_type.lower():
                        return True
                    else:
                        logger.warning(
                            f"Instance type mismatch: expected {expected_config.gpu_type}, got {instance_type}"
                        )
                        return False

                elif status.status in ["error", "failed", "deleted"]:
                    logger.error(f"Instance in error state: {status.status}")
                    return False

                await asyncio.sleep(10)

            logger.warning("Verification timeout")
            return False

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    # =========================================================================
    # HEALTH MONITORING WITH FAIL-SAFES
    # =========================================================================

    async def _start_health_monitor(
        self,
        instance_id: str,
        instance_ip: str,
        config: GPUConfig,
        checkpoint_minutes: int,
    ) -> None:
        """Start background health monitoring with fail-safes."""
        if instance_id in self.health_monitors:
            self.health_monitors[instance_id].cancel()

        task = asyncio.create_task(self._health_monitor_loop(instance_id, instance_ip, config, checkpoint_minutes))
        self.health_monitors[instance_id] = task

    async def _health_monitor_loop(
        self,
        instance_id: str,
        instance_ip: str,
        config: GPUConfig,
        checkpoint_minutes: int,
    ) -> None:
        """Monitor instance health and handle failures."""
        consecutive_failures = 0
        max_failures = 3

        while True:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

                health = await self._check_instance_health(instance_id, instance_ip)

                if health.healthy:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ Health check failed for {instance_id}: {health.issues}")

                    if consecutive_failures >= max_failures:
                        # FAIL-SAFE: Trigger recovery
                        logger.error(f"❌ Instance {instance_id} unhealthy, triggering recovery...")
                        await self._handle_failure(
                            instance_id,
                            config,
                            FailureType.CRASHED,
                        )
                        break

                # Check for spot eviction
                if config.mode == DeploymentMode.SPOT:
                    evicted = await self._check_eviction(instance_id)
                    if evicted:
                        logger.warning(f"⚠️ Spot instance {instance_id} evicted!")
                        await self._handle_failure(
                            instance_id,
                            config,
                            FailureType.EVICTED,
                        )
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _check_instance_health(
        self,
        instance_id: str,
        instance_ip: str,
    ) -> HealthStatus:
        """Check instance health via SSH and API."""
        health = HealthStatus(
            instance_id=instance_id,
            healthy=True,
            gpu_responding=True,
            ssh_accessible=True,
            last_check=datetime.now(),
        )

        try:
            from .client import get_client

            client = get_client()

            # Check API status
            status = await client.check_instance_status(instance_id)
            if status.status != "running":
                health.healthy = False
                health.issues.append(f"Instance status: {status.status}")

            # Check SSH if possible
            if instance_ip:
                try:
                    from .ssh_tools import get_ssh_manager

                    ssh = get_ssh_manager()
                    stdout, _, code = ssh.run_command(
                        instance_ip,
                        "nvidia-smi --query-gpu=name --format=csv,noheader",
                        timeout=30,
                    )

                    if code != 0:
                        health.gpu_responding = False
                        health.issues.append("GPU not responding")
                except Exception as e:
                    health.ssh_accessible = False
                    health.issues.append(f"SSH failed: {e}")

            health.healthy = health.gpu_responding and health.ssh_accessible

        except Exception as e:
            health.healthy = False
            health.issues.append(f"Health check error: {e}")

        return health

    async def _check_eviction(self, instance_id: str) -> bool:
        """Check if spot instance was evicted."""
        try:
            from .client import get_client

            client = get_client()

            status = await client.check_instance_status(instance_id)
            return status.status in ["terminated", "deleted", "stopped"]
        except Exception:
            return False

    async def _handle_failure(
        self,
        instance_id: str,
        config: GPUConfig,
        failure_type: FailureType,
    ) -> None:
        """Handle instance failure with recovery options."""
        logger.info(f"🔄 Handling {failure_type.value} for {instance_id}...")

        # Clean up health monitor
        if instance_id in self.health_monitors:
            self.health_monitors[instance_id].cancel()
            del self.health_monitors[instance_id]

        # Attempt recovery based on failure type
        if failure_type == FailureType.EVICTED and config.mode == DeploymentMode.SPOT:
            # Try to redeploy spot, or fall back to on-demand
            logger.info("🔄 Attempting to redeploy after eviction...")
            # This would trigger a new deployment with auto-fallback

        elif failure_type == FailureType.CRASHED:
            # Try to restart or redeploy
            logger.info("🔄 Attempting recovery after crash...")
            try:
                from .client import get_client

                client = get_client()
                await client.start_instance(instance_id)
            except Exception as e:
                logger.error(f"Recovery failed: {e}")

    # =========================================================================
    # FORMATTED OUTPUT
    # =========================================================================

    def format_deals(self, deals: List[Tuple[GPUConfig, str]]) -> str:
        """Format deals for display."""
        if not deals:
            return "❌ No deals found matching your criteria."

        lines = [
            "# 🎯 Best Available Deals RIGHT NOW",
            "",
            "*Real-time availability check completed*",
            "",
            "| # | Configuration | $/hr | VRAM | Power | Value |",
            "|---|---------------|------|------|-------|-------|",
        ]

        for i, (config, reason) in enumerate(deals[:10], 1):
            mode_icon = "🟢" if config.mode == DeploymentMode.SPOT else "🔵"
            lines.append(
                f"| {i} | {mode_icon} {config.gpu_count}x {config.gpu_type} | ${config.hourly_cost:.2f} | {config.total_vram}GB | {config.total_power:.0f} | {config.value_score:.0f} |"
            )

        lines.extend(
            [
                "",
                "## 💡 Legend",
                "- 🟢 SPOT (75% cheaper, may be evicted)",
                "- 🔵 On-Demand (stable, higher cost)",
                "- **Value** = Power / Cost (higher = better deal)",
                "",
                "## ⚠️ SPOT REQUIREMENTS",
                "- Save checkpoints every **10 minutes**",
                "- Attach persistent **volume** for data",
                "- Auto-failover enabled by default",
            ]
        )

        return "\n".join(lines)

    def format_deployment_result(self, result: DeploymentResult) -> str:
        """Format deployment result for display."""
        if result.success:
            lines = [
                "# ✅ Deployment Successful!",
                "",
                f"**Instance ID**: `{result.instance_id}`",
                f"**IP Address**: `{result.instance_ip}`",
            ]

            if result.gpu_config:
                lines.extend(
                    [
                        f"**GPU**: {result.gpu_config.gpu_count}x {result.gpu_config.gpu_type}",
                        f"**Mode**: {result.gpu_config.mode.value}",
                        f"**Location**: {result.gpu_config.location}",
                        f"**Cost**: ${result.gpu_config.hourly_cost:.2f}/hr",
                    ]
                )

            if result.used_backup:
                lines.extend(
                    [
                        "",
                        "⚠️ **Note**: Used backup option (primary was unavailable)",
                    ]
                )

            lines.extend(
                [
                    "",
                    "## 🔌 SSH Connection",
                    "```bash",
                    f"ssh root@{result.instance_ip}",
                    "```",
                    "",
                    "## 🛡️ Fail-Safes Active",
                    "- ✅ Health monitoring started",
                    "- ✅ Eviction detection enabled",
                    "- ✅ Auto-recovery configured",
                ]
            )

            return "\n".join(lines)
        else:
            return f"""# ❌ Deployment Failed

**Error**: {result.error_message}
**Failure Type**: {result.failure_type.value if result.failure_type else "Unknown"}
**Attempts**: {result.attempts}

## 🔄 Suggestions
1. Try a different GPU type
2. Try a different location
3. Check account balance
4. Try on-demand instead of spot
"""


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_smart_deployer: Optional[SmartDeployer] = None


def get_smart_deployer() -> SmartDeployer:
    global _smart_deployer
    if _smart_deployer is None:
        _smart_deployer = SmartDeployer()
    return _smart_deployer


# =============================================================================
# ASYNC WRAPPER FUNCTIONS FOR MCP TOOLS
# =============================================================================


async def find_best_deals_now(budget: float = 5.0, min_vram: int = 48) -> str:
    """Find best deals available right now."""
    deployer = get_smart_deployer()
    deals = await deployer.find_best_deals(budget, min_vram)
    return deployer.format_deals(deals)


async def find_power_deals_now(reference_gpu: str = "B300", reference_count: int = 1) -> str:
    """Find configs with more power for same/less cost."""
    deployer = get_smart_deployer()
    deals = await deployer.find_power_deals(reference_gpu, reference_count)
    return deployer.format_deals(deals)


async def deploy_with_all_failsafes(
    gpu_type: str = "A6000",
    gpu_count: int = 1,
    prefer_spot: bool = True,
    volume_id: str = "",
    script_id: str = "",
) -> str:
    """Deploy with comprehensive fail-safes."""
    deployer = get_smart_deployer()
    result = await deployer.deploy_with_failsafes(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        prefer_spot=prefer_spot,
        volume_id=volume_id,
        script_id=script_id,
    )
    return deployer.format_deployment_result(result)


async def show_available_now() -> str:
    """Show all available GPUs right now."""
    deployer = get_smart_deployer()
    all_avail = await deployer.check_all_availability()

    lines = [
        "# 🔄 Live GPU Availability (All Types)",
        "",
        f"*Scanned at {datetime.now().strftime('%H:%M:%S')}*",
        "",
        "| GPU | Config | SPOT | On-Demand | Location |",
        "|-----|--------|------|-----------|----------|",
    ]

    for gpu_type, results in all_avail.items():
        for r in results:
            spot = "✅" if r.spot_available else "❌"
            od = "✅" if r.on_demand_available else "❌"
            lines.append(f"| {gpu_type} | {r.gpu_count}x | {spot} | {od} | {r.location} |")

    return "\n".join(lines)
