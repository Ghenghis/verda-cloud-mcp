"""Testing Tools for Verda MCP Server.

Automated testing suite integrated directly into the MCP.
Allows testing all features with a single command.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class TestResult:
    """Single test result."""

    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class MCPTestSuite:
    """Comprehensive test suite for all MCP features."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def _add_result(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        """Add a test result."""
        self.results.append(TestResult(name, passed, message, duration_ms))

    async def _timed_test(self, name: str, test_func) -> bool:
        """Run a test with timing."""
        start = datetime.now()
        try:
            result, message = await test_func()
            duration = (datetime.now() - start).total_seconds() * 1000
            self._add_result(name, result, message, duration)
            return result
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self._add_result(name, False, f"Exception: {str(e)[:100]}", duration)
            return False

    # =========================================================================
    # API Tests
    # =========================================================================

    async def test_config_loading(self) -> tuple:
        """Test configuration loading."""
        try:
            from .config import get_config

            config = get_config()
            if config.client_id and config.client_secret:
                return True, f"Config loaded: client_id={config.client_id[:8]}..."
            return False, "Missing credentials"
        except Exception as e:
            return False, str(e)

    async def test_gpu_type_mapping(self) -> tuple:
        """Test GPU type to instance type mapping."""
        from .client import get_instance_type_from_gpu_type_and_count

        tests = [
            ("B300", 1, "1B300.30V"),
            ("A6000", 1, "1A6000.10V"),
            ("L40S", 1, "1L40S.12V"),
        ]

        for gpu, count, expected in tests:
            result = get_instance_type_from_gpu_type_and_count(gpu, count)
            if result != expected:
                return False, f"{gpu}x{count}: got {result}, expected {expected}"

        return True, f"All {len(tests)} GPU mappings correct"

    async def test_list_instances(self) -> tuple:
        """Test listing instances via API."""
        try:
            from .client import get_client

            client = get_client()
            instances = await client.list_instances()
            return True, f"Found {len(instances)} instances"
        except Exception as e:
            return False, str(e)

    async def test_list_volumes(self) -> tuple:
        """Test listing volumes via API."""
        try:
            from .client import get_client

            client = get_client()
            volumes = await client.list_volumes()
            return True, f"Found {len(volumes)} volumes"
        except Exception as e:
            return False, str(e)

    async def test_list_ssh_keys(self) -> tuple:
        """Test listing SSH keys via API."""
        try:
            from .client import get_client

            client = get_client()
            keys = await client.list_ssh_keys()
            return True, f"Found {len(keys)} SSH keys"
        except Exception as e:
            return False, str(e)

    async def test_list_images(self) -> tuple:
        """Test listing OS images via API."""
        try:
            from .client import get_client

            client = get_client()
            images = await client.list_images()
            return True, f"Found {len(images)} images"
        except Exception as e:
            return False, str(e)

    async def test_check_availability(self) -> tuple:
        """Test checking GPU availability."""
        try:
            from .client import get_client

            client = get_client()
            result = await client.check_spot_availability("A6000", 1)
            return (
                True,
                f"A6000 available: {result.available} at {result.location or 'N/A'}",
            )
        except Exception as e:
            return False, str(e)

    # =========================================================================
    # Extended Tools Tests
    # =========================================================================

    async def test_cost_estimator(self) -> tuple:
        """Test cost estimation."""
        try:
            from .extended_tools import CostEstimator

            est = CostEstimator.estimate_cost("A6000", 1, 1, False)
            if est["total_cost"] == 0.49:
                return True, f"A6000 1hr = ${est['total_cost']:.2f}"
            return False, f"Expected $0.49, got ${est['total_cost']:.2f}"
        except Exception as e:
            return False, str(e)

    async def test_log_parser(self) -> tuple:
        """Test training log parser."""
        try:
            from .extended_tools import TrainingLogParser

            sample = "Step 100, Loss: 2.5\nStep 200, Loss: 2.3\nSaving checkpoint"
            parser = TrainingLogParser()
            metrics = parser.parse_logs(sample)

            if len(metrics["losses"]) >= 2:
                return True, f"Parsed {len(metrics['losses'])} loss values"
            return False, "Failed to parse losses"
        except Exception as e:
            return False, str(e)

    async def test_gdrive_module(self) -> tuple:
        """Test Google Drive module availability."""
        try:
            from .gdrive_tools import GDOWN_AVAILABLE, GoogleDriveManager

            if GDOWN_AVAILABLE:
                mgr = GoogleDriveManager()
                return True, f"gdown OK, dir: {mgr.downloads_dir}"
            return False, "gdown not installed"
        except Exception as e:
            return False, str(e)

    async def test_watchdog_module(self) -> tuple:
        """Test WatchDog module."""
        try:
            from .watchdog import get_reporter

            reporter = get_reporter()
            return True, f"WatchDog OK, reports: {reporter.reports_dir}"
        except Exception as e:
            return False, str(e)

    async def test_ssh_module(self) -> tuple:
        """Test SSH module availability."""
        try:
            from .ssh_tools import PARAMIKO_AVAILABLE

            if PARAMIKO_AVAILABLE:
                return True, "paramiko installed, SSH ready"
            return False, "paramiko not installed"
        except Exception as e:
            return False, str(e)

    # =========================================================================
    # SSH Tests (Requires Instance)
    # =========================================================================

    async def test_ssh_connection(self, instance_ip: str) -> tuple:
        """Test SSH connection to instance."""
        try:
            from .ssh_tools import get_ssh_manager

            manager = get_ssh_manager()
            loop = asyncio.get_event_loop()
            stdout, stderr, code = await loop.run_in_executor(
                None, lambda: manager.run_command(instance_ip, "echo 'MCP_SSH_OK'")
            )

            if "MCP_SSH_OK" in stdout:
                return True, "SSH connection successful"
            return False, f"Unexpected: {stdout[:50]}"
        except Exception as e:
            return False, str(e)

    async def test_remote_gpu_status(self, instance_ip: str) -> tuple:
        """Test getting GPU status via SSH."""
        try:
            from .ssh_tools import ssh_get_gpu_status

            result = await ssh_get_gpu_status(instance_ip)
            if "nvidia" in result.lower() or "gpu" in result.lower():
                return True, "nvidia-smi OK"
            return False, "No GPU info"
        except Exception as e:
            return False, str(e)

    async def test_remote_run_command(self, instance_ip: str) -> tuple:
        """Test running remote command."""
        try:
            from .ssh_tools import ssh_run_command

            result = await ssh_run_command(instance_ip, "hostname")
            if "Exit Code" in result:
                return True, "Command execution OK"
            return False, "Unexpected output"
        except Exception as e:
            return False, str(e)

    async def test_remote_file_ops(self, instance_ip: str) -> tuple:
        """Test remote file read/write."""
        try:
            from .ssh_tools import ssh_read_file, ssh_write_file

            test_content = f"MCP_TEST_{datetime.now().isoformat()}"
            await ssh_write_file(instance_ip, "/tmp/mcp_test.txt", test_content)
            result = await ssh_read_file(instance_ip, "/tmp/mcp_test.txt")

            if "MCP_TEST" in result:
                return True, "File read/write OK"
            return False, "Verification failed"
        except Exception as e:
            return False, str(e)

    async def test_health_check(self, instance_ip: str) -> tuple:
        """Test comprehensive health check."""
        try:
            from .extended_tools import HealthChecker

            checker = HealthChecker()
            checks = await checker.comprehensive_health_check(instance_ip)

            status = checks.get("overall", "unknown")
            return status in ["healthy", "warning"], f"Status: {status}"
        except Exception as e:
            return False, str(e)

    async def test_list_dir(self, instance_ip: str) -> tuple:
        """Test remote directory listing."""
        try:
            from .ssh_tools import ssh_list_dir

            result = await ssh_list_dir(instance_ip, "/workspace")
            if "total" in result.lower() or "Directory" in result:
                return True, "Directory listing OK"
            return False, "Unexpected output"
        except Exception as e:
            return False, str(e)

    # =========================================================================
    # Run Tests
    # =========================================================================

    async def run_api_tests(self) -> str:
        """Run all API tests (no instance required)."""
        self.results = []
        self.start_time = datetime.now()

        await self._timed_test("Config Loading", self.test_config_loading)
        await self._timed_test("GPU Type Mapping", self.test_gpu_type_mapping)
        await self._timed_test("List Instances", self.test_list_instances)
        await self._timed_test("List Volumes", self.test_list_volumes)
        await self._timed_test("List SSH Keys", self.test_list_ssh_keys)
        await self._timed_test("List Images", self.test_list_images)
        await self._timed_test("Check Availability", self.test_check_availability)
        await self._timed_test("Cost Estimator", self.test_cost_estimator)
        await self._timed_test("Log Parser", self.test_log_parser)
        await self._timed_test("GDrive Module", self.test_gdrive_module)
        await self._timed_test("WatchDog Module", self.test_watchdog_module)
        await self._timed_test("SSH Module", self.test_ssh_module)

        self.end_time = datetime.now()
        return self._format_results("API & Module Tests")

    async def run_ssh_tests(self, instance_ip: str) -> str:
        """Run SSH/remote tests (instance required)."""
        self.results = []
        self.start_time = datetime.now()

        await self._timed_test("SSH Connection", lambda: self.test_ssh_connection(instance_ip))
        await self._timed_test("GPU Status", lambda: self.test_remote_gpu_status(instance_ip))
        await self._timed_test("Run Command", lambda: self.test_remote_run_command(instance_ip))
        await self._timed_test("File Ops", lambda: self.test_remote_file_ops(instance_ip))
        await self._timed_test("List Directory", lambda: self.test_list_dir(instance_ip))
        await self._timed_test("Health Check", lambda: self.test_health_check(instance_ip))

        self.end_time = datetime.now()
        return self._format_results(f"SSH Tests ({instance_ip})")

    async def run_all_tests(self, instance_ip: Optional[str] = None) -> str:
        """Run all tests."""
        self.results = []
        self.start_time = datetime.now()

        # API Tests
        await self._timed_test("Config Loading", self.test_config_loading)
        await self._timed_test("GPU Type Mapping", self.test_gpu_type_mapping)
        await self._timed_test("List Instances", self.test_list_instances)
        await self._timed_test("List Volumes", self.test_list_volumes)
        await self._timed_test("List SSH Keys", self.test_list_ssh_keys)
        await self._timed_test("List Images", self.test_list_images)
        await self._timed_test("Check Availability", self.test_check_availability)

        # Extended Tools
        await self._timed_test("Cost Estimator", self.test_cost_estimator)
        await self._timed_test("Log Parser", self.test_log_parser)
        await self._timed_test("GDrive Module", self.test_gdrive_module)
        await self._timed_test("WatchDog Module", self.test_watchdog_module)
        await self._timed_test("SSH Module", self.test_ssh_module)

        # SSH Tests (if instance provided)
        if instance_ip:
            await self._timed_test("SSH Connection", lambda: self.test_ssh_connection(instance_ip))
            await self._timed_test("GPU Status", lambda: self.test_remote_gpu_status(instance_ip))
            await self._timed_test("Run Command", lambda: self.test_remote_run_command(instance_ip))
            await self._timed_test("File Ops", lambda: self.test_remote_file_ops(instance_ip))
            await self._timed_test("List Directory", lambda: self.test_list_dir(instance_ip))
            await self._timed_test("Health Check", lambda: self.test_health_check(instance_ip))

        self.end_time = datetime.now()
        title = "All Tests" + (f" ({instance_ip})" if instance_ip else " (No Instance)")
        return self._format_results(title)

    def _format_results(self, title: str) -> str:
        """Format test results as markdown."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        duration = (self.end_time - self.start_time).total_seconds()

        lines = [
            f"# 🧪 {title}",
            f"**Timestamp**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration**: {duration:.2f}s",
            "",
            "## Summary",
            f"- ✅ **Passed**: {passed}",
            f"- ❌ **Failed**: {failed}",
            f"- 📊 **Total**: {len(self.results)}",
            "",
            "## Results",
            "",
            "| Status | Test | Time (ms) | Details |",
            "|--------|------|-----------|---------|",
        ]

        for r in self.results:
            status = "✅" if r.passed else "❌"
            msg = r.message[:40] + "..." if len(r.message) > 40 else r.message
            lines.append(f"| {status} | {r.name} | {r.duration_ms:.0f} | {msg} |")

        # Failed details
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            lines.append("")
            lines.append("## ❌ Failed Tests Details")
            for r in failed_results:
                lines.append(f"- **{r.name}**: {r.message}")

        return "\n".join(lines)


# Global test suite instance
_test_suite: Optional[MCPTestSuite] = None


def get_test_suite() -> MCPTestSuite:
    """Get the global test suite."""
    global _test_suite
    if _test_suite is None:
        _test_suite = MCPTestSuite()
    return _test_suite


# Async wrappers for MCP tools


async def run_api_tests() -> str:
    """Run API and module tests (no instance required)."""
    suite = get_test_suite()
    return await suite.run_api_tests()


async def run_ssh_tests(instance_ip: str) -> str:
    """Run SSH/remote tests on an instance."""
    suite = get_test_suite()
    return await suite.run_ssh_tests(instance_ip)


async def run_all_tests(instance_ip: Optional[str] = None) -> str:
    """Run all tests."""
    suite = get_test_suite()
    return await suite.run_all_tests(instance_ip)
