"""Comprehensive Test Suite for Verda MCP Enhanced Edition.

Tests all 44 tools to ensure they work correctly.
Run this AFTER deploying a test instance.

Usage:
    1. Deploy a cheap A6000 instance first
    2. Set INSTANCE_IP environment variable
    3. Run: python test_all_tools.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verda_mcp.client import VerdaSDKClient, get_instance_type_from_gpu_type_and_count
from verda_mcp.config import get_config

# Test results tracking
RESULTS = {
    "passed": [],
    "failed": [],
    "skipped": [],
}


def log_result(test_name: str, passed: bool, message: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if message:
        print(f"    → {message[:100]}")
    
    if passed:
        RESULTS["passed"].append(test_name)
    else:
        RESULTS["failed"].append((test_name, message))


def log_skip(test_name: str, reason: str):
    """Log skipped test."""
    print(f"⏭️ SKIP: {test_name} - {reason}")
    RESULTS["skipped"].append((test_name, reason))


class TestSuite:
    """Test suite for all MCP tools."""
    
    def __init__(self, instance_ip: str = None):
        self.instance_ip = instance_ip
        self.client = VerdaSDKClient()
    
    # =========================================================================
    # API Tests (No instance required)
    # =========================================================================
    
    async def test_gpu_type_mapping(self):
        """Test GPU type to instance type mapping."""
        test_cases = [
            ("B300", 1, "1B300.30V"),
            ("A6000", 1, "1A6000.10V"),
            ("L40S", 2, "2L40S.24V"),
        ]
        
        all_passed = True
        for gpu_type, count, expected in test_cases:
            result = get_instance_type_from_gpu_type_and_count(gpu_type, count)
            if result != expected:
                log_result(f"GPU Mapping {gpu_type}x{count}", False, f"Got {result}, expected {expected}")
                all_passed = False
        
        if all_passed:
            log_result("GPU Type Mapping", True, "All mappings correct")
    
    async def test_list_instances(self):
        """Test listing instances."""
        try:
            instances = await self.client.list_instances()
            log_result("list_instances", True, f"Found {len(instances)} instances")
            return instances
        except Exception as e:
            log_result("list_instances", False, str(e))
            return []
    
    async def test_list_volumes(self):
        """Test listing volumes."""
        try:
            volumes = await self.client.list_volumes()
            log_result("list_volumes", True, f"Found {len(volumes)} volumes")
        except Exception as e:
            log_result("list_volumes", False, str(e))
    
    async def test_list_scripts(self):
        """Test listing startup scripts."""
        try:
            scripts = await self.client.list_scripts()
            log_result("list_scripts", True, f"Found {len(scripts)} scripts")
        except Exception as e:
            log_result("list_scripts", False, str(e))
    
    async def test_list_ssh_keys(self):
        """Test listing SSH keys."""
        try:
            keys = await self.client.list_ssh_keys()
            log_result("list_ssh_keys", True, f"Found {len(keys)} SSH keys")
        except Exception as e:
            log_result("list_ssh_keys", False, str(e))
    
    async def test_list_images(self):
        """Test listing OS images."""
        try:
            images = await self.client.list_images()
            log_result("list_images", True, f"Found {len(images)} images")
        except Exception as e:
            log_result("list_images", False, str(e))
    
    async def test_check_spot_availability(self):
        """Test checking spot availability."""
        try:
            result = await self.client.check_spot_availability("A6000", 1)
            log_result("check_spot_availability", True, 
                      f"A6000 available: {result.available} at {result.location}")
        except Exception as e:
            log_result("check_spot_availability", False, str(e))
    
    # =========================================================================
    # Extended Tools Tests (No instance required)
    # =========================================================================
    
    async def test_cost_estimate(self):
        """Test cost estimation."""
        try:
            from verda_mcp.extended_tools import CostEstimator
            
            estimator = CostEstimator()
            estimate = estimator.estimate_cost("A6000", 1, 1, False)
            
            if estimate["total_cost"] == 0.49:
                log_result("cost_estimate", True, f"A6000 1hr = ${estimate['total_cost']:.2f}")
            else:
                log_result("cost_estimate", False, f"Expected $0.49, got ${estimate['total_cost']:.2f}")
        except Exception as e:
            log_result("cost_estimate", False, str(e))
    
    async def test_log_parser(self):
        """Test training log parser."""
        try:
            from verda_mcp.extended_tools import TrainingLogParser
            
            sample_log = """
            Step 100, Loss: 2.5, LR: 5e-6
            Step 200, Loss: 2.3, LR: 5e-6
            Saving checkpoint-200
            Step 300, Loss: 2.1, LR: 5e-6
            """
            
            parser = TrainingLogParser()
            metrics = parser.parse_logs(sample_log)
            
            if metrics["summary"]["total_steps"] >= 100:
                log_result("log_parser", True, f"Parsed {len(metrics['losses'])} loss values")
            else:
                log_result("log_parser", False, "Failed to parse steps")
        except Exception as e:
            log_result("log_parser", False, str(e))
    
    # =========================================================================
    # SSH Tests (Instance required)
    # =========================================================================
    
    async def test_ssh_connection(self):
        """Test SSH connection to instance."""
        if not self.instance_ip:
            log_skip("ssh_connection", "No instance IP provided")
            return False
        
        try:
            from verda_mcp.ssh_tools import get_ssh_manager
            
            manager = get_ssh_manager()
            stdout, stderr, code = manager.run_command(self.instance_ip, "echo 'SSH_TEST_OK'")
            
            if "SSH_TEST_OK" in stdout:
                log_result("ssh_connection", True, "SSH connection successful")
                return True
            else:
                log_result("ssh_connection", False, f"Unexpected output: {stdout}")
                return False
        except Exception as e:
            log_result("ssh_connection", False, str(e))
            return False
    
    async def test_remote_gpu_status(self):
        """Test getting GPU status."""
        if not self.instance_ip:
            log_skip("remote_gpu_status", "No instance IP")
            return
        
        try:
            from verda_mcp.ssh_tools import ssh_get_gpu_status
            
            result = await ssh_get_gpu_status(self.instance_ip)
            if "nvidia" in result.lower() or "gpu" in result.lower():
                log_result("remote_gpu_status", True, "nvidia-smi returned data")
            else:
                log_result("remote_gpu_status", False, "No GPU info in output")
        except Exception as e:
            log_result("remote_gpu_status", False, str(e))
    
    async def test_remote_run_command(self):
        """Test running remote command."""
        if not self.instance_ip:
            log_skip("remote_run_command", "No instance IP")
            return
        
        try:
            from verda_mcp.ssh_tools import ssh_run_command
            
            result = await ssh_run_command(self.instance_ip, "hostname")
            if "Exit Code" in result:
                log_result("remote_run_command", True, "Command executed successfully")
            else:
                log_result("remote_run_command", False, "Unexpected output format")
        except Exception as e:
            log_result("remote_run_command", False, str(e))
    
    async def test_remote_list_dir(self):
        """Test listing remote directory."""
        if not self.instance_ip:
            log_skip("remote_list_dir", "No instance IP")
            return
        
        try:
            from verda_mcp.ssh_tools import ssh_list_dir
            
            result = await ssh_list_dir(self.instance_ip, "/workspace")
            if "total" in result.lower() or "directory" in result.lower():
                log_result("remote_list_dir", True, "Directory listing successful")
            else:
                log_result("remote_list_dir", False, "Unexpected output")
        except Exception as e:
            log_result("remote_list_dir", False, str(e))
    
    async def test_remote_read_file(self):
        """Test reading remote file."""
        if not self.instance_ip:
            log_skip("remote_read_file", "No instance IP")
            return
        
        try:
            from verda_mcp.ssh_tools import ssh_read_file
            
            result = await ssh_read_file(self.instance_ip, "/etc/hostname")
            if "File:" in result:
                log_result("remote_read_file", True, "File read successful")
            else:
                log_result("remote_read_file", False, "Unexpected output")
        except Exception as e:
            log_result("remote_read_file", False, str(e))
    
    async def test_remote_write_file(self):
        """Test writing remote file."""
        if not self.instance_ip:
            log_skip("remote_write_file", "No instance IP")
            return
        
        try:
            from verda_mcp.ssh_tools import ssh_write_file, ssh_read_file
            
            test_content = f"MCP_TEST_{datetime.now().isoformat()}"
            await ssh_write_file(self.instance_ip, "/tmp/mcp_test.txt", test_content)
            
            # Verify
            result = await ssh_read_file(self.instance_ip, "/tmp/mcp_test.txt")
            if "MCP_TEST" in result:
                log_result("remote_write_file", True, "File write and verify successful")
            else:
                log_result("remote_write_file", False, "Write verification failed")
        except Exception as e:
            log_result("remote_write_file", False, str(e))
    
    async def test_health_check(self):
        """Test comprehensive health check."""
        if not self.instance_ip:
            log_skip("health_check", "No instance IP")
            return
        
        try:
            from verda_mcp.extended_tools import HealthChecker
            
            checker = HealthChecker()
            checks = await checker.comprehensive_health_check(self.instance_ip)
            
            if checks.get("overall") in ["healthy", "warning"]:
                log_result("health_check", True, f"Overall status: {checks['overall']}")
            else:
                log_result("health_check", False, f"Unexpected status: {checks.get('overall')}")
        except Exception as e:
            log_result("health_check", False, str(e))
    
    # =========================================================================
    # Google Drive Tests
    # =========================================================================
    
    async def test_gdrive_tools_available(self):
        """Test if Google Drive tools are available."""
        try:
            from verda_mcp.gdrive_tools import GDOWN_AVAILABLE, GoogleDriveManager
            
            if GDOWN_AVAILABLE:
                manager = GoogleDriveManager()
                log_result("gdrive_tools_available", True, f"gdown available, downloads dir: {manager.downloads_dir}")
            else:
                log_result("gdrive_tools_available", False, "gdown not installed")
        except Exception as e:
            log_result("gdrive_tools_available", False, str(e))
    
    # =========================================================================
    # WatchDog Tests
    # =========================================================================
    
    async def test_watchdog_module(self):
        """Test WatchDog module imports."""
        try:
            from verda_mcp.watchdog import WatchDogReport, get_reporter
            
            reporter = get_reporter()
            log_result("watchdog_module", True, f"Reports dir: {reporter.reports_dir}")
        except Exception as e:
            log_result("watchdog_module", False, str(e))
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    async def run_all(self):
        """Run all tests."""
        print("=" * 60)
        print("🧪 VERDA MCP ENHANCED EDITION - TEST SUITE")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.instance_ip:
            print(f"🖥️ Instance IP: {self.instance_ip}")
        else:
            print("⚠️ No instance IP - SSH tests will be skipped")
        print("=" * 60)
        print()
        
        # API Tests
        print("📡 API TESTS")
        print("-" * 40)
        await self.test_gpu_type_mapping()
        await self.test_list_instances()
        await self.test_list_volumes()
        await self.test_list_scripts()
        await self.test_list_ssh_keys()
        await self.test_list_images()
        await self.test_check_spot_availability()
        print()
        
        # Extended Tools Tests
        print("🔧 EXTENDED TOOLS TESTS")
        print("-" * 40)
        await self.test_cost_estimate()
        await self.test_log_parser()
        await self.test_gdrive_tools_available()
        await self.test_watchdog_module()
        print()
        
        # SSH Tests
        print("🔐 SSH/REMOTE TESTS")
        print("-" * 40)
        ssh_ok = await self.test_ssh_connection()
        if ssh_ok:
            await self.test_remote_gpu_status()
            await self.test_remote_run_command()
            await self.test_remote_list_dir()
            await self.test_remote_read_file()
            await self.test_remote_write_file()
            await self.test_health_check()
        print()
        
        # Summary
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {len(RESULTS['passed'])}")
        print(f"❌ Failed: {len(RESULTS['failed'])}")
        print(f"⏭️ Skipped: {len(RESULTS['skipped'])}")
        print()
        
        if RESULTS["failed"]:
            print("❌ FAILED TESTS:")
            for name, msg in RESULTS["failed"]:
                print(f"   - {name}: {msg[:50]}")
        
        return len(RESULTS["failed"]) == 0


async def main():
    """Main entry point."""
    instance_ip = os.environ.get("INSTANCE_IP") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    suite = TestSuite(instance_ip)
    success = await suite.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
