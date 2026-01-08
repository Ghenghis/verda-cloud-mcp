"""
Dashboard UI Tests - Playwright-based tests for the Verda Dashboard.

Tests all dashboard UI components:
- Sidebar navigation
- GPU metrics display
- Training status panel
- Terminal, SSH, Jupyter panels
- Settings panel
- WebSocket connectivity
- API endpoints
"""

import asyncio
import multiprocessing
import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if playwright is available
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Check if FastAPI/uvicorn available
try:
    import uvicorn

    from verda_mcp.api_server import create_app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def run_server(port: int):
    """Run the dashboard server in a subprocess."""
    if FASTAPI_AVAILABLE:
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


class DashboardUITester:
    """Playwright-based Dashboard UI tester."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        self.base_url = base_url
        self.browser = None
        self.context = None
        self.page = None
        self.results = []
        self.server_process = None

    async def setup(self):
        """Initialize Playwright and start server."""
        if not PLAYWRIGHT_AVAILABLE:
            return False

        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            return True
        except Exception as e:
            print(f"Playwright setup failed: {e}")
            return False

    async def teardown(self):
        """Clean up resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    def start_server(self, port: int = 8765):
        """Start the dashboard server in a subprocess."""
        if not FASTAPI_AVAILABLE:
            return False

        self.server_process = multiprocessing.Process(target=run_server, args=(port,))
        self.server_process.start()
        time.sleep(2)  # Wait for server to start
        return self.server_process.is_alive()

    def stop_server(self):
        """Stop the dashboard server."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.join(timeout=5)

    async def test_dashboard_loads(self) -> dict:
        """Test dashboard HTML loads correctly."""
        result = {"name": "dashboard_loads", "passed": False, "message": ""}

        try:
            response = await self.page.goto(self.base_url)
            await self.page.wait_for_load_state("networkidle")

            # Check page loaded
            if response and response.status == 200:
                title = await self.page.title()
                result["passed"] = "Verda" in title or "Dashboard" in title
                result["message"] = f"Dashboard loaded: {title}"
            else:
                result["message"] = f"HTTP {response.status if response else 'No response'}"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_sidebar_navigation(self) -> dict:
        """Test sidebar navigation items."""
        result = {"name": "sidebar_navigation", "passed": False, "message": ""}

        try:
            # Check sidebar exists
            sidebar = await self.page.query_selector("aside, .sidebar, nav")
            if not sidebar:
                result["message"] = "Sidebar not found"
                self.results.append(result)
                return result

            # Check for navigation items
            nav_items = await self.page.query_selector_all("[onclick*='showSection'], .nav-item")
            if len(nav_items) >= 5:
                result["passed"] = True
                result["message"] = f"Found {len(nav_items)} navigation items"
            else:
                result["message"] = f"Only found {len(nav_items)} nav items (expected >= 5)"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_gpu_metrics_section(self) -> dict:
        """Test GPU metrics section displays."""
        result = {"name": "gpu_metrics", "passed": False, "message": ""}

        try:
            # Look for GPU section
            gpu_section = await self.page.query_selector("#gpus, [id*='gpu'], .gpu-section")
            if gpu_section:
                # Check for GPU cards or metrics
                gpu_cards = await self.page.query_selector_all(".gpu-card, [class*='gpu']")
                result["passed"] = True
                result["message"] = f"GPU section found with {len(gpu_cards)} GPU cards"
            else:
                result["message"] = "GPU section not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_training_section(self) -> dict:
        """Test training status section."""
        result = {"name": "training_section", "passed": False, "message": ""}

        try:
            # Look for training section
            training = await self.page.query_selector("#training, [id*='training']")
            if training:
                result["passed"] = True
                result["message"] = "Training section found"
            else:
                result["message"] = "Training section not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_terminal_panel(self) -> dict:
        """Test Terminal panel exists and has input."""
        result = {"name": "terminal_panel", "passed": False, "message": ""}

        try:
            terminal = await self.page.query_selector("#terminal, [id*='terminal']")
            if terminal:
                # Check for input field
                input_field = await self.page.query_selector("#terminalInput, input[id*='terminal']")
                result["passed"] = input_field is not None
                result["message"] = "Terminal panel found" + (" with input" if input_field else " (no input)")
            else:
                result["message"] = "Terminal panel not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_ssh_panel(self) -> dict:
        """Test SSH panel exists."""
        result = {"name": "ssh_panel", "passed": False, "message": ""}

        try:
            ssh = await self.page.query_selector("#ssh, [id*='ssh']")
            if ssh:
                result["passed"] = True
                result["message"] = "SSH panel found"
            else:
                result["message"] = "SSH panel not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_jupyter_panel(self) -> dict:
        """Test Jupyter panel exists."""
        result = {"name": "jupyter_panel", "passed": False, "message": ""}

        try:
            jupyter = await self.page.query_selector("#jupyter, [id*='jupyter']")
            if jupyter:
                result["passed"] = True
                result["message"] = "Jupyter panel found"
            else:
                result["message"] = "Jupyter panel not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_javascript_functions(self) -> dict:
        """Test JavaScript functions are defined."""
        result = {"name": "javascript_functions", "passed": False, "message": ""}

        try:
            # Check for key JavaScript functions
            functions = [
                "showSection",
                "refreshSection",
                "fetchWithRetry",
                "showToast",
            ]

            defined = []
            for fn in functions:
                is_defined = await self.page.evaluate(f"typeof {fn} === 'function'")
                if is_defined:
                    defined.append(fn)

            result["passed"] = len(defined) >= 2
            result["message"] = f"JS functions defined: {', '.join(defined) if defined else 'none'}"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_api_health(self) -> dict:
        """Test API health endpoint."""
        result = {"name": "api_health", "passed": False, "message": ""}

        try:
            response = await self.page.goto(f"{self.base_url}/api/health")
            if response and response.status == 200:
                content = await self.page.content()
                result["passed"] = "healthy" in content.lower()
                result["message"] = "API health check passed"
            else:
                result["message"] = f"API health returned {response.status if response else 'no response'}"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_api_gpu(self) -> dict:
        """Test GPU API endpoint."""
        result = {"name": "api_gpu", "passed": False, "message": ""}

        try:
            response = await self.page.goto(f"{self.base_url}/api/gpu")
            if response and response.status == 200:
                content = await self.page.content()
                result["passed"] = "gpus" in content.lower() or "utilization" in content.lower()
                result["message"] = "GPU API endpoint works"
            else:
                result["message"] = f"GPU API returned {response.status if response else 'no response'}"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def run_all_tests(self) -> dict:
        """Run all dashboard UI tests."""
        setup_ok = await self.setup()

        if not setup_ok:
            return {
                "status": "skipped",
                "message": "Playwright not available",
                "results": [],
            }

        try:
            # Run UI tests
            await self.test_dashboard_loads()
            await self.test_sidebar_navigation()
            await self.test_gpu_metrics_section()
            await self.test_training_section()
            await self.test_terminal_panel()
            await self.test_ssh_panel()
            await self.test_jupyter_panel()
            await self.test_javascript_functions()

            # Run API tests
            await self.test_api_health()
            await self.test_api_gpu()
        finally:
            await self.teardown()

        passed = sum(1 for r in self.results if r["passed"])
        return {
            "status": "completed",
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "results": self.results,
        }


# Pytest tests
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_dashboard_ui():
    """Test dashboard UI components."""
    tester = DashboardUITester()
    server_started = tester.start_server(port=8766)

    if not server_started:
        pytest.skip("Could not start dashboard server")

    try:
        setup_ok = await tester.setup()
        if not setup_ok:
            pytest.skip("Playwright setup failed")

        tester.base_url = "http://127.0.0.1:8766"

        result = await tester.test_dashboard_loads()
        assert result["passed"], result["message"]
    finally:
        await tester.teardown()
        tester.stop_server()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_api_endpoints():
    """Test API endpoints."""
    tester = DashboardUITester()
    server_started = tester.start_server(port=8767)

    if not server_started:
        pytest.skip("Could not start dashboard server")

    try:
        setup_ok = await tester.setup()
        if not setup_ok:
            pytest.skip("Playwright setup failed")

        tester.base_url = "http://127.0.0.1:8767"

        result = await tester.test_api_health()
        assert result["passed"], result["message"]
    finally:
        await tester.teardown()
        tester.stop_server()


# CLI entry point
async def main():
    """Run dashboard UI tests from command line."""
    print("=" * 60)
    print("  DASHBOARD UI TEST SUITE")
    print("=" * 60)

    if not PLAYWRIGHT_AVAILABLE:
        print("\n❌ Playwright not installed.")
        print("   Install with: pip install playwright")
        print("   Then run: playwright install chromium")
        return 1

    if not FASTAPI_AVAILABLE:
        print("\n❌ FastAPI not installed.")
        print("   Install with: pip install fastapi uvicorn")
        return 1

    tester = DashboardUITester()

    # Start server
    print("\n🚀 Starting dashboard server...")
    server_started = tester.start_server(port=8768)
    if not server_started:
        print("❌ Failed to start server")
        return 1

    tester.base_url = "http://127.0.0.1:8768"

    try:
        report = await tester.run_all_tests()

        print(f"\nStatus: {report['status']}")
        print(f"Passed: {report.get('passed', 0)}/{report.get('total', 0)}")

        for result in report.get("results", []):
            icon = "✅" if result["passed"] else "❌"
            print(f"  {icon} {result['name']}: {result['message']}")

        return 0 if report.get("failed", 0) == 0 else 1
    finally:
        tester.stop_server()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
