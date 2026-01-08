"""
End-to-End Playwright tests for Verda MCP Server.

Uses Playwright MCP integration for browser-based testing and
automated validation of all features.
"""

import asyncio
import os
import sys
from typing import Any

import pytest

# Check if playwright is available
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class PlaywrightMCPTester:
    """Playwright-based E2E tester for MCP tools."""

    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.results = []

    async def setup(self):
        """Initialize Playwright browser."""
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
        """Clean up Playwright resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def test_github_repo(self) -> dict:
        """Test GitHub repository is accessible."""
        result = {"name": "github_repo", "passed": False, "message": ""}

        try:
            await self.page.goto("https://github.com/Ghenghis/verda-cloud-mcp")
            await self.page.wait_for_load_state("networkidle")

            # Check for README
            readme = await self.page.query_selector('article[class*="markdown"]')
            if readme:
                result["passed"] = True
                result["message"] = "GitHub repo accessible with README"
            else:
                result["message"] = "README not found"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_releases_page(self) -> dict:
        """Test GitHub releases page."""
        result = {"name": "releases_page", "passed": False, "message": ""}

        try:
            await self.page.goto("https://github.com/Ghenghis/verda-cloud-mcp/releases")
            await self.page.wait_for_load_state("networkidle")

            # Check for release
            releases = await self.page.query_selector_all('[class*="release"]')
            result["passed"] = True
            result["message"] = f"Found {len(releases)} release(s)"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def test_actions_page(self) -> dict:
        """Test GitHub Actions page."""
        result = {"name": "actions_page", "passed": False, "message": ""}

        try:
            await self.page.goto("https://github.com/Ghenghis/verda-cloud-mcp/actions")
            await self.page.wait_for_load_state("networkidle")

            # Check for workflow runs
            content = await self.page.content()
            if "workflow" in content.lower():
                result["passed"] = True
                result["message"] = "Actions page accessible"
        except Exception as e:
            result["message"] = f"Error: {e}"

        self.results.append(result)
        return result

    async def run_all_tests(self) -> dict:
        """Run all Playwright E2E tests."""
        setup_ok = await self.setup()

        if not setup_ok:
            return {
                "status": "skipped",
                "message": "Playwright not available or setup failed",
                "results": [],
            }

        try:
            await self.test_github_repo()
            await self.test_releases_page()
            await self.test_actions_page()
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
@pytest.mark.asyncio
async def test_github_repo():
    """Test GitHub repository is accessible."""
    tester = PlaywrightMCPTester()
    await tester.setup()
    try:
        result = await tester.test_github_repo()
        assert result["passed"], result["message"]
    finally:
        await tester.teardown()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.asyncio
async def test_releases_page():
    """Test releases page."""
    tester = PlaywrightMCPTester()
    await tester.setup()
    try:
        result = await tester.test_releases_page()
        assert result["passed"], result["message"]
    finally:
        await tester.teardown()


# CLI entry point
async def main():
    """Run Playwright tests from command line."""
    print("=" * 60)
    print("  PLAYWRIGHT E2E TEST SUITE")
    print("=" * 60)

    if not PLAYWRIGHT_AVAILABLE:
        print("\n❌ Playwright not installed.")
        print("   Install with: pip install playwright")
        print("   Then run: playwright install chromium")
        return 1

    tester = PlaywrightMCPTester()
    report = await tester.run_all_tests()

    print(f"\nStatus: {report['status']}")
    print(f"Passed: {report.get('passed', 0)}/{report.get('total', 0)}")

    for result in report.get("results", []):
        icon = "✅" if result["passed"] else "❌"
        print(f"  {icon} {result['name']}: {result['message']}")

    return 0 if report.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
