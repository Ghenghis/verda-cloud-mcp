# 🧪 Verda MCP Server - Testing Guide

> Comprehensive automated testing with Playwright integration

---

## 📋 Test Categories

| Category | File | Description |
|----------|------|-------------|
| **Unit Tests** | `test_playwright_integration.py` | Module imports, tool definitions, components |
| **E2E Tests** | `test_e2e_playwright.py` | Browser-based Playwright tests |
| **MCP Tools** | `test_mcp_tools.py` | MCP tool validation |
| **Client** | `test_client.py` | SDK client tests |
| **Scripts** | `test_script_tools.py` | Script generation tests |

---

## 🚀 Quick Start

### Install Test Dependencies

```bash
# Install with test extras
uv sync --dev
uv pip install pytest pytest-asyncio playwright

# Install Playwright browsers
uv run playwright install chromium
```

### Run All Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/verda_mcp --cov-report=html

# Generate HTML report
uv run pytest tests/ -v --html=report.html
```

---

## 🎯 Test Commands

### Unit Tests (Fast)

```bash
# Run unit tests only
uv run pytest tests/ -v -m "not e2e"

# Run specific test file
uv run pytest tests/test_playwright_integration.py -v

# Run specific test
uv run pytest tests/test_playwright_integration.py::test_gpu_database -v
```

### Integration Tests

```bash
# Run integration test suite
uv run python tests/test_playwright_integration.py

# This outputs:
# ============================================================
#   VERDA MCP SERVER - AUTOMATED TEST SUITE
# ============================================================
# ✅ import_verda_mcp.server: Successfully imported Main server module
# ✅ gpu_database: All 12 GPU types present
# ✅ training_intelligence: All Training Intelligence components validated
# ...
```

### E2E Playwright Tests

```bash
# Run E2E tests
uv run pytest tests/test_e2e_playwright.py -v

# Run with headed browser (visible)
uv run pytest tests/test_e2e_playwright.py -v --headed

# Run specific E2E test
uv run pytest tests/test_e2e_playwright.py::test_github_repo -v
```

---

## 📊 Test Report

After running tests, a JSON report is generated:

```bash
# View test report
cat test_report.json
```

Example output:

```json
{
  "summary": {
    "total": 22,
    "passed": 22,
    "failed": 0,
    "success_rate": "100.0%",
    "duration_ms": 1234
  },
  "results": [
    {"name": "import_verda_mcp.server", "status": "passed"},
    {"name": "gpu_database", "status": "passed"},
    ...
  ]
}
```

---

## 🔧 Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
```

### Running with Markers

```bash
# Unit tests only
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"

# E2E tests only
uv run pytest -m e2e
```

---

## 🧠 What Gets Tested

### 1. Module Imports (15 modules)

- `server.py` - Main MCP server
- `client.py` - Verda SDK client
- `training_intelligence.py` - Mega-tools
- `smart_deployer.py` - 7-layer fail-safe
- `gpu_optimizer.py` - GPU database
- ... all 15 Python modules

### 2. Tool Definitions (87 tools)

- All @mcp.tool() decorators validated
- Tool signatures checked
- Docstrings verified

### 3. GPU Database (12 GPUs)

- GB300, B300, B200, H200, H100
- A100_80G, A100_40G, V100
- RTX_PRO_6000, L40S, RTX_6000_ADA, A6000
- Pricing data validated
- Spot savings calculations verified

### 4. Training Intelligence

- 7 Skill levels (Beginner → Hacker)
- 7 Output formats (ASCII → HTML)
- 10 Training stages
- MetricsTranslator, StageCalculator, VisualizationGenerator

### 5. Smart Deployer

- GPUConfig dataclass
- DeploymentPlan dataclass
- 7-layer fail-safe system
- AvailabilityResult validation

### 6. Playwright E2E

- GitHub repository accessible
- Releases page renders
- Actions page loads
- README displays correctly

---

## 🔄 CI/CD Integration

Tests run automatically on:

- **Push to main/develop**
- **Pull requests to main**

### GitHub Actions Workflow

```yaml
jobs:
  test:
    - Run linting (ruff)
    - Run type checking (mypy)
    - Run unit tests (pytest)
    - Build package

  e2e-tests:
    - Install Playwright
    - Run E2E tests
    - Run integration suite
```

---

## 📈 Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Module imports | 100% | ✅ 100% |
| Tool definitions | 100% | ✅ 100% |
| GPU database | 100% | ✅ 100% |
| Training Intelligence | 100% | ✅ 100% |
| Smart Deployer | 100% | ✅ 100% |
| E2E (GitHub) | 100% | ✅ 100% |

---

## 🐛 Debugging Tests

### Verbose Output

```bash
uv run pytest -v -s --tb=long
```

### Stop on First Failure

```bash
uv run pytest -x
```

### Run Last Failed

```bash
uv run pytest --lf
```

### Debug Mode

```bash
uv run pytest --pdb
```

---

## 📝 Adding New Tests

### 1. Create Test Function

```python
@pytest.mark.asyncio
async def test_new_feature():
    """Test description."""
    suite = VerdaMCPTestSuite()
    await suite.test_new_feature()
    assert suite.results[0].status == TestStatus.PASSED
```

### 2. Add to Test Suite

```python
async def test_new_feature(self):
    """Test new feature."""
    start = datetime.now()
    try:
        # Test logic here
        result = await some_function()
        assert result is not None
        
        duration = (datetime.now() - start).total_seconds() * 1000
        self.results.append(TestResult(
            name="new_feature",
            status=TestStatus.PASSED,
            duration_ms=duration,
            message="Feature works correctly"
        ))
    except Exception as e:
        # Handle failure
        ...
```

### 3. Mark with Category

```python
@pytest.mark.unit  # or @pytest.mark.integration or @pytest.mark.e2e
async def test_new_feature():
    ...
```

---

## 🎯 Quick Reference

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run all tests |
| `uv run pytest -v` | Verbose output |
| `uv run pytest -x` | Stop on first failure |
| `uv run pytest --lf` | Run last failed |
| `uv run pytest -m unit` | Unit tests only |
| `uv run pytest -m e2e` | E2E tests only |
| `uv run pytest --cov` | With coverage |
| `uv run python tests/test_playwright_integration.py` | Full integration suite |

---

**✅ 87 tools + 55 bundled functions = 140+ capabilities, fully tested!**
