# Contributing to Verda MCP

Thank you for your interest in contributing! This document provides guidelines and information about contributing to the Verda Cloud MCP Server.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Adding New Tools](#adding-new-tools)

---

## 📜 Code of Conduct

Be respectful, inclusive, and constructive. We're all here to make GPU training easier!

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git
- Verda Cloud account (for testing)

### Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/verda-cloud-mcp.git
cd verda-cloud-mcp
```

---

## 🛠️ Development Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### 2. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

### 3. Configure Credentials

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your Verda API credentials
```

### 4. Run the Server

```bash
uv run python -m verda_mcp
```

---

## ✏️ Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add multi-GPU comparison tool
fix: correct spot pricing calculation
docs: update GPU catalog in README
refactor: simplify SSH manager
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_client.py

# Run with coverage
uv run pytest --cov=src/verda_mcp
```

### Linting and Formatting

```bash
# Check linting
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/

# Type checking
uv run mypy src/
```

---

## 📤 Submitting Changes

### 1. Create a Pull Request

1. Push your branch to your fork
2. Open a PR against `main`
3. Fill out the PR template
4. Wait for review

### 2. PR Requirements

- [ ] All tests pass
- [ ] Code is formatted and linted
- [ ] CHANGELOG.md updated (for features/fixes)
- [ ] Documentation updated if needed
- [ ] No merge conflicts

### 3. Review Process

- Maintainers will review your PR
- Address any feedback
- Once approved, it will be merged

---

## 🎨 Style Guidelines

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all public functions

### Example Function

```python
async def my_new_tool(
    param1: str,
    param2: int = 10,
) -> str:
    """Short description of what the tool does.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2 (default: 10).

    Returns:
        Description of return value.
    """
    # Implementation
    return result
```

---

## 🔧 Adding New Tools

### 1. Choose the Right Module

| Tool Type | Module |
|-----------|--------|
| Instance management | `server.py` |
| SSH operations | `ssh_tools.py` |
| Spot management | `spot_manager.py` |
| Training helpers | `training_tools.py` |
| GPU optimization | `gpu_optimizer.py` |
| Live API data | `live_data.py` |
| Advanced/Beta | `advanced_tools.py` |

### 2. Create the Tool Function

```python
# In the appropriate module (e.g., training_tools.py)

async def my_new_feature(param: str) -> str:
    """Tool description for Claude."""
    # Implementation
    return formatted_result
```

### 3. Register in server.py

```python
# Import
from .my_module import my_new_feature

# Add MCP tool wrapper
@mcp.tool()
async def my_tool_name(param: str) -> str:
    """Tool description shown to Claude.

    Args:
        param: Parameter description.

    Returns:
        What the tool returns.
    """
    return await my_new_feature(param)
```

### 4. Update Tool Count

In `server.py` main function, update the tool count:

```python
my_module_tools = X if MY_MODULE_AVAILABLE else 0
total = base_tools + ... + my_module_tools
```

### 5. Update Documentation

- Add to README.md tool inventory
- Add to CHANGELOG.md under [Unreleased]

---

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

---

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!
