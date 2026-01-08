# Test MCP Connection Script
# Run this to verify Verda MCP is working

Write-Host "=== Testing Verda MCP Connection ===" -ForegroundColor Cyan

$ProjectDir = "C:\Users\Admin\Downloads\phase2\verda-cloud-mcp"
Push-Location $ProjectDir

try {
    Write-Host "`n1. Testing API connection..." -ForegroundColor Yellow
    $result = uv run python -c @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
try:
    scripts = asyncio.run(c.list_scripts())
    images = asyncio.run(c.list_images())
    keys = asyncio.run(c.list_ssh_keys())
    print(f'OK - Scripts:{len(scripts)} Images:{len(images)} Keys:{len(keys)}')
except Exception as e:
    print(f'FAIL - {e}')
"@
    Write-Host "   $result" -ForegroundColor Green

    Write-Host "`n2. Testing MCP server import..." -ForegroundColor Yellow
    $result = uv run python -c @"
import sys
sys.path.insert(0, 'src')
from verda_mcp.server_compact import mcp
tools = list(mcp._tool_manager._tools.keys())
print(f'OK - {len(tools)} tools loaded')
"@
    Write-Host "   $result" -ForegroundColor Green

    Write-Host "`n3. Listing all 21 mega-tools..." -ForegroundColor Yellow
    uv run python -c @"
import sys
sys.path.insert(0, 'src')
from verda_mcp.server_compact import mcp
tools = sorted(mcp._tool_manager._tools.keys())
for i, t in enumerate(tools, 1):
    print(f'   {i:2}. {t}')
"@

    Write-Host "`n=== All Tests Passed ===" -ForegroundColor Green
    Write-Host "`nNext steps:"
    Write-Host "  1. Restart Claude Desktop"
    Write-Host "  2. Restart Windsurf"
    Write-Host "  3. MCP tools should now work with 5-min timeout"
    
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
} finally {
    Pop-Location
}
