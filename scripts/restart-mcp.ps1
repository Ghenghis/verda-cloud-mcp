# Restart MCP Servers Script
# Kills Python processes and restarts Claude Desktop

Write-Host "=== Restarting MCP Servers ===" -ForegroundColor Cyan

# Kill any stuck Python processes (MCP servers)
Write-Host "`n1. Stopping Python processes..." -ForegroundColor Yellow
$pythonProcs = Get-Process -Name python* -ErrorAction SilentlyContinue
if ($pythonProcs) {
    $pythonProcs | Stop-Process -Force
    Write-Host "   Killed $($pythonProcs.Count) Python processes" -ForegroundColor Green
} else {
    Write-Host "   No Python processes running" -ForegroundColor Gray
}

# Restart Claude Desktop
Write-Host "`n2. Restarting Claude Desktop..." -ForegroundColor Yellow
$claude = Get-Process -Name "Claude*" -ErrorAction SilentlyContinue
if ($claude) {
    $claude | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$claudePath = "C:\Users\Admin\AppData\Local\Programs\Claude\Claude.exe"
if (Test-Path $claudePath) {
    Start-Process $claudePath
    Write-Host "   Claude Desktop restarted" -ForegroundColor Green
} else {
    Write-Host "   Claude Desktop not found at expected path" -ForegroundColor Yellow
    Write-Host "   Please restart manually" -ForegroundColor Yellow
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Wait 10-15 seconds for MCP servers to initialize"
Write-Host "Then check Claude Desktop for '21/21 tools'"
