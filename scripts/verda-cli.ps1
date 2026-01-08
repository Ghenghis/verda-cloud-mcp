# Verda Cloud CLI Helper Script
# Usage: .\verda-cli.ps1 <command> [args]
# Commands: list-gpus, list-instances, list-volumes, list-scripts, list-images, deploy, status

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1,
    
    [Parameter(Position=2)]
    [string]$Arg2
)

$ProjectDir = "C:\Users\Admin\Downloads\phase2\verda-cloud-mcp"

function Invoke-Verda {
    param([string]$PythonCode)
    Push-Location $ProjectDir
    try {
        uv run python -c $PythonCode
    } finally {
        Pop-Location
    }
}

switch ($Command) {
    "list-gpus" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
gpus = asyncio.run(c.list_gpus())
print('Available GPUs:')
for g in gpus:
    print(f'  {g.gpu_type}: {g.vram}GB VRAM, \${g.price_per_hour}/hr')
"@
    }
    
    "list-instances" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
instances = asyncio.run(c.list_instances())
if not instances:
    print('No instances found')
else:
    for i in instances:
        print(f'{i.id}: {i.status} - {i.gpu_type}x{i.gpu_count}')
"@
    }
    
    "list-volumes" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
volumes = asyncio.run(c.list_volumes())
if not volumes:
    print('No volumes found')
else:
    for v in volumes:
        print(f'{v.id}: {v.name} - {v.size}GB')
"@
    }
    
    "list-scripts" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
scripts = asyncio.run(c.list_scripts())
print(f'Scripts ({len(scripts)}):')
for s in scripts:
    print(f'  {s.id}: {s.name}')
"@
    }
    
    "list-images" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
images = asyncio.run(c.list_images())
print(f'Images ({len(images)}):')
for img in images:
    print(f'  {img.name}')
"@
    }
    
    "status" {
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
print('=== VERDA API STATUS ===')
print(f'Scripts: {len(asyncio.run(c.list_scripts()))}')
print(f'Images: {len(asyncio.run(c.list_images()))}')
print(f'SSH Keys: {len(asyncio.run(c.list_ssh_keys()))}')
print(f'Instances: {len(asyncio.run(c.list_instances()))}')
print(f'Volumes: {len(asyncio.run(c.list_volumes()))}')
print('========================')
print('API Connection: OK')
"@
    }
    
    "deploy" {
        if (-not $Arg1) { $Arg1 = "V100" }
        if (-not $Arg2) { $Arg2 = "1" }
        Write-Host "Deploying $Arg1 x $Arg2..."
        Invoke-Verda @"
import asyncio
from verda_mcp.client import get_client
c = get_client()
result = asyncio.run(c.deploy_instance(gpu_type='$Arg1', gpu_count=$Arg2))
print(f'Deployed: {result}')
"@
    }
    
    "help" {
        Write-Host @"
Verda Cloud CLI Helper
======================
Usage: .\verda-cli.ps1 <command> [args]

Commands:
  list-gpus      - List available GPU types and prices
  list-instances - List running instances
  list-volumes   - List storage volumes  
  list-scripts   - List startup scripts
  list-images    - List OS images
  status         - Show API connection status
  deploy <gpu> <count> - Deploy instance (e.g., deploy V100 1)
  help           - Show this help

Examples:
  .\verda-cli.ps1 status
  .\verda-cli.ps1 list-gpus
  .\verda-cli.ps1 deploy V100 1
"@
    }
    
    default {
        Write-Host "Unknown command: $Command"
        Write-Host "Run '.\verda-cli.ps1 help' for usage"
    }
}
