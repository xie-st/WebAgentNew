# WebAgent 演示面板启动脚本（等价包装；app.py 亦可独立启动）。
# 用法：powershell -ExecutionPolicy Bypass -File code\harness\demo\start_demo.ps1 [-Port 8787]
param(
    [int]$Port = 8787
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "未找到 $python ，请先在仓库根目录创建 .venv 虚拟环境。"
}
& $python "code\harness\demo\app.py" --port $Port
