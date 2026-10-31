param([int]$Port = 8787)
$ErrorActionPreference = 'Stop'
$deliveryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Push-Location -LiteralPath $deliveryRoot
try {
    & .\.venv\Scripts\python.exe code\harness\demo\app.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Demo exited with an error' }
} finally { Pop-Location }
