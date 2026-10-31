$ErrorActionPreference = 'Stop'
$deliveryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Push-Location -LiteralPath $deliveryRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating environment failed' }
    }
    & uv pip install --python .venv/Scripts/python.exe -e code/vendor/effgen -e 'code/harness[real,dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed' }
    & .\.venv\Scripts\python.exe -m playwright install chromium
    if ($LASTEXITCODE -ne 0) { throw 'Installing Chromium failed' }
    & uv pip check --python .venv/Scripts/python.exe
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
} finally { Pop-Location }
