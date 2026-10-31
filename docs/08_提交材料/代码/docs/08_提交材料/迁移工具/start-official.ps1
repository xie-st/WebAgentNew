param([Parameter(Mandatory = $true)][string]$ConnectionFile)
$ErrorActionPreference = 'Stop'
$deliveryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$connectionPath = (Resolve-Path -LiteralPath $ConnectionFile).Path
$deliveryPython = Join-Path $deliveryRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath (Join-Path $deliveryRoot 'code\official_client')
try {
    & $deliveryPython -m third_party_agent connect --connection-file $connectionPath --agent webagent.official_agent:OfficialAgent
    if ($LASTEXITCODE -ne 0) { throw 'Official connector exited with an error' }
} finally { Pop-Location }
