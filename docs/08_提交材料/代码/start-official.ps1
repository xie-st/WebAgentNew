param([Parameter(Mandatory = $true)][string]$ConnectionFile)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'docs\08_提交材料\迁移工具\start-official.ps1') -ConnectionFile $ConnectionFile
