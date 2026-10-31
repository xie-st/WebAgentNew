param([int]$Port = 8787)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'docs\08_提交材料\迁移工具\start-demo.ps1') -Port $Port
