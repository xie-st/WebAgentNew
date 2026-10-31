$ErrorActionPreference = 'Stop'
$reportSourceDir = $PSScriptRoot
$reportBuildDir = Join-Path $reportSourceDir 'build'
$reportOutputDir = Join-Path (Split-Path -Parent $reportSourceDir) '成品'
New-Item -ItemType Directory -Path $reportBuildDir -Force | Out-Null
New-Item -ItemType Directory -Path $reportOutputDir -Force | Out-Null
Push-Location $reportSourceDir
try {
    for ($reportPass = 1; $reportPass -le 2; $reportPass++) {
        & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX compilation failed on pass $reportPass" }
    }
    Copy-Item -LiteralPath (Join-Path $reportBuildDir 'main.pdf') -Destination (Join-Path $reportOutputDir 'WebAgent_方案报告.pdf') -Force
} finally {
    Pop-Location
}
