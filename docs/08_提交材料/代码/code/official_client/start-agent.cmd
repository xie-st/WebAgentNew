@echo off
setlocal
if "%~1"=="" (
  echo Usage: start-agent.cmd "absolute-path-to-connection-file.json"
  exit /b 2
)
set "CONNECTION_FILE=%~f1"
cd /d "%~dp0"
"..\..\.venv\Scripts\python.exe" -m third_party_agent connect --connection-file "%CONNECTION_FILE%" --agent webagent.official_agent:OfficialAgent
exit /b %errorlevel%
