@echo off
cd /d "%~dp0"
python -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
pause
