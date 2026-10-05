@echo off
cd /d "%~dp0"
echo Starting the IPI replication UI at http://127.0.0.1:8765
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" app\server.py
