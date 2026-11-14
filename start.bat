@echo off
setlocal
cd /d "%~dp0"
set "SLAG_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "SLAG_PYTHON=.venv\Scripts\python.exe"
echo Open http://127.0.0.1:18765 after the service starts.
"%SLAG_PYTHON%" -B -X utf8 server.py %*
if errorlevel 1 pause
