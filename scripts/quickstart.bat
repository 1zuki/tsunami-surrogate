@echo off
cd /d "%~dp0\.."
python scripts\run_smoke_pipeline.py %*
exit /b %errorlevel%
