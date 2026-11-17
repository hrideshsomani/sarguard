@echo off
title AML Compliance Portal - Always Active Watchdog
cd /d "%~dp0"
echo ===============================================================================
echo     ENTERPRISE AML RESILIENT ALWAYS-ACTIVE SUPERVISOR LAUNCHER
echo ===============================================================================
echo.
echo Launching 24/7 Watchdog Supervisor...
echo Streamlit Portal:  http://localhost:8501
echo FastAPI Engine:    http://localhost:8000
echo.
echo If any service crashes or hangs, the watchdog will automatically resurrect it.
echo Keep this window open or minimized to maintain continuous uptime.
echo ===============================================================================
python serve_always_active.py
pause
