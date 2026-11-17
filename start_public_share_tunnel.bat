@echo off
title AML Compliance Portal - Global Public Sharing Tunnel
cd /d "%~dp0"
echo ===============================================================================
echo     ENTERPRISE AML COMPLIANCE - GLOBAL PUBLIC TUNNEL LAUNCHER
echo ===============================================================================
echo.
echo Launching secure Cloudflare Public Tunnel for Streamlit Web Portal (Port 8501)...
echo Anyone on the internet can access the models, datasets, and live simulation
echo using the generated HTTPS link.
echo.
echo Keep this window open to maintain the public URL.
echo ===============================================================================
python start_public_tunnel.py
pause
