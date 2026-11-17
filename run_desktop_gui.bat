@echo off
title Enterprise AML Compliance - Desktop GUI
echo ========================================================
echo Starting Enterprise AML AI SAR Narrative Generator (Desktop GUI)...
echo ========================================================
cd /d "%~dp0"
"C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\python.exe" app_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    pause
)

