@echo off
title Enterprise AML Compliance - Desktop GUI
echo ========================================================
echo Starting Enterprise AML AI SAR Narrative Generator (Desktop GUI)...
echo ========================================================
cd /d "%~dp0"
set PYTHONUNBUFFERED=1

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python app_gui.py
) else (
    "C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe" app_gui.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Process exited with code %ERRORLEVEL%
    pause
)


