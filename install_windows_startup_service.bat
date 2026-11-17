@echo off
title Install AML Portal to Windows Startup
cd /d "%~dp0"
echo Setting up AML Portal to start automatically upon Windows user login...

set SCRIPT_PATH=%~dp0start_always_active.bat
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_VBS=%TEMP%\create_aml_shortcut.vbs

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%SHORTCUT_VBS%"
echo sLinkFile = "%STARTUP_DIR%\AML_Always_Active_Portal.lnk" >> "%SHORTCUT_VBS%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%SHORTCUT_VBS%"
echo oLink.TargetPath = "%SCRIPT_PATH%" >> "%SHORTCUT_VBS%"
echo oLink.WorkingDirectory = "%~dp0" >> "%SHORTCUT_VBS%"
echo oLink.Description = "Enterprise AML Compliance Portal (Always Active)" >> "%SHORTCUT_VBS%"
echo oLink.WindowStyle = 7 >> "%SHORTCUT_VBS%"
echo oLink.Save >> "%SHORTCUT_VBS%"

cscript //nologo "%SHORTCUT_VBS%"
del "%SHORTCUT_VBS%"

echo.
echo [SUCCESS] Windows Startup shortcut created in:
echo %STARTUP_DIR%\AML_Always_Active_Portal.lnk
echo.
echo The portal and API watchdog will now automatically boot and stay active whenever you turn on or restart your PC.
echo.
pause
