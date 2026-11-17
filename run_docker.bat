@echo off
title Enterprise AML Compliance - Docker MLOps Launcher
echo ===============================================================================
echo   ENTERPRISE AML COMPLIANCE ^| DOCKER CONTAINER LAUNCHER
echo ===============================================================================
echo.
echo Checking for Docker daemon...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Docker daemon is not running or Docker is not installed in PATH.
    echo [*] Please launch Docker Desktop, or use 'run_localhost.bat' to host directly on localhost without Docker.
    pause
    exit /b 1
)

echo [+] Building and starting Docker container with Docker Compose...
docker compose up -d --build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo   [OK] CONTAINER RUNNING SUCCESSFULLY ON LOCALHOST!
    echo ===============================================================================
    echo   * Streamlit Compliance Portal: http://localhost:8501
    echo   * FastAPI MLOps Swagger UI:    http://localhost:8000/docs
    echo   * Healthcheck Endpoint:        http://localhost:8000/health
    echo   * Prometheus / MLOps Metrics:  http://localhost:8000/metrics
    echo ===============================================================================
    echo.
    echo Opening dashboard in your browser...
    start http://localhost:8501
) else (
    echo [!] Docker compose exited with an error. Falling back to direct 'docker run'...
    docker build -t aml-sar-generator:latest .
    docker run -d --name aml_sar_mlops_container -p 8501:8501 -p 8000:8000 -v "%cd%\data:/app/data" -v "%cd%\models:/app/models" aml-sar-generator:latest
    start http://localhost:8501
)
pause
