@echo off
title Enterprise AML Compliance - Localhost MLOps Launcher
echo ===============================================================================
echo   ENTERPRISE AML COMPLIANCE ^| DIRECT LOCALHOST DUAL LAUNCHER
echo ===============================================================================
echo.
echo Starting FastAPI Model Serving Engine and Streamlit Compliance Dashboard...
echo.
echo   * Streamlit Compliance Portal: http://localhost:8501
echo   * FastAPI MLOps Swagger UI:    http://localhost:8000/docs
echo   * Healthcheck Endpoint:        http://localhost:8000/health
echo   * Prometheus / MLOps Metrics:  http://localhost:8000/metrics
echo.
echo ===============================================================================

start "AML FastAPI MLOps Service (Port 8000)" python app_api.py
timeout /t 3 /nobreak >nul
start "AML Streamlit Web Portal (Port 8501)" python -m streamlit run app_web.py --server.port 8501

echo [OK] Both services are running concurrently on localhost.
echo Press any key to stop all background python windows if needed.
pause
