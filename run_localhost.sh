#!/bin/bash
# Enterprise AML Compliance - Direct Localhost Dual Launcher for Linux / macOS / WSL

set -e

echo "==============================================================================="
echo "  ENTERPRISE AML COMPLIANCE | DIRECT LOCALHOST DUAL LAUNCHER"
echo "==============================================================================="
echo ""
echo "Starting FastAPI Model Serving Engine and Streamlit Compliance Dashboard..."
echo ""
echo "  * Streamlit Compliance Portal: http://localhost:8501"
echo "  * FastAPI MLOps Swagger UI:    http://localhost:8000/docs"
echo "  * Healthcheck Endpoint:        http://localhost:8000/health"
echo "  * Prometheus / MLOps Metrics:  http://localhost:8000/metrics"
echo ""
echo "==============================================================================="

python app_api.py &
API_PID=$!
sleep 3

python -m streamlit run app_web.py --server.port 8501 &
ST_PID=$!

trap "kill $API_PID $ST_PID; exit 0" SIGINT SIGTERM
echo "[OK] Both services running. Press Ctrl+C to terminate."
wait
