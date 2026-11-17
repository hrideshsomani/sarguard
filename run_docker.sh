#!/bin/bash
# Enterprise AML Compliance - Docker MLOps Launcher for Linux / macOS / WSL

set -e

echo "==============================================================================="
echo "  ENTERPRISE AML COMPLIANCE | DOCKER CONTAINER LAUNCHER"
echo "==============================================================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "[!] Docker is not installed or not in PATH."
    echo "[*] Please install Docker or run './run_localhost.sh' for direct localhost execution."
    exit 1
fi

echo "[+] Building and starting Docker container with Docker Compose..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

echo ""
echo "==============================================================================="
echo "  [OK] CONTAINER RUNNING SUCCESSFULLY ON LOCALHOST!"
echo "==============================================================================="
echo "  * Streamlit Compliance Portal: http://localhost:8501"
echo "  * FastAPI MLOps Swagger UI:    http://localhost:8000/docs"
echo "  * Healthcheck Endpoint:        http://localhost:8000/health"
echo "  * Prometheus / MLOps Metrics:  http://localhost:8000/metrics"
echo "==============================================================================="
echo ""

# Attempt to open browser
if which xdg-open > /dev/null; then
    xdg-open http://localhost:8501
elif which open > /dev/null; then
    open http://localhost:8501
fi
