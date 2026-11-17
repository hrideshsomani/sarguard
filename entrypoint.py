"""
Enterprise AML MLOps Container Entrypoint
Concurrently supervises and runs:
1. FastAPI Model Serving & Healthcheck Engine (Port 8000)
2. Streamlit Interactive Compliance Dashboard (Port 8501)
"""

import sys
import os
import time
import signal
import subprocess

def main():
    print("=" * 80)
    print("  STARTING ENTERPRISE AML SAR MLOPS CONTAINER SERVICES")
    print("=" * 80)
    print("  [1] FastAPI MLOps Service:       http://0.0.0.0:8000")
    print("      -> Swagger UI Documentation: http://0.0.0.0:8000/docs")
    print("      -> Health Check Probe:       http://0.0.0.0:8000/health")
    print("      -> MLOps Metrics Endpoint:   http://0.0.0.0:8000/metrics")
    print("  [2] Streamlit Compliance Portal: http://0.0.0.0:8501")
    print("=" * 80 + "\n")

    python_exe = sys.executable

    # 1. Launch FastAPI Uvicorn Server
    api_cmd = [
        python_exe, "-m", "uvicorn", "app_api:app",
        "--host", "0.0.0.0",
        "--port", os.environ.get("API_PORT", "8000")
    ]
    print(f"[+] Launching REST API: {' '.join(api_cmd)}")
    api_proc = subprocess.Popen(api_cmd)

    # Brief delay to allow API to initialize model singleton
    time.sleep(2)

    # 2. Launch Streamlit Dashboard
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "app_web.py",
        "--server.port", os.environ.get("STREAMLIT_PORT", "8501"),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    print(f"[+] Launching Streamlit Web Portal: {' '.join(streamlit_cmd)}")
    st_proc = subprocess.Popen(streamlit_cmd)

    def shutdown_handler(signum, frame):
        print("\n[!] Received shutdown signal. Gracefully stopping microservices...")
        try:
            api_proc.terminate()
            st_proc.terminate()
            api_proc.wait(timeout=5)
            st_proc.wait(timeout=5)
        except Exception:
            api_proc.kill()
            st_proc.kill()
        print("[OK] Container services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Monitor child processes
    while True:
        if api_proc.poll() is not None:
            print("[!] FastAPI process terminated unexpectedly.")
            st_proc.terminate()
            sys.exit(api_proc.returncode or 1)
        if st_proc.poll() is not None:
            print("[!] Streamlit process terminated unexpectedly.")
            api_proc.terminate()
            sys.exit(st_proc.returncode or 1)
        time.sleep(2)

if __name__ == '__main__':
    main()
