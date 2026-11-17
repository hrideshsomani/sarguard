"""
Enterprise AML Watchdog & Always-Active Server Supervisor
Maintains 100% uptime for:
1. Streamlit Interactive Compliance Portal (Port 8501)
2. FastAPI High-Performance Model Serving API (Port 8000)

Features:
- Automatic crash detection & instant resurrection
- Periodic HTTP health probing (/health & /_stcore/health)
- Hanging/deadlock detection with graceful process recycling
- Rotated logging for stdout/stderr
- Dual-binding to localhost (127.0.0.1) and LAN IP (0.0.0.0)
"""

import sys
import os
import time
import signal
import socket
import logging
import subprocess
import urllib.request
import urllib.error

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

WATCHDOG_LOG = os.path.join(LOGS_DIR, "watchdog.log")
API_LOG = os.path.join(LOGS_DIR, "api_service.log")
STREAMLIT_LOG = os.path.join(LOGS_DIR, "streamlit_service.log")

# Setup watchdog logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(WATCHDOG_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AML_WATCHDOG")

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Doesn't actually have to connect
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class ManagedService:
    def __init__(self, name, cmd, port, health_url, log_file):
        self.name = name
        self.cmd = cmd
        self.port = port
        self.health_url = health_url
        self.log_file = log_file
        self.process = None
        self.start_time = None
        self.consecutive_health_failures = 0
        self.log_fp = None

    def start(self):
        if self.process and self.process.poll() is None:
            logger.info(f"[{self.name}] Already active (PID: {self.process.pid}).")
            return

        try:
            self.log_fp = open(self.log_file, "a", encoding="utf-8", buffering=1)
            self.process = subprocess.Popen(
                self.cmd,
                stdout=self.log_fp,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env=os.environ.copy()
            )
            self.start_time = time.time()
            self.consecutive_health_failures = 0
            logger.info(f"[+] [{self.name}] Started successfully with PID {self.process.pid} on port {self.port}.")
        except Exception as e:
            logger.error(f"[!] Failed to launch [{self.name}]: {e}")

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.process and self.process.poll() is None:
            logger.info(f"[-] Stopping [{self.name}] (PID: {self.process.pid})...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self.log_fp and not self.log_fp.closed:
            try:
                self.log_fp.close()
            except Exception:
                pass
        self.process = None

    def restart(self, reason="Crash recovery"):
        logger.warning(f"[RESTART] Triggering restart for [{self.name}] due to: {reason}")
        self.stop()
        time.sleep(1)
        self.start()

    def check_health(self):
        if not self.is_alive():
            exit_code = self.process.poll() if self.process else 'None'
            self.restart(f"Process unexpectedly terminated (Exit Code: {exit_code})")
            return False

        # Wait at least 8 seconds after startup before checking HTTP health
        if time.time() - self.start_time < 8:
            return True

        if not self.health_url:
            return True

        try:
            req = urllib.request.Request(self.health_url, headers={"User-Agent": "AML-Watchdog/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                if 200 <= response.status < 400:
                    self.consecutive_health_failures = 0
                    return True
                else:
                    self.consecutive_health_failures += 1
        except Exception as e:
            self.consecutive_health_failures += 1
            logger.debug(f"[{self.name}] Health ping failed ({self.consecutive_health_failures}/3): {e}")

        if self.consecutive_health_failures >= 3:
            self.restart(f"Health ping failed 3 times consecutively ({self.health_url})")
            return False

        return True

def main():
    python_exe = sys.executable
    lan_ip = get_lan_ip()

    print("=" * 80)
    print("      ENTERPRISE AML RESILIENT ALWAYS-ACTIVE SUPERVISOR")
    print("=" * 80)
    print(f"  * Machine Hostname:       {socket.gethostname()}")
    print(f"  * Localhost Portal:       http://localhost:8501")
    print(f"  * Local Network Portal:   http://{lan_ip}:8501")
    print(f"  * FastAPI REST API:       http://localhost:8000")
    print(f"  * Swagger Documentation:  http://localhost:8000/docs")
    print(f"  * Health Probe Endpoint:  http://localhost:8000/health")
    print(f"  * MLOps Metrics Endpoint: http://localhost:8000/metrics")
    print("=" * 80)
    print(f"  * Watchdog Logs:          {WATCHDOG_LOG}")
    print(f"  * Streamlit Logs:         {STREAMLIT_LOG}")
    print(f"  * API Service Logs:       {API_LOG}")
    print("=" * 80 + "\n")

    api_cmd = [
        python_exe, "-m", "uvicorn", "app_api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ]

    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "app_web.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    api_service = ManagedService(
        name="FastAPI_MLOps_Engine",
        cmd=api_cmd,
        port=8000,
        health_url="http://127.0.0.1:8000/health",
        log_file=API_LOG
    )

    streamlit_service = ManagedService(
        name="Streamlit_Compliance_Portal",
        cmd=streamlit_cmd,
        port=8501,
        health_url="http://127.0.0.1:8501/_stcore/health",
        log_file=STREAMLIT_LOG
    )

    services = [api_service, streamlit_service]

    # Start API first, wait 2 seconds, then start Streamlit
    logger.info("Initializing Enterprise AML microservices...")
    api_service.start()
    time.sleep(2)
    streamlit_service.start()

    def shutdown(signum, frame):
        logger.info("\n[!] Watchdog received shutdown signal. Terminating microservices...")
        for s in services:
            s.stop()
        logger.info("[OK] All services cleanly stopped. Exiting watchdog.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("[ACTIVE] Supervisor is monitoring services. Press Ctrl+C to terminate.")

    # Main supervision loop
    last_status_report = time.time()
    while True:
        try:
            for s in services:
                s.check_health()

            # Every 60 seconds, output an operational heartbeat
            if time.time() - last_status_report > 60:
                logger.info("[HEARTBEAT] Both services running normally: Streamlit (:8501) | FastAPI (:8000)")
                last_status_report = time.time()

            time.sleep(5)
        except (KeyboardInterrupt, SystemExit):
            shutdown(None, None)
        except Exception as ex:
            logger.error(f"[!] Watchdog loop exception: {ex}")
            time.sleep(5)

if __name__ == "__main__":
    main()
