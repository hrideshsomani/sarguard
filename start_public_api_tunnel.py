"""
Public Tunnel Launcher for FastAPI MLOps Engine (Port 8000)
Exposes the REST API, Swagger UI (/docs), /health, and /metrics publicly.
"""

import sys
import os
import re
import time
import subprocess

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_EXE = os.path.join(PROJECT_ROOT, "cloudflared.exe")
URL_FILE = os.path.join(PROJECT_ROOT, "public_api_url.txt")

def main():
    if not os.path.exists(CLOUDFLARED_EXE):
        print(f"[!] Error: {CLOUDFLARED_EXE} not found.")
        sys.exit(1)

    cmd = [
        CLOUDFLARED_EXE, "tunnel",
        "--url", "http://127.0.0.1:8000",
        "--no-autoupdate"
    ]

    print("=" * 80)
    print("      ENTERPRISE AML REST API - GLOBAL PUBLIC TUNNEL")
    print("=" * 80)
    print("[+] Launching Cloudflare Tunnel for port 8000 (FastAPI MLOps Engine)...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    start_time = time.time()
    while time.time() - start_time < 30:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
            continue

        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "=" * 80)
        print("  [SUCCESS] PUBLIC FASTAPI SWAGGER API IS LIVE!")
        print("=" * 80)
        print(f"  Interactive Swagger Docs: {public_url}/docs")
        print(f"  API Health Probe:         {public_url}/health")
        print(f"  MLOps Metrics Telemetry:  {public_url}/metrics")
        print("=" * 80 + "\n")

        with open(URL_FILE, "w", encoding="utf-8") as f:
            f.write(public_url.strip())

        try:
            while proc.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=5)
    else:
        print("[!] Could not parse Cloudflare public URL.")
        proc.terminate()
        sys.exit(1)

if __name__ == '__main__':
    main()
