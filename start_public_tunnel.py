"""
Public Tunnel Launcher for Enterprise AML Compliance Suite
Uses Cloudflare Quick Tunnel to expose the local Streamlit application (port 8501)
to the global internet with a secure HTTPS URL without account or login.
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
URL_FILE = os.path.join(PROJECT_ROOT, "public_url.txt")
TUNNEL_LOG = os.path.join(PROJECT_ROOT, "logs", "tunnel.log")

os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

def main():
    if not os.path.exists(CLOUDFLARED_EXE):
        print(f"[!] Error: {CLOUDFLARED_EXE} not found.")
        sys.exit(1)

    cmd = [
        CLOUDFLARED_EXE, "tunnel",
        "--url", "http://127.0.0.1:8501",
        "--protocol", "http2",
        "--no-autoupdate"
    ]

    print("=" * 80)
    print("      ENTERPRISE AML COMPLIANCE - GLOBAL PUBLIC TUNNEL")
    print("=" * 80)
    print("[+] Launching Cloudflare Tunnel for port 8501 (Streamlit Web Portal)...")

    log_fp = open(TUNNEL_LOG, "w", encoding="utf-8", buffering=1)
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

    # Read output to extract the public URL
    start_time = time.time()
    while time.time() - start_time < 30:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
            continue

        log_fp.write(line)
        log_fp.flush()
        try:
            print(line.strip())
        except Exception:
            pass

        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "=" * 80)
        print("  [SUCCESS] GLOBAL PUBLIC HTTPS URL IS LIVE AND READY FOR SHARING!")
        print("=" * 80)
        print(f"  Public Web Portal:  {public_url}")
        print("=" * 80)
        print("  Anyone on the internet can now open this URL from any device")
        print("  (phone, tablet, laptop, desktop) to see:")
        print("  - All trained models (Random Forest, Isolation Forest, Rules)")
        print("  - All datasets (IBM AML, PaySim, FinCEN synthetic, Escrow, etc.)")
        print("  - The live 50 txns / 10s streaming simulation and SAR alerts")
        print("  - Download the PDF architectural dossiers")
        print("=" * 80 + "\n")

        with open(URL_FILE, "w", encoding="utf-8") as f:
            f.write(public_url.strip())

        # Keep running to maintain tunnel
        try:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    log_fp.write(line)
                    log_fp.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[!] Stopping Cloudflare Tunnel...")
            proc.terminate()
            proc.wait(timeout=5)
    else:
        print("[!] Could not parse Cloudflare public URL within 30s. Check logs/tunnel.log.")
        proc.terminate()
        sys.exit(1)

if __name__ == '__main__':
    main()
