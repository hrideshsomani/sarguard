"""
Launcher for Streamlit Web Dashboard
Run: streamlit run app_web.py
Or:  python app_web.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    # If executed with python directly, launch streamlit via subprocess
    import subprocess
    target = os.path.join(PROJECT_ROOT, "src", "ui", "web_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", target]
    print(f"[+] Launching Streamlit web application: {' '.join(cmd)}")
    subprocess.run(cmd)
else:
    # If imported by streamlit run
    from src.ui.web_app import *
