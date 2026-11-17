"""
Launcher for Python Desktop GUI (Tkinter + ttk + Matplotlib)
Run: python app_gui.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui.desktop_app import launch_desktop_gui

if __name__ == '__main__':
    print("[+] Launching Enterprise AML Compliance Desktop GUI...", flush=True)
    launch_desktop_gui()
