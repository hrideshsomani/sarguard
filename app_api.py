"""
Launcher for FastAPI MLOps Model Serving Engine
Run: uvicorn app_api:app --host 0.0.0.0 --port 8000
Or:  python app_api.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app import app

if __name__ == '__main__':
    import uvicorn
    print("[+] Launching Enterprise AML MLOps API on http://localhost:8000 ...")
    print("[+] Swagger UI Documentation available at http://localhost:8000/docs")
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
