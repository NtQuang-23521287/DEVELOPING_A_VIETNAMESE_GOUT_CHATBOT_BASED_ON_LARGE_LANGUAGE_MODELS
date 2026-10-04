"""Cross-platform launcher for FastAPI backend + Streamlit UI."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parent
BACKEND_PORT = int(os.environ.get("GOUT_BACKEND_PORT", "8000"))
STREAMLIT_PORT = int(os.environ.get("GOUT_STREAMLIT_PORT", "8501"))
BACKEND_URL = os.environ.get("GOUT_BACKEND_URL", f"http://127.0.0.1:{BACKEND_PORT}").rstrip("/")


def wait_for_backend(process: subprocess.Popen, seconds: int = 30) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Backend exited early with code {process.returncode}")
        try:
            if requests.get(f"{BACKEND_URL}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Backend did not become ready")


def main() -> int:
    env = os.environ.copy()
    env["GOUT_BACKEND_URL"] = BACKEND_URL
    backend = subprocess.Popen([sys.executable, str(ROOT / "run_backend.py")], cwd=ROOT, env=env)
    try:
        wait_for_backend(backend)
        print(f"Backend: {BACKEND_URL}")
        print(f"UI:      http://127.0.0.1:{STREAMLIT_PORT}")
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "streamlit_app.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(STREAMLIT_PORT),
            ],
            cwd=ROOT,
            env=env,
        )
    finally:
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.kill()


if __name__ == "__main__":
    raise SystemExit(main())
