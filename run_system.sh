#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${GOUT_BACKEND_PORT:-8000}"
STREAMLIT_PORT="${GOUT_STREAMLIT_PORT:-8501}"
export GOUT_BACKEND_URL="${GOUT_BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python run_backend.py &
BACKEND_PID=$!

python - <<'PY'
import os, time, requests
url = os.environ.get('GOUT_BACKEND_URL', 'http://127.0.0.1:8000') + '/health'
for _ in range(60):
    try:
        if requests.get(url, timeout=1).ok:
            print('Backend ready:', url)
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    raise SystemExit('Backend did not become ready')
PY

streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port "$STREAMLIT_PORT"
