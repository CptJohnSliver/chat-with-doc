#!/usr/bin/env sh
set -eu

APP_MODULE="${APP_MODULE:-src.app.api.api:app}"
HOST="${HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

VENV_BIN="${VENV_BIN:-/app/backend/.venv/bin}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_BIN/python}"

# Prefer running via `python -m uvicorn` to avoid issues with console-script entrypoints
# when virtualenv paths differ between build/runtime stages.
if [ -x "$PYTHON_BIN" ]; then
  exec "$PYTHON_BIN" -m uvicorn "$APP_MODULE" \
    --host "$HOST" \
    --port "$PORT" \
    --proxy-headers \
    --server-header \
    --no-access-log \
    --log-level critical

fi

# Fallback: rely on python on PATH (in case venv layout differs)
exec python -m uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  --proxy-headers \
  --server-header \
  --no-access-log \
  --log-level critical
