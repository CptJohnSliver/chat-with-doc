#!/usr/bin/env sh
set -eu

export PATH=/app/.venv/bin:$PATH
export PYTHONPATH="/app/backend:${PYTHONPATH:-/app/src}"

exec python -m uvicorn ${APP_MODULE:-src.app.api.api:app} \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --no-access-log \
    --no-server-header \
    --workers "${WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}"
