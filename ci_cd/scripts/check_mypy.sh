#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-.}"
export PATH="${ROOT_DIR}/backend/.venv/bin:${PATH}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting mypy checks for 'backend/src'..."
if mypy --config-file backend/pyproject.toml backend/src; then
    log "SUCCESS: All mypy checks passed for 'backend/src'!"
    exit 0
fi

log "FAILED: Some mypy checks failed for 'backend/src'!"
exit 1
