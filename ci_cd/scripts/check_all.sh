#!/bin/bash
# Same checks as .pre-commit-config.yaml (ruff check/format, mypy, bandit),
# using backend/.venv so mypy sees real project dependencies.

set -euo pipefail

FINAL_RESULT=0
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-.}"
export PATH="${ROOT_DIR}/backend/.venv/bin:${PATH}"

CONFIG="backend/pyproject.toml"
TARGET="${1:-backend/src}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

safe_check() {
    local name="$1"
    shift
    log "******* Starting ${name} *******"
    if "$@"; then
        log "SUCCESS: ${name} passed"
        return 0
    fi
    log "ERROR: ${name} failed"
    FINAL_RESULT=1
    return 0
}

main() {
    if [[ ! -d "$TARGET" ]]; then
        log "ERROR: Directory '$TARGET' not found."
        exit 1
    fi

    log "Starting all checks for '$TARGET' (aligned with pre-commit)..."

    safe_check "ruff check" ruff check "$TARGET" --config "$CONFIG"
    safe_check "ruff format --check" ruff format --check "$TARGET" --config "$CONFIG"
    safe_check "mypy" mypy --config-file "$CONFIG" "$TARGET"
    safe_check "bandit" bandit -c "$CONFIG" -ll -r "$TARGET"

    if [[ $FINAL_RESULT -eq 0 ]]; then
        log "SUCCESS: All checks passed for '$TARGET'!"
    else
        log "FAILED: Some checks failed for '$TARGET'!"
    fi

    exit $FINAL_RESULT
}

main "$@"
