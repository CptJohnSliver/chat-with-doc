#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)
cd "${REPO_ROOT}"

setup_project_paths() {
    local selected_dir="${PROJECT_DIR:-}"

    if [ -z "${selected_dir}" ]; then
        if [ -f "${REPO_ROOT}/pyproject.toml" ]; then
            selected_dir="."
        elif [ -f "${REPO_ROOT}/backend/pyproject.toml" ]; then
            selected_dir="backend"
        else
            echo "❌ Не найден pyproject.toml в корне репозитория или backend/" >&2
            exit 1
        fi
    fi

    selected_dir="${selected_dir%/}"
    if [ -z "${selected_dir}" ]; then
        selected_dir="."
    fi

    if [ "${selected_dir}" = "." ]; then
        PROJECT_PATH="${REPO_ROOT}"
        PYPROJECT_FILE="pyproject.toml"
        UV_LOCK_FILE="uv.lock"
    else
        PROJECT_PATH="${REPO_ROOT}/${selected_dir}"
        PYPROJECT_FILE="${selected_dir}/pyproject.toml"
        UV_LOCK_FILE="${selected_dir}/uv.lock"
    fi

    if [ ! -f "${REPO_ROOT}/${PYPROJECT_FILE}" ]; then
        echo "❌ Файл ${PYPROJECT_FILE} не найден" >&2
        exit 1
    fi

    PROJECT_DIR="${selected_dir}"
}

ensure_clean_worktree() {
    if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo "❌ Рабочая директория содержит незакоммиченные изменения" >&2
        echo "   Закоммитьте или уберите их перед запуском скрипта" >&2
        exit 1
    fi
}

fetch_upstream() {
    git fetch upstream --tags --prune
}

resolve_commit_ref() {
    local ref="${1:?ref is required}"

    if ! git rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
        echo "❌ ${ref} не является коммитом" >&2
        echo "   Для тегов используйте имя тега или refs/tags/<tag>, не upstream/tags/<tag>" >&2
        exit 1
    fi

    git rev-parse --verify "${ref}^{commit}"
}
