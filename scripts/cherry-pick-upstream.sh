#!/bin/bash
set -euo pipefail

# ============================================
# cherry-pick-upstream.sh
# Подтягивание коммита из GitLab в текущую ветку
#
# Использование:
#   ./scripts/cherry-pick-upstream.sh <SHA> [<SHA2> ...]
#   PROJECT_DIR=backend ./scripts/cherry-pick-upstream.sh <SHA> [<SHA2> ...]
#
# Пример:
#   ./scripts/cherry-pick-upstream.sh abc1234 def5678
# ============================================

source "$(dirname "$0")/lib-project.sh"

if [ $# -eq 0 ]; then
    echo "❌ Укажите SHA коммита(ов) из GitLab"
    echo "   Использование: ./scripts/cherry-pick-upstream.sh <SHA> [<SHA2> ...]"
    exit 1
fi

BRANCH_NAME=$(git branch --show-current)
setup_project_paths
ensure_clean_worktree

echo "🍒 Cherry-pick из GitLab"
echo "   Ветка: ${BRANCH_NAME}"
echo "   Коммиты: $*"
echo "   Проект: ${PROJECT_DIR}"
echo ""

# Обновляем upstream
fetch_upstream

# Cherry-pick всех указанных коммитов
for SHA in "$@"; do
    COMMIT=$(resolve_commit_ref "${SHA}")
    echo "📌 Cherry-pick ${SHA}..."
    if ! git cherry-pick "${COMMIT}"; then
        echo ""
        echo "❌ Конфликт при cherry-pick ${SHA}"
        echo "📌 Решите конфликт вручную, затем:"
        echo "   git add ."
        echo "   git cherry-pick --continue"
        exit 1
    fi
done

# Если есть изменения в pyproject.toml (из-за новых зависимостей), пересобираем lock
if git diff "HEAD~$#" HEAD --name-only | grep -q "^${PYPROJECT_FILE}$"; then
    echo ""
    echo "🔄 Обнаружены изменения в ${PYPROJECT_FILE}, пересобираем ${UV_LOCK_FILE}..."
    (cd "${PROJECT_PATH}" && uv lock)
    git add "${UV_LOCK_FILE}"
    if ! git diff --cached --quiet; then
        git commit -m "chore: regenerate ${UV_LOCK_FILE}"
    else
        echo "   ℹ️  ${UV_LOCK_FILE} не изменился"
    fi
fi

echo ""
echo "✅ Готово! Не забудьте запушить: git push origin ${BRANCH_NAME}"
