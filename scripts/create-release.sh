#!/bin/bash
set -euo pipefail

# ============================================
# create-release.sh
# Создание release-ветки из GitLab с адаптацией под закрытый контур
#
# Использование:
#   ./scripts/create-release.sh <LOS_NUMBER> [UPSTREAM_REF]
#   PROJECT_DIR=backend ./scripts/create-release.sh <LOS_NUMBER> [UPSTREAM_REF]
#
# Примеры:
#   ./scripts/create-release.sh 42
#   ./scripts/create-release.sh 42 refs/tags/v2.5.0
#   PUSH=0 ./scripts/create-release.sh 42 upstream/main
# ============================================

source "$(dirname "$0")/lib-project.sh"

LOS_NUMBER="${1:?❌ Укажите номер LOS: ./scripts/create-release.sh <NUMBER> [UPSTREAM_REF]}"
UPSTREAM_REF="${2:-upstream/main}"
BRANCH_NAME="release/LOS-${LOS_NUMBER}-$(date +%Y)"
PUSH="${PUSH:-1}"

# Проверка, что upstream настроен
if ! git remote | grep -q "^upstream$"; then
    echo "❌ Remote 'upstream' не настроен. Запустите ./scripts/init-repo-in-bb.sh"
    exit 1
fi

ensure_clean_worktree

echo "═══════════════════════════════════════════"
echo "🔄 Создание релиза LOS-${LOS_NUMBER}"
echo "   Источник: ${UPSTREAM_REF}"
echo "   Ветка:    ${BRANCH_NAME}"
echo "   Push:     ${PUSH}"
echo "═══════════════════════════════════════════"

echo ""
echo "📌 Шаг 1/6: Получаем изменения из GitLab..."
fetch_upstream
UPSTREAM_COMMIT=$(resolve_commit_ref "${UPSTREAM_REF}")

echo "📌 Шаг 2/6: Создаём ветку ${BRANCH_NAME} из ${UPSTREAM_REF}..."
if git rev-parse --verify "${BRANCH_NAME}" >/dev/null 2>&1; then
    echo "❌ Ветка ${BRANCH_NAME} уже существует!"
    exit 1
fi
git checkout -b "${BRANCH_NAME}" "${UPSTREAM_COMMIT}"

echo "📌 Шаг 3/6: Копируем закрытые файлы из master BitBucket..."
if git rev-parse --verify master >/dev/null 2>&1; then
    if git ls-tree -r master --name-only | grep -q "^sberbank/"; then
        git checkout master -- sberbank/
        echo "   ✅ sberbank/ скопирован из master"
    else
        echo "   ⚠️  sberbank/ отсутствует в master, создаём базовую структуру..."
        mkdir -p sberbank/run_scripts sberbank/settings
        cat > sberbank/README.md << 'EOF'
# Специфика закрытого контура

Эта директория содержит файлы, специфичные для закрытого контура:
- `run_scripts/` — скрипты запуска для закрытого контура
- `settings/` — переопределения конфигов
EOF
        git add sberbank/
        git commit -m "chore: initialize sberbank/ directory structure"
    fi

    if git ls-tree -r master --name-only | grep -q "^scripts/"; then
        git checkout master -- scripts/
        echo "   ✅ scripts/ скопирован из master"
    else
        echo "   ⚠️  scripts/ отсутствует в master"
    fi
else
    echo "   ⚠️  Ветка master не найдена, пропускаем копирование закрытых файлов"
fi

echo "📌 Шаг 4/6: Пересобираем uv.lock..."
setup_project_paths
echo "   Проект: ${PROJECT_DIR}"

(cd "${PROJECT_PATH}" && uv lock)

echo "📌 Шаг 5/6: Коммитим адаптацию..."
if [ -e sberbank ]; then
    git add sberbank/
fi
if [ -e scripts ]; then
    git add scripts/
fi
if [ -e "${UV_LOCK_FILE}" ]; then
    git add "${UV_LOCK_FILE}"
fi
if ! git diff --cached --quiet; then
    UPSTREAM_SHA=$(git rev-parse --short "${UPSTREAM_COMMIT}")
    git commit -m "chore(los-${LOS_NUMBER}): sync from GitLab ${UPSTREAM_REF}

- Merged upstream changes into closed contour
- Regenerated ${UV_LOCK_FILE}
- Source: ${UPSTREAM_REF} (${UPSTREAM_SHA})"
else
    echo "   ℹ️  Нет изменений для коммита"
fi

if [ "${PUSH}" = "1" ]; then
    echo "📌 Шаг 6/6: Пушим ветку в BitBucket..."
    git push origin "${BRANCH_NAME}"
else
    echo "📌 Шаг 6/6: PUSH=0, пропускаем push"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "✅ Готово! Ветка ${BRANCH_NAME} создана."
echo "═══════════════════════════════════════════"
echo ""
echo "📌 Следующие шаги:"
echo "  1. Дождитесь проверок в BitBucket"
echo "  2. Протестируйте релиз в закрытом контуре"
echo "  3. Если нужен фикс из GitLab:"
echo "     ./scripts/cherry-pick-upstream.sh <SHA>"
echo "  4. После успеха:"
echo "     ./scripts/finalize-release.sh ${BRANCH_NAME}"
