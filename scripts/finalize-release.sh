#!/bin/bash
set -euo pipefail

# ============================================
# finalize-release.sh
# Слияние release-ветки в master + тегирование
#
# Использование:
#   ./scripts/finalize-release.sh <release-branch>
#
# Пример:
#   ./scripts/finalize-release.sh release/LOS-42-2026
# ============================================

RELEASE_BRANCH="${1:?❌ Укажите release-ветку: ./scripts/finalize-release.sh <branch>}"

# Проверяем существование ветки
if ! git rev-parse --verify "${RELEASE_BRANCH}" >/dev/null 2>&1; then
    echo "❌ Ветка ${RELEASE_BRANCH} не найдена"
    exit 1
fi

# Извлекаем номер LOS из имени ветки
LOS_NUMBER=$(echo "${RELEASE_BRANCH}" | sed -E 's|release/LOS-([0-9]+)-([0-9]+)|\1|')
YEAR=$(echo "${RELEASE_BRANCH}" | sed -E 's|release/LOS-([0-9]+)-([0-9]+)|\2|')
TAG_NAME="los-${LOS_NUMBER}-${YEAR}"

echo "═══════════════════════════════════════════"
echo "🏁 Завершение релиза"
echo "   Ветка: ${RELEASE_BRANCH}"
echo "   Тег:   ${TAG_NAME}"
echo "═══════════════════════════════════════════"
echo ""

# Получаем информацию об источнике из последнего коммита
SOURCE_INFO=$(git log -1 --format=%s "${RELEASE_BRANCH}" | grep -oP 'from GitLab \K.*' || echo "unknown")

echo "📌 Шаг 1/4: Переключаемся на master..."
git checkout master
git pull origin master

echo "📌 Шаг 2/4: Сливаем ${RELEASE_BRANCH} в master..."
git merge --no-ff "${RELEASE_BRANCH}" -m "release: LOS-${LOS_NUMBER}-${YEAR} (from GitLab ${SOURCE_INFO})"

echo "📌 Шаг 3/4: Создаём тег ${TAG_NAME}..."
git tag -a "${TAG_NAME}" -m "Release LOS-${LOS_NUMBER} based on GitLab ${SOURCE_INFO}"

echo "📌 Шаг 4/4: Пушим master и теги..."
git push origin master
git push origin "${TAG_NAME}"

echo ""
echo "═══════════════════════════════════════════"
echo "✅ Релиз LOS-${LOS_NUMBER}-${YEAR} завершён!"
echo "═══════════════════════════════════════════"
echo ""
echo "📌 Опционально: удалить release-ветку"
echo "   git branch -d ${RELEASE_BRANCH}"
echo "   git push origin --delete ${RELEASE_BRANCH}"
