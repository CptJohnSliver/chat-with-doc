#!/bin/bash
set -euo pipefail

# ============================================
# init-repo-in-bb.sh
# Первичная настройка репозитория
# Сценарий: клон сделан из GitLab, нужно добавить BitBucket
#
# Запускается ОДИН РАЗ после git clone <GitLab URL>
# ============================================

echo "🔧 Настройка репозитория..."
echo ""

# Проверяем, что мы в корне git-репозитория
if [ ! -d ".git" ]; then
    echo "❌ Ошибка: скрипт должен запускаться в корне git-репозитория"
    exit 1
fi

# ============================================
# ШАГ 1: Проверяем текущий origin (должен быть GitLab)
# ============================================
if ! git remote | grep -q "^origin$"; then
    echo "❌ Remote 'origin' не найден. Скрипт предполагает клон из GitLab."
    exit 1
fi

CURRENT_ORIGIN=$(git remote get-url origin)
echo "📍 Текущий origin (предположительно GitLab):"
echo "   ${CURRENT_ORIGIN}"
echo ""

# ============================================
# ШАГ 2: Переименовываем origin → upstream
# ============================================
if git remote | grep -q "^upstream$"; then
    echo "⚠️  Remote 'upstream' уже существует."
    read -p "   Перезаписать? (y/N): " CONFIRM
    if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
        echo "❌ Отменено"
        exit 1
    fi
    git remote remove upstream
fi

echo "📌 Переименовываем origin → upstream..."
git remote rename origin upstream
echo "   ✅ GitLab теперь доступен как 'upstream'"

# ============================================
# ШАГ 3: Запрашиваем URL BitBucket
# ============================================
echo ""
read -p "🔗 Введите URL BitBucket (read+write): " BB_URL

if [ -z "${BB_URL}" ]; then
    echo "❌ URL не может быть пустым"
    echo "   Откатываем изменения..."
    git remote rename upstream origin
    exit 1
fi

# ============================================
# ШАГ 4: Добавляем BitBucket как origin
# ============================================
echo ""
echo "📌 Добавляем BitBucket как 'origin'..."
git remote add origin "${BB_URL}"

# Запрещаем пуш в upstream (GitLab) — защита от случайностей
git remote set-url --push upstream no_push

# ============================================
# ШАГ 5: Fetch и настройка веток
# ============================================
echo ""
echo "📌 Получаем данные из BitBucket..."
git fetch origin

# Важный момент: после клона из GitLab мы на ветке main,
# но в BitBucket основная ветка — master. Создаём локальный master.
if git ls-remote --heads origin | grep -q "refs/heads/master$"; then
    echo "📌 Создаём локальную ветку master, отслеживающую origin/master..."

    # Если локальная ветка master уже есть — переключаемся на неё
    if git rev-parse --verify master >/dev/null 2>&1; then
        git checkout master
        # Обновляем upstream tracking
        git branch --set-upstream-to=origin/master master 2>/dev/null || true
    else
        git checkout -b master origin/master
    fi
    echo "   ✅ Теперь вы на ветке master (BitBucket)"
else
    echo "⚠️  Ветка master не найдена в BitBucket."
    echo "   Возможно, это первый релиз. Ветка master будет создана"
    echo "   автоматически при первом push через finalize-release.sh"
    echo ""
    echo "   Текущая ветка: $(git branch --show-current)"
fi

# ============================================
# ШАГ 6: Финальная проверка
# ============================================
echo ""
echo "═══════════════════════════════════════════"
echo "✅ Настройка завершена!"
echo "═══════════════════════════════════════════"
echo ""
echo "📍 Текущие remotes:"
git remote -v | sed 's/^/   /'
echo ""
echo "📍 Текущая ветка: $(git branch --show-current)"
echo ""
echo "📌 Проверка настроек:"
echo "   • git pull upstream main   — получить изменения из GitLab"
echo "   • git push                 — запушить в BitBucket (по умолчанию)"
echo "   • git push upstream        — ❌ ЗАБЛОКИРОВАНО (защита от случайностей)"
echo ""
echo "📌 Следующие шаги:"
echo "   1. git fetch upstream"
echo "   2. ./scripts/create-release.sh <LOS_NUMBER> [refs/tags/vX.Y.Z]"
