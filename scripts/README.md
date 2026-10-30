# Repository Sync Scripts

Набор скриптов в этой директории помогает синхронизировать закрытый BitBucket-репозиторий с upstream-репозиторием в GitLab.

## Основная схема

Рекомендуемый поток:

1. Один раз настроить remotes:
   ```bash
   ./scripts/init-repo-in-bb.sh
   ```
2. Создать release-ветку из upstream:
   ```bash
   ./scripts/create-release.sh 42 upstream/main
   ```
3. Для релиза от тега использовать `refs/tags/<tag>`:
   ```bash
   ./scripts/create-release.sh 42 refs/tags/los-2-2026-v1
   ```
4. Если нужен отдельный фикс из upstream:
   ```bash
   ./scripts/cherry-pick-upstream.sh <SHA>
   ```
5. После проверки релиза слить его в master и создать тег:
   ```bash
   ./scripts/finalize-release.sh release/LOS-42-2026
   ```

## Где лежит Python-проект

Скрипты автоматически ищут `pyproject.toml`:

- сначала в корне репозитория;
- затем в `backend/`.

Если структура нестандартная, укажите путь явно:

```bash
PROJECT_DIR=backend ./scripts/create-release.sh 42 refs/tags/los-2-2026-v1
PROJECT_DIR=services/api ./scripts/cherry-pick-upstream.sh <SHA>
```

`uv lock` запускается внутри найденной директории проекта.

## Важные правила

- Перед запуском рабочая директория должна быть чистой: без staged, unstaged и untracked изменений.
- Для тегов используйте `refs/tags/<tag>` или просто имя тега, но не `upstream/tags/<tag>`.
- `create-release.sh` по умолчанию пушит release-ветку в `origin`.
- Для тестового запуска без push используйте:
  ```bash
  PUSH=0 ./scripts/create-release.sh 42 refs/tags/los-2-2026-v1
  ```

## Что делают скрипты

### `init-repo-in-bb.sh`

Первичная настройка remotes. Скрипт переименовывает исходный GitLab `origin` в `upstream` и добавляет BitBucket как новый `origin`.

### `create-release.sh`

Создает ветку `release/LOS-<number>-<year>` из upstream ref.

Скрипт:

- делает `git fetch upstream --tags --prune`;
- проверяет, что upstream ref является коммитом;
- копирует закрытые файлы `sberbank/` и `scripts/` из `master`;
- пересобирает `uv.lock`;
- коммитит адаптацию;
- пушит release-ветку, если не задан `PUSH=0`.

### `cherry-pick-upstream.sh`

Подтягивает один или несколько коммитов из upstream в текущую ветку:

```bash
./scripts/cherry-pick-upstream.sh abc1234 def5678
```

Если в cherry-pick попал `pyproject.toml`, скрипт пересобирает `uv.lock` и создает отдельный commit:

```text
chore: regenerate backend/uv.lock
```

### `finalize-release.sh`

Сливает release-ветку в `master`, создает тег `los-<number>-<year>` и пушит master с тегом.

## Типовые ошибки

### `upstream/tags/... не является коммитом`

Использован неверный формат ref для тега.

Правильно:

```bash
./scripts/create-release.sh 2 refs/tags/los-2-2026-v1
```

Неправильно:

```bash
./scripts/create-release.sh 2 upstream/tags/los-2-2026-v1
```

### `Рабочая директория содержит незакоммиченные изменения`

Скрипты специально останавливаются, чтобы случайно не включить локальные изменения в release commit.

Проверьте:

```bash
git status --short
```

Затем закоммитьте, уберите или сохраните изменения отдельно.
