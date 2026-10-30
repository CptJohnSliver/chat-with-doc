## Multi-stage version for kaniko
# Сборка
FROM cr.yandex/crpel251bjgcjbhtu439/infrastruction/python:3.13-slim as builder

# Pip без кэширования
RUN pip install --no-cache-dir "uv"

WORKDIR /app/backend

# Копирование файлов зависимостей
COPY ./backend/pyproject.toml ./backend/uv.lock ./

# venv для зависимостей
# Создаём project venv рядом с pyproject/uv.lock, чтобы `uv sync` гарантированно ставил пакеты
# именно в этот venv (иначе он может создать/использовать другой env по умолчанию).
RUN uv venv .venv
ENV PATH="/app/backend/.venv/bin:$PATH"

# Установка только runtime-зависимостей (без [dependency-groups] dev)
RUN uv sync --locked --no-dev

# === Финальная сборка ===
FROM python:3.13.7-slim

# Нерутовый юзер
ARG USER_ID=10001
ARG GROUP_ID=10001

# Создание non-root пользователя
RUN addgroup --gid $GROUP_ID --system appgroup && \
    adduser --uid $USER_ID --system --ingroup appgroup appuser

# Переход в рабочую директорию
WORKDIR /app/backend

# Копирование исходников приложения
COPY --chown=$USER_ID:$GROUP_ID ./backend .

# Копирование зависимостей из builder (после COPY исходников, чтобы локальная .venv не перезатирала linux-venv)
COPY --from=builder /app/backend/.venv /app/backend/.venv

# 🛠 Создаём папку cache и даём права
RUN mkdir -p /app/backend/cache \
    && chown -R $USER_ID:$GROUP_ID /app/backend/cache

# Копирование скрипта запуска из ci_cd/start_scripts
COPY --chown=$USER_ID:$GROUP_ID ./ci_cd/start_scripts/kplus_start.sh /app/backend/start.sh

RUN chmod +x /app/backend/start.sh

# Установка переменной PATH
ENV PATH="/app/backend/.venv/bin:$PATH"
ENV VENV_BIN="/app/backend/.venv/bin"

# Запуск под non-root пользователем
USER $USER_ID:$GROUP_ID

# Команда запуска
CMD ["sh", "start.sh"]
