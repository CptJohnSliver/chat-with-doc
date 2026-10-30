ARG BASE_IMAGE="docker-release.registry-ci.delta.sbrf.ru/ci00224507/ci04415252/anthropic-python-image:3.13-sl10.1.2-v1.2"
ARG SERVICE_NAME="legalant-chat-with-documents"

FROM ${BASE_IMAGE} AS builder
ARG SERVICE_NAME
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_MANAGED_PYTHON=1 \
    UV_CONFIG_FILE=/build/settings/uv/uv.toml \
    PIP_CONFIG_FILE=/build/settings/pip/pip.conf \
    PATH=/app/.venv/bin:$PATH

USER root

WORKDIR /app

COPY /$SERVICE_NAME/backend/pyproject.toml /app
COPY /$SERVICE_NAME/backend/uv.lock        /app
COPY /$SERVICE_NAME/sberbank/settings      /build/settings

RUN --mount=type=secret,id=OSC_TOKEN \
    --mount=type=secret,id=NEXUS_TOKEN \
    OSC_TOKEN=$(cat /run/secrets/OSC_TOKEN) && \
    NEXUS_TOKEN=$(cat /run/secrets/NEXUS_TOKEN) && \
    sed -i "s/<OSC_TOKEN>/$OSC_TOKEN/g" $PIP_CONFIG_FILE $UV_CONFIG_FILE && \
    sed -i "s/<NEXUS_TOKEN>/$NEXUS_TOKEN/g" $PIP_CONFIG_FILE $UV_CONFIG_FILE && \
    pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev --no-install-project

FROM ${BASE_IMAGE} AS runner
ARG SERVICE_NAME
ENV LANG=en_US.UTF-8 \
    TZ=Europe/Moscow \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH

ARG USER_ID=10001
ARG GROUP_ID=10001

USER root

RUN groupadd --gid $GROUP_ID --system appgroup && \
    useradd --uid $USER_ID --system --gid appgroup appuser

COPY --chown=$USER_ID:$GROUP_ID /$SERVICE_NAME/backend/ /app/backend/
COPY --chown=$USER_ID:$GROUP_ID --from=builder /app/.venv /app/.venv
COPY --chown=$USER_ID:$GROUP_ID --chmod=755 /$SERVICE_NAME/sberbank/run_scripts/ /app/

WORKDIR /app
USER $USER_ID:$GROUP_ID

CMD ["/bin/sh", "full_start.sh"]
