## Запуск локально в Docker


### 1) Подготовить сеть (обязательно)

В `ci_cd/docker/docker-compose.yml` сеть `front_net` объявлена как **external**, поэтому её нужно создать один раз:

```bash
docker network inspect front_net >/dev/null 2>&1 || docker network create front_net
```

### 2) Подготовить env-файл для compose

`docker compose` берёт переменные из **`ci_cd/docker/.env`** (важно: не из `backend/.env` и не из корня).

```bash
cp backend/env.local.example ci_cd/docker/.env
```

Дальше открой `ci_cd/docker/.env` и заполни минимум ключи, которые тебе нужны (подглядеть можно в `.env.example`).

Полезные переменные:
- **`BASE_MCP_URL`** или **`BASE_URL`**: адрес MCP-сервера (если используешь MCP-инструменты).
- **`MCP_WARMUP_ENABLED`**: прогрев MCP tools при старте (по умолчанию `false`; включай `true`, если MCP доступен).

### 3) Собрать и запустить контейнер

Из корня репозитория:

```bash
docker compose -f ci_cd/docker/docker-compose.yml up --build -d backend
```

Остановка:

```bash
docker compose -f ci_cd/docker/docker-compose.yml down
```

### 4) Проверка работоспособности

- Swagger: `http://localhost:8000/docs`
- OpenAPI: `http://localhost:8000/openapi.json`
- или
```bash
docker logs -n 200 k-agent-backend
```

### 5) Если захотелось пересобрать

```bash
docker compose -f ci_cd/docker/docker-compose.yml down
docker compose -f ci_cd/docker/docker-compose.yml build --no-cache backend
docker compose -f ci_cd/docker/docker-compose.yml up -d backend
```


### Типовые проблемы

- **`network front_net not found`**: выполни команду создания сети из шага 1.
- **Compose ругается на `.env`**: проверь, что файл именно `ci_cd/docker/.env`.
- **В логах ошибки MCP (`/kb_searcher/mcp` 404 и т.п.)**: это значит, что `BASE_MCP_URL/BASE_URL` указывает не туда или MCP не поднят. Либо настрой URL, либо оставь `MCP_WARMUP_ENABLED=false`.
