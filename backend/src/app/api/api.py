from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.app.api.exception_handlers import register_exception_handlers
from src.app.api.routers.document_process import router as document_process_router
from src.app.api.routers.health import router as health_router
from src.app.mcp import mcp_app
from src.common.config.config import settings
from src.infrastructure.document_store import document_local_store
from src.infrastructure.logger import bind_context, configure_logging, get_logger
from src.infrastructure.request_identity import RequestIdentityMiddleware

logger = get_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
        stale_max_age_s = max(0.0, float(settings.get("DOCUMENT_WORKSPACE_MAX_AGE_SECONDS", 86400)))
        removed_workspaces = await asyncio.to_thread(
            document_local_store.cleanup_stale_workspaces,
            max_age_seconds=stale_max_age_s,
        )
        await logger.ainfo(
            "service_starting",
            version=settings.get("APP_VERSION", "1.0.0"),
            removed_stale_workspaces=removed_workspaces,
        )
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_app.lifespan(app))
            yield
        await logger.ainfo("service_stopping")

    app = FastAPI(
        title="Chat with documents",
        version=str(settings.get("APP_VERSION", "1.0.0")),
        root_path=str(settings.get("API_ROOT", "") or ""),
        debug=False,
        lifespan=lifespan,
    )

    app.include_router(
        document_process_router,
        prefix="/document_process",
        tags=["document_process"],
    )
    app.include_router(health_router, prefix="/health", tags=["Health"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Last added = outermost: identity is bound before CORS and routers, including /mcp.
    app.add_middleware(RequestIdentityMiddleware)

    app.mount("/mcp", mcp_app)

    return app


configure_logging()
bind_context(service_name="chat-with-documents")

app = create_app()
