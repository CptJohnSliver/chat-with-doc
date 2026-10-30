import asyncio
from contextvars import copy_context

import httpx
import pytest
from src.infrastructure.request_identity import (
    HEADER_CHAT_ID,
    HEADER_THREAD_ID,
    HEADER_USER_ID,
    RequestIdentity,
    RequestIdentityMiddleware,
    attach_identity_headers,
    attach_identity_headers_async,
    bind_conversation,
    bind_identity,
    current_identity,
    identity_from_headers,
    overlay_identity,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def test_identity_from_headers_is_case_insensitive() -> None:
    identity = identity_from_headers({"x-user-id": "user-42", "X-CHAT-ID": "chat-9"})

    assert identity.user_id == "user-42"
    assert identity.chat_id == "chat-9"


def test_mcp_session_task_replaces_stale_identity_then_fills_old_contract() -> None:
    """Stateful FastMCP keeps initialize-time ContextVar; tools/call must replace it."""

    def _run() -> RequestIdentity:
        overlay_identity(user_id="from-initialize", chat_id="init-chat")
        bind_identity(identity_from_headers({"x-user-id": "from-tools-call"}))
        return bind_conversation(chat_id="dialog-from-body", message_id="msg-from-body")

    identity = copy_context().run(_run)

    assert identity.user_id == "from-tools-call"
    assert identity.chat_id == "dialog-from-body"
    assert identity.thread_id == "msg-from-body"


def test_identity_from_headers_clips_and_skips_blank() -> None:
    identity = identity_from_headers(
        {
            HEADER_USER_ID: "  user-1  ",
            HEADER_CHAT_ID: "   ",
            HEADER_THREAD_ID: "t" * 300,
        }
    )

    assert identity.user_id == "user-1"
    assert identity.chat_id is None
    assert identity.thread_id is not None and len(identity.thread_id) == 256


def test_headers_win_over_old_contract_overlay() -> None:
    bound = identity_from_headers({HEADER_CHAT_ID: "from-header"}).fill_missing(
        chat_id="from-body-thread-id",
        thread_id="message-9",
        user_id="from-body-user",
    )

    assert bound.chat_id == "from-header"
    assert bound.thread_id == "message-9"
    assert bound.user_id == "from-body-user"


def test_bind_conversation_maps_thread_id_to_chat_and_message_to_thread() -> None:
    def _run() -> RequestIdentity:
        overlay_identity(user_id="u-1")
        return bind_conversation(chat_id="dialog-1", message_id="msg-2")

    identity = copy_context().run(_run)

    assert identity.user_id == "u-1"
    assert identity.chat_id == "dialog-1"
    assert identity.thread_id == "msg-2"


def test_overlay_user_id_from_body_fills_when_header_absent() -> None:
    def _run() -> RequestIdentity:
        bind_identity(identity_from_headers({HEADER_CHAT_ID: "chat-1"}))
        bind_conversation(chat_id="dialog-from-body", message_id="msg-1")
        return overlay_identity(user_id="body-user")

    identity = copy_context().run(_run)

    assert identity.user_id == "body-user"
    assert identity.chat_id == "chat-1"
    assert identity.thread_id == "msg-1"


def test_header_user_id_wins_over_body_user_id() -> None:
    def _run() -> RequestIdentity:
        bind_identity(identity_from_headers({HEADER_USER_ID: "header-user"}))
        return overlay_identity(user_id="body-user")

    identity = copy_context().run(_run)
    assert identity.user_id == "header-user"


def test_attach_identity_headers_mints_request_id_per_llm_call() -> None:
    def _run() -> tuple[httpx.Request, httpx.Request]:
        overlay_identity(user_id="u-1", chat_id="c-1")
        first = httpx.Request("POST", "http://gateway/v1/chat/completions", headers={HEADER_USER_ID: "already"})
        second = httpx.Request("POST", "http://gateway/v1/chat/completions")
        attach_identity_headers(first)
        attach_identity_headers(second)
        return first, second

    first, second = copy_context().run(_run)

    assert first.headers[HEADER_USER_ID] == "already"
    assert first.headers[HEADER_CHAT_ID] == "c-1"
    assert first.headers["X-Request-ID"]
    assert second.headers["X-Request-ID"]
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert "X-Service-Name" not in first.headers


def test_attach_identity_headers_keeps_existing_request_id() -> None:
    def _run() -> httpx.Request:
        request = httpx.Request("POST", "http://gateway/v1/chat/completions", headers={"X-Request-ID": "already"})
        attach_identity_headers(request)
        return request

    request = copy_context().run(_run)
    assert request.headers["X-Request-ID"] == "already"


def test_attach_identity_headers_async_mints_request_id() -> None:
    def _run() -> httpx.Request:
        overlay_identity(user_id="u-1", chat_id="c-1")
        request = httpx.Request("POST", "http://gateway/v1/chat/completions")
        asyncio.run(attach_identity_headers_async(request))
        return request

    request = copy_context().run(_run)

    assert request.headers[HEADER_USER_ID] == "u-1"
    assert request.headers[HEADER_CHAT_ID] == "c-1"
    assert request.headers["X-Request-ID"]


def test_async_httpx_client_awaits_async_identity_hook() -> None:
    """httpx 0.28 AsyncClient always `await`s request hooks; the async hook must be awaitable."""
    request = httpx.Request("POST", "http://gateway/v1/chat/completions")

    async def _run_hooks() -> None:
        overlay_identity(user_id="u-1")
        async with httpx.AsyncClient(event_hooks={"request": [attach_identity_headers_async]}) as client:
            for hook in client.event_hooks["request"]:
                await hook(request)

    copy_context().run(lambda: asyncio.run(_run_hooks()))
    assert request.headers[HEADER_USER_ID] == "u-1"
    assert request.headers["X-Request-ID"]


def test_async_httpx_client_cannot_await_sync_identity_hook() -> None:
    async def _send() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
            event_hooks={"request": [attach_identity_headers]},
        ) as client:
            await client.post("http://gateway/v1/chat/completions")

    with pytest.raises(TypeError, match="await"):
        asyncio.run(_send())


def _identity_app() -> Starlette:
    async def ping(_request: Request) -> JSONResponse:
        identity = current_identity()
        return JSONResponse(
            {
                "user_id": identity.user_id,
                "chat_id": identity.chat_id,
                "thread_id": identity.thread_id,
            }
        )

    async def stream(_request: Request) -> StreamingResponse:
        async def chunks():
            identity = current_identity()
            yield f"{identity.chat_id or ''}:{identity.user_id or 'missing'}".encode()

        return StreamingResponse(chunks(), media_type="text/plain")

    return Starlette(
        middleware=[Middleware(RequestIdentityMiddleware)],
        routes=[
            Route("/ping", ping),
            Route("/document_process/document_sse/{thread_id}", stream, methods=["POST"]),
            Route("/stream", stream),
        ],
    )


def test_middleware_binds_headers_immediately() -> None:
    client = TestClient(_identity_app())

    response = client.get(
        "/ping",
        headers={HEADER_USER_ID: "user-42", HEADER_CHAT_ID: "chat-9"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-42",
        "chat_id": "chat-9",
        "thread_id": None,
    }
    assert current_identity() == RequestIdentity()


def test_middleware_fills_chat_id_from_sse_path_when_header_absent() -> None:
    client = TestClient(_identity_app())

    response = client.post("/document_process/document_sse/dialog-uuid", headers={HEADER_USER_ID: "u-1"})

    assert response.status_code == 200
    assert response.text == "dialog-uuid:u-1"


def test_middleware_header_chat_id_wins_over_sse_path() -> None:
    captured: dict[str, str | None] = {}

    async def ping(_request: Request) -> JSONResponse:
        captured["chat_id"] = current_identity().chat_id
        return JSONResponse({"ok": True})

    app = Starlette(
        middleware=[Middleware(RequestIdentityMiddleware)],
        routes=[Route("/document_process/document_sse/{thread_id}", ping, methods=["POST"])],
    )
    client = TestClient(app)

    client.post("/document_process/document_sse/from-path", headers={HEADER_CHAT_ID: "from-header"})

    assert captured["chat_id"] == "from-header"


def test_middleware_streaming_sees_bound_identity() -> None:
    client = TestClient(_identity_app())

    response = client.get("/stream", headers={HEADER_USER_ID: "stream-user"})

    assert response.text == ":stream-user"
    assert current_identity().user_id is None
