"""Per-request identity for llm-gateway headers.

Bound once at the HTTP edge (ASGI middleware) from inbound headers, then
optionally filled from the old chat-with-documents contract (`thread_id` = dialog = chat,
`message_id` = one invoke = gateway thread). ChatOpenAI clients read it via
the httpx request hook. Incoming `X-Service-Name` is not forwarded: the LLM
client keeps `chat_w_doc`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from uuid import uuid4

import httpx
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from src.infrastructure.logger import bind_context, unbind_context

HEADER_USER_ID = "X-User-Id"
HEADER_ORGANIZATION_ID = "X-Organization-Id"
HEADER_CHAT_ID = "X-Chat-Id"
HEADER_THREAD_ID = "X-Thread-Id"
HEADER_REQUEST_ID = "X-Request-ID"  # per ChatOpenAI hop, not part of bound identity

_HEADER_VALUE_MAX = 256
_SSE_CHAT_ID_PATH = re.compile(r"/document_process/document_sse/([^/]+)$")


def _clip(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped[:_HEADER_VALUE_MAX]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return value
    lower = name.lower()
    for key, candidate in headers.items():
        if key.lower() == lower:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    user_id: str | None = None
    organization_id: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None

    def fill_missing(
        self,
        *,
        user_id: str | None = None,
        organization_id: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
    ) -> RequestIdentity:
        """Keep already bound values (headers win over the old body/path contract)."""
        return RequestIdentity(
            user_id=self.user_id or _clip(user_id),
            organization_id=self.organization_id or _clip(organization_id),
            chat_id=self.chat_id or _clip(chat_id),
            thread_id=self.thread_id or _clip(thread_id),
        )

    def as_gateway_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.user_id is not None:
            headers[HEADER_USER_ID] = self.user_id
        if self.organization_id is not None:
            headers[HEADER_ORGANIZATION_ID] = self.organization_id
        if self.chat_id is not None:
            headers[HEADER_CHAT_ID] = self.chat_id
        if self.thread_id is not None:
            headers[HEADER_THREAD_ID] = self.thread_id
        return headers


_EMPTY_IDENTITY = RequestIdentity()
_IDENTITY = ContextVar("request_identity", default=_EMPTY_IDENTITY)


def current_identity() -> RequestIdentity:
    return _IDENTITY.get()


def identity_from_headers(headers: Mapping[str, str]) -> RequestIdentity:
    return RequestIdentity(
        user_id=_clip(_header(headers, HEADER_USER_ID)),
        organization_id=_clip(_header(headers, HEADER_ORGANIZATION_ID)),
        chat_id=_clip(_header(headers, HEADER_CHAT_ID)),
        thread_id=_clip(_header(headers, HEADER_THREAD_ID)),
    )


def _log_request_id(headers: Mapping[str, str]) -> str:
    """Correlation id for this chat-with-documents HTTP/MCP call. Not forwarded to the gateway."""
    return _clip(_header(headers, HEADER_REQUEST_ID)) or uuid4().hex


def bind_identity(identity: RequestIdentity) -> RequestIdentity:
    """Replace the current identity. Used on the MCP session task for this tools/call."""
    _IDENTITY.set(identity)
    return identity


def overlay_identity(
    *,
    user_id: str | None = None,
    organization_id: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
) -> RequestIdentity:
    updated = current_identity().fill_missing(
        user_id=user_id,
        organization_id=organization_id,
        chat_id=chat_id,
        thread_id=thread_id,
    )
    _IDENTITY.set(updated)
    return updated


def bind_conversation(*, chat_id: str, message_id: str | None) -> RequestIdentity:
    """Old contract: `thread_id` is the dialog; `message_id` is one invoke."""
    return overlay_identity(chat_id=chat_id, thread_id=message_id)


def attach_identity_headers(request: httpx.Request) -> None:
    for name, value in current_identity().as_gateway_headers().items():
        if name not in request.headers:
            request.headers[name] = value
    if HEADER_REQUEST_ID not in request.headers:
        request.headers[HEADER_REQUEST_ID] = uuid4().hex


async def attach_identity_headers_async(request: httpx.Request) -> None:
    """httpx 0.28 AsyncClient always `await`s request hooks; a sync hook returns None."""
    attach_identity_headers(request)


def _chat_id_from_path(path: str) -> str | None:
    match = _SSE_CHAT_ID_PATH.search(path)
    if match is None:
        return None
    return match.group(1)


class RequestIdentityMiddleware:
    """Bind identity as soon as the request is received.

    Pure ASGI — `BaseHTTPMiddleware` buffers the body and breaks SSE.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        identity = identity_from_headers(headers).fill_missing(chat_id=_chat_id_from_path(scope.get("path") or ""))
        token: Token[RequestIdentity] = _IDENTITY.set(identity)
        bind_context(request_id=_log_request_id(headers))
        try:
            await self.app(scope, receive, send)
        finally:
            _IDENTITY.reset(token)
            unbind_context("request_id")
