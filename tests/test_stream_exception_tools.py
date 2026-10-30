from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.app.api.utils.sse import sse_pack
from src.common.exceptions import (
    BadRequest,
    ServerException,
    build_error_terminal_packet,
    build_sse_error_frame,
    metrics_exception_type,
    mcp_exception_handler,
    stream_exception_handler,
    wrap_exception,
)

TERMINAL_STATUSES = {"success", "error"}


def _sse_event_data_str(event_text: str) -> str:
    data_lines: list[str] = []
    for line in event_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            _, value = line.split(":", 1)
            data_lines.append(value.lstrip())
    return "\n".join(data_lines).strip()


def parse_terminal_payload(event_text: str) -> dict[str, Any] | None:
    data_str = _sse_event_data_str(event_text)
    if not data_str:
        return None

    obj = json.loads(data_str)
    if not isinstance(obj, dict):
        return None

    payload = obj.get("data")
    if not isinstance(payload, dict):
        return None

    if payload.get("status") in TERMINAL_STATUSES:
        return payload

    return None


def test_wrap_exception_returns_domain_exception_as_is() -> None:
    exc = BadRequest(detail="bad")
    assert wrap_exception(exc) is exc


def test_wrap_exception_maps_unknown_to_server_exception() -> None:
    source = RuntimeError("secret internal URL")
    wrapped = wrap_exception(source)
    assert isinstance(wrapped, ServerException)
    assert wrapped.source is source
    assert wrapped.user_message == "Ошибка сервера"
    assert "secret internal URL" not in wrapped.user_message


def test_metrics_exception_type_for_server_exception_uses_source() -> None:
    source = RuntimeError("secret internal URL")
    wrapped = ServerException(source=source)
    assert metrics_exception_type(wrapped) == "RuntimeError"


def test_domain_exception_to_base_uses_user_message_as_detail() -> None:
    base = BadRequest(detail="technical only").to_base()
    assert base.detail == base.user_message == "Некорректный запрос"


def test_build_sse_error_frame_for_direct_client() -> None:
    frame = build_sse_error_frame(ServerException(source=RuntimeError("secret internal URL")).to_base())
    assert frame.startswith("event: error\n")
    assert "Ошибка сервера" in frame
    assert "secret internal URL" not in frame


def test_build_error_terminal_packet_for_orchestrator() -> None:
    error = ServerException(source=RuntimeError("secret")).to_base()
    packet = build_error_terminal_packet(
        message_id="m1",
        created_at=100.0,
        fragments=[],
        error=error,
    )
    frame = sse_pack(packet)
    assert frame.startswith("data: ")
    assert "event: error" not in frame
    terminal = parse_terminal_payload(frame)
    assert terminal is not None
    assert terminal["status"] == "error"
    assert terminal["fragments"][-1]["content"] == "Ошибка сервера"


@pytest.mark.asyncio
async def test_stream_exception_handler_yields_error_frame_without_orchestrator() -> None:
    @stream_exception_handler(log_event="test_stream_error")
    async def failing_stream():
        yield "data: ok\n\n"
        raise RuntimeError("secret internal URL")

    chunks = [chunk async for chunk in failing_stream()]

    assert chunks[0] == "data: ok\n\n"
    assert "event: error" in chunks[-1]
    assert "Ошибка сервера" in chunks[-1]
    assert "secret internal URL" not in chunks[-1]


@pytest.mark.asyncio
async def test_stream_exception_handler_yields_terminal_packet_for_orchestrator() -> None:
    @stream_exception_handler(
        log_event="test_terminal_handler",
        to_terminal_packet=lambda error, message_id, created_at, **_: build_error_terminal_packet(
            message_id=message_id,
            created_at=created_at,
            fragments=[],
            error=error,
        ),
    )
    async def failing_stream(message_id: str, created_at: float):
        if False:
            yield ""
        raise RuntimeError("secret internal URL")

    chunks = [chunk async for chunk in failing_stream("m1", 100.0)]

    assert len(chunks) == 1
    assert "secret internal URL" not in chunks[0]
    assert "event: error" not in chunks[0]
    terminal = parse_terminal_payload(chunks[0])
    assert terminal is not None
    assert terminal["status"] == "error"
    assert "Ошибка сервера" in chunks[0]


@pytest.mark.asyncio
async def test_stream_exception_handler_preserves_domain_exception_message() -> None:
    @stream_exception_handler(
        log_event="test_domain_stream_error",
        to_terminal_packet=lambda error, **_: build_error_terminal_packet(
            message_id="m1",
            created_at=100.0,
            fragments=[],
            error=error,
        ),
    )
    async def failing_stream():
        if False:
            yield ""
        raise BadRequest(detail="invalid payload")

    chunks = [chunk async for chunk in failing_stream()]

    assert len(chunks) == 1
    assert "event: error" not in chunks[0]
    terminal = parse_terminal_payload(chunks[0])
    assert terminal is not None
    assert terminal["status"] == "error"
    assert "Некорректный запрос" in chunks[0]


@pytest.mark.asyncio
async def test_mcp_exception_handler_returns_error_result() -> None:
    ctx = {"message_id": "m1", "created_at": 100.0, "fragments": []}
    events: list[dict[str, Any]] = []

    async def capture_event(payload: dict[str, Any]) -> None:
        events.append(payload)

    @mcp_exception_handler(
        log_event="test_mcp_handler",
        to_terminal_packet=lambda error, stream_ctx, **_: build_error_terminal_packet(
            message_id=stream_ctx["message_id"],
            created_at=stream_ctx["created_at"],
            fragments=stream_ctx["fragments"],
            error=error,
        ),
    )
    async def failing_stream(stream_ctx, *, on_event=None):
        raise RuntimeError("secret internal URL")

    final, finish_reason = await failing_stream(ctx, on_event=capture_event)

    assert finish_reason == "error"
    assert final["data"]["status"] == "error"
    assert len(events) == 1
    assert events[0]["data"]["status"] == "error"
    assert "secret internal URL" not in str(final)


@pytest.mark.asyncio
async def test_mcp_exception_handler_matches_multi_arg_stream_signature() -> None:
    ctx = {"message_id": "m1", "created_at": 100.0, "fragments": []}
    events: list[dict[str, Any]] = []

    async def capture_event(payload: dict[str, Any]) -> None:
        events.append(payload)

    @mcp_exception_handler(
        log_event="test_mcp_multi_arg",
        to_terminal_packet=lambda error, _service, _document_app, stream_ctx, **_: build_error_terminal_packet(
            message_id=stream_ctx["message_id"],
            created_at=stream_ctx["created_at"],
            fragments=stream_ctx["fragments"],
            error=error,
        ),
    )
    async def failing_stream(service, document_app, stream_ctx, *, on_event=None):
        raise RuntimeError("secret internal URL")

    final, finish_reason = await failing_stream(object(), object(), ctx, on_event=capture_event)

    assert finish_reason == "error"
    assert final["data"]["status"] == "error"
    assert final["data"]["fragments"][-1]["content"] == "Ошибка сервера"
    assert len(events) == 1
    assert "secret internal URL" not in str(final)


@pytest.mark.asyncio
async def test_mcp_exception_handler_reraises_cancelled() -> None:
    @mcp_exception_handler(
        log_event="test_mcp_cancelled",
        to_terminal_packet=lambda error, stream_ctx, **_: build_error_terminal_packet(
            message_id=stream_ctx["message_id"],
            created_at=stream_ctx["created_at"],
            fragments=stream_ctx["fragments"],
            error=error,
        ),
    )
    async def cancelled_stream(stream_ctx, *, on_event=None):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled_stream({"message_id": "m1", "created_at": 100.0, "fragments": []})


@pytest.mark.asyncio
async def test_stream_exception_handler_reraises_cancelled() -> None:
    @stream_exception_handler(log_event="test_cancelled")
    async def cancelled_stream():
        yield "data: ok\n\n"
        raise asyncio.CancelledError

    stream = cancelled_stream()
    assert await anext(stream) == "data: ok\n\n"
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)
