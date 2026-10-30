"""Run document_process graph for MCP tool calls with streaming."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID, uuid4

from src.agents.document_process import get_document_app
from src.agents.document_process.graph import astream_events
from src.app.api.utils.document_text import extract_document_from_s3
from src.app.api.utils.stub_response import (
    STUB_ANONYMIZATION_PENDING,
    STUB_FILE_LOAD_FAILED,
    STUB_TOO_LONG,
    STUB_UPLOAD_AND_ASK,
    STUB_UPLOAD_REQUIRED,
    build_stub_message,
)
from src.app.services.document import DocumentService
from src.common.exceptions import build_error_terminal_packet, mcp_exception_handler
from src.common.schemas.models import HistoryMessage, Message, MessageFragment
from src.infrastructure.document_store import document_local_store
from src.infrastructure.request_identity import bind_conversation, overlay_identity

StreamCallback = Callable[[dict[str, Any]], Awaitable[None]]

_STUB_FINISH_REASONS = {
    STUB_TOO_LONG: "stub_too_long",
    STUB_UPLOAD_AND_ASK: "stub_upload_and_ask",
    STUB_UPLOAD_REQUIRED: "stub_upload_required",
    STUB_FILE_LOAD_FAILED: "stub_file_load_failed",
    STUB_ANONYMIZATION_PENDING: "stub_anonymization_pending",
}


def _extract_answer_from_final(final: dict[str, Any]) -> str:
    data = final.get("data")
    if not isinstance(data, dict):
        return ""
    for fragment in reversed(data.get("fragments") or []):
        if fragment.get("fragment_type") == "response":
            return str(fragment.get("content") or "")
    return ""


def _resolve_finish_reason(
    *,
    handler_finish: str,
    final: dict[str, Any],
    stub_key: str | None,
) -> str:
    if stub_key is not None:
        return _STUB_FINISH_REASONS.get(stub_key, "stub")
    if handler_finish == "error":
        return "error"
    data = final.get("data")
    if isinstance(data, dict) and data.get("status") in {"success", "error"}:
        return str(data["status"])
    return "unknown"


@mcp_exception_handler(
    log_event="chat_w_doc_mcp_stream_error",
    to_terminal_packet=lambda error, _service, _document_app, ctx, **_: build_error_terminal_packet(
        message_id=ctx["message_id"],
        created_at=ctx["created_at"],
        fragments=ctx["fragments"],
        error=error,
    ),
)
async def _stream_chat_w_doc(
    service: DocumentService,
    document_app: Any,
    ctx: dict[str, Any],
    *,
    on_event: StreamCallback | None = None,
) -> dict[str, Any]:
    """Happy path: prepare + graph fragments + terminal; ошибки — в декораторе."""
    result = await service.execute()
    if isinstance(result, str):
        ctx["stub_key"] = result
        stub = build_stub_message(service.message_id, result)
        packet = {"data": stub.model_dump(mode="json")}
        if on_event is not None:
            await on_event(packet)
        return packet

    state = result
    final_message: Message | None = None
    try:
        async for event in astream_events(document_app, state):
            if isinstance(event, MessageFragment):
                ctx["fragments"].append(event.model_dump(mode="json"))
            elif isinstance(event, Message):
                ctx["fragments"] = [fragment.model_dump(mode="json") for fragment in event.fragments]
                final_message = event
                if on_event is not None and event.status in {"success", "error"}:
                    await on_event({"data": event.model_dump(mode="json")})
                continue
            if on_event is not None:
                await on_event(event.model_dump(mode="json"))
    finally:
        docs = state.get("attached_documents") or []
        document_local_store.release_attached_local_docs([doc.local_doc_id for doc in docs])

    if final_message is not None and final_message.status in {"success", "error"}:
        return {"data": final_message.model_dump(mode="json")}

    raise RuntimeError("graph stream ended without terminal message")


async def run_chat_w_doc_stream(
    *,
    thread_id: str,
    history_messages: list[HistoryMessage],
    message_id: str | None = None,
    file_ids: list[str] | None = None,
    user_id: str | None = None,
    intent: str | None = None,
    on_event: StreamCallback | None = None,
) -> dict[str, Any]:
    """Invoke document_process with streaming; push chunks via on_event (MCP logs)."""
    thread_id_str = str(thread_id)
    message_id_uuid = UUID(str(message_id)) if message_id else uuid4()
    message_id_str = str(message_id_uuid)
    bind_conversation(chat_id=thread_id_str, message_id=message_id_str)
    overlay_identity(user_id=user_id)

    if file_ids and history_messages:
        history_messages[-1].file_ids = [UUID(str(fid)) for fid in file_ids]

    document_app = get_document_app()
    service = DocumentService(
        document_app,
        history_messages,
        thread_id_str,
        message_id_uuid,
        extract_document_from_s3,
        intent=intent,
    )
    ctx: dict[str, Any] = {
        "message_id": message_id_str,
        "created_at": time.time(),
        "fragments": [],
    }

    final, handler_finish = await _stream_chat_w_doc(
        service,
        document_app,
        ctx,
        on_event=on_event,
    )
    stub_key = ctx.get("stub_key")
    finish_reason = _resolve_finish_reason(
        handler_finish=handler_finish,
        final=final,
        stub_key=stub_key,
    )

    return {
        "answer": stub_key if stub_key is not None else _extract_answer_from_final(final),
        "thread_id": thread_id_str,
        "message_id": message_id_str,
        "finish_reason": finish_reason,
    }
