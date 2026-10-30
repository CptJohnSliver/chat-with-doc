"""FastMCP server exposing chat_w_doc skill and invoke tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from pydantic import BaseModel, Field

from src.app.mcp.runner import run_chat_w_doc_stream
from src.common.schemas.models import HistoryMessage
from src.infrastructure.request_identity import bind_identity, identity_from_headers

SKILLS_ROOT = Path(__file__).resolve().parents[3] / "skills"

mcp = FastMCP(
    name="chat_w_doc",
    providers=[SkillsDirectoryProvider(roots=SKILLS_ROOT)],
)


class ChatWDocInvokeRequest(BaseModel):
    thread_id: str = Field(description="Идентификатор диалога")
    history_messages: list[HistoryMessage] = Field(
        description="История сообщений; последнее — текущий вопрос пользователя",
    )
    message_id: str | None = Field(
        default=None,
        description="Идентификатор сообщения (опционально)",
    )
    file_ids: list[str] | None = Field(
        default=None,
        description="ID файлов текущего сообщения (S3 keys)",
    )
    user_id: str | None = Field(
        default=None,
        description="Идентификатор пользователя (опционально; по умолчанию = thread_id)",
    )
    intent: Literal["summarize", "qa"] | None = Field(
        default=None,
        description=(
            "Что нужно сделать с документом: 'summarize' — краткое содержание, "
            "'qa' — ответ на вопрос по тексту. Передавай явно, когда знаешь: "
            "документ приложен без пояснений — 'summarize'; спрашивают про пункт, "
            "сумму или срок — 'qa'. Без параметра навык определит сам по тексту запроса."
        ),
    )


@mcp.tool()
async def chat_w_doc_invoke(data: ChatWDocInvokeRequest, ctx: Context) -> dict[str, Any]:
    """Работа с документами (chat_w_doc).

    Саммари и Q&A по загруженным документам. Стримит промежуточные события
    через MCP notifications, затем возвращает итоговый ответ.
    """
    try:
        UUID(data.thread_id)
    except ValueError as exc:
        raise ValueError("thread_id must be a valid UUID string") from exc

    # Streamable HTTP runs the tool in a long-lived session task. Parent
    # middleware ContextVar from this POST does not copy over — read headers
    # from FastMCP's request_ctx (the current tools/call), then runner fills
    # chat/thread from the old body contract if those headers are absent.
    try:
        mcp_http_request = get_http_request()
    except RuntimeError:
        pass
    else:
        bind_identity(identity_from_headers(mcp_http_request.headers))

    async def on_event(payload: dict[str, Any]) -> None:
        await ctx.info(
            json.dumps(payload, ensure_ascii=False, default=str),
            logger_name="chat_w_doc.stream",
        )

    return await run_chat_w_doc_stream(
        thread_id=data.thread_id,
        history_messages=data.history_messages,
        message_id=data.message_id,
        file_ids=data.file_ids,
        user_id=data.user_id,
        intent=data.intent,
        on_event=on_event,
    )


mcp_app = mcp.http_app(path="/", transport="http")
