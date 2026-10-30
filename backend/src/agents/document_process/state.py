"""LangGraph state for the document-process agent."""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    TypedDict,
)
from uuid import UUID

from langchain_core.messages import BaseMessage

from src.common.schemas.models import HistoryMessage, Message

from .models import AttachedDocument, RelevantDocs, merge_relevant_docs


# TODO-REFACTOR total=False переработать
class AgentState(TypedDict, total=False):
    # infra
    meta: dict[str, Any]
    chat_id: str

    # io
    input_message: Message
    output_message: Message

    # history
    income_history_messages: list[HistoryMessage]
    messages: list[BaseMessage]
    relevant_docs: Annotated[RelevantDocs, merge_relevant_docs]

    # workflow flags
    normalized_user_query: str
    enriched_user_query: str
    is_valid_request: bool
    request_error_message: str
    message_id: UUID

    # document_process fields
    document_intent: str
    document_intent_hint: str
    intent_classification_failed: bool
    attached_documents: list[AttachedDocument]
    # Файлы, которые пользователь приложил, но которые не доехали до ответа.
    # Дописываются к телу ответа: молчаливый пропуск неотличим от «не нашёл».
    attachment_warnings: list[str]
