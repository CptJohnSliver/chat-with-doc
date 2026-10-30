from __future__ import annotations

import time
import uuid
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class MessageFragment(BaseModel):
    fragment_id: int
    fragment_type: Literal["request", "think", "response"]
    status: Literal["error", "success", "in_progress"]
    streaming: bool = False
    content: str | None = None
    token_usage: int | None = None  # filled on completion
    duration_s: float | None = None  # filled on completion


class Message(BaseModel):
    message_id: UUID = Field(default_factory=uuid.uuid4)
    parent_id: UUID | None  # parent message_id

    role: Literal["user", "assistant"]
    status: Literal["error", "success", "in_progress"]

    fragments: list[MessageFragment] = Field(default_factory=list)

    created_at: float = Field(default_factory=time.time)
    updated_at: float | None = None

    thinking_duration_s: float | None = None  # filled on completion
    accumulated_token_usage: int | None = None  # filled on completion


class HistoryMessageFragment(BaseModel):
    fragment_type: Literal["request", "think", "response"]
    content: str | None = Field(default=None, max_length=200_000)


class HistoryMessage(BaseModel):
    """Dialog history item for MCP / orchestrator invoke."""

    role: Literal["user", "assistant"]
    fragments: list[HistoryMessageFragment] = Field(
        default_factory=list,
        examples=[[{"fragment_type": "request", "content": "Проанализируй этот документ"}]],
    )
    file_ids: list[UUID] = Field(
        default_factory=list,
        examples=[["123e4567-e89b-12d3-a456-426614174000"]],
    )
