"""Shared node helpers for document-process graph."""

from __future__ import annotations

import time
from collections.abc import Iterator
from itertools import count
from threading import Lock
from typing import Any

from langgraph.config import get_stream_writer

from src.agents.document_process.state import AgentState
from src.common.schemas.models import Message, MessageFragment


class OrderingIdGenerator:
    """Per-chat fragment id counters. Entries are removed on reset_for to avoid leaks."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, Iterator[int]] = {}

    def next_for(self, chat_id: str) -> int:
        with self._lock:
            counter = self._counters.get(chat_id)
            if counter is None:
                counter = count(1)
                self._counters[chat_id] = counter
            return next(counter)

    def reset_for(self, chat_id: str) -> None:
        with self._lock:
            self._counters.pop(chat_id, None)


ordering_ids = OrderingIdGenerator()


def _chat_id_from_state(state: AgentState) -> str:
    if "chat_id" in state:
        return state["chat_id"]
    meta = state.get("meta") or {}
    cfg = meta.get("configurable") or {}
    return cfg.get("thread_id", "default")


def _next_oid(state: AgentState) -> int:
    return ordering_ids.next_for(_chat_id_from_state(state))


class BaseNode:
    step: str

    def __init__(self, step: str):
        self.step = step

    def __call__(self, state: AgentState) -> Any:
        raise NotImplementedError


def with_attachment_warnings(state: AgentState, body: str) -> str:
    """Дописать к телу ответа сообщение о непрочитанных вложениях.

    Пользователь должен видеть, что часть приложенного не учтена: молчаливый
    пропуск неотличим от «прочитал и не нашёл».
    """
    warnings = state.get("attachment_warnings") or []
    if not warnings:
        return body
    notice = "\n".join(f"⚠️ {warning}" for warning in warnings)
    return f"{body}\n\n{notice}" if body else notice


def _upsert_fragment(message: Message, fragment: MessageFragment) -> None:
    for idx, existing in enumerate(message.fragments):
        if existing.fragment_id == fragment.fragment_id:
            message.fragments[idx] = fragment
            return
    message.fragments.append(fragment)


def _emit_ui_event(
    state: AgentState,
    step: str,
    fragment_id: int,
    *,
    fragment_type: str = "think",
    status: str,
    content: str,
    streaming: bool = False,
    token_usage: int | None = None,
    t0: float | None = None,
    duration_s: float | None = None,
) -> MessageFragment:
    if duration_s is None and t0 and status != "in_progress":
        duration_s = time.perf_counter() - t0
    frag = MessageFragment(
        fragment_id=fragment_id,
        fragment_type=fragment_type,
        status=status,
        streaming=streaming,
        content=content,
        token_usage=token_usage,
        duration_s=duration_s,
    )
    _upsert_fragment(state["output_message"], frag)
    get_stream_writer()({step: {"events": [frag]}})
    return frag
