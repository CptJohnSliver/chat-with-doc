"""Lazy, cached LLM clients for document-process roles."""

from __future__ import annotations

from typing import Any

from src.agents.llm.factory import build_llm_from_config

_CACHE: dict[str, Any] = {}

_ROLE_NAMES = {
    "validation_llm": ("validation", "document_validation"),
    "classifier_llm": ("classifier", "document_classifier"),
    "answer_llm": ("answer", "document_answer"),
    "summarize_llm": ("summarize", "document_summarize"),
    "summarize_map_llm": ("summarize_map", "document_summarize_map"),
}


def get_cached_llm(attr_name: str) -> Any:
    cached = _CACHE.get(attr_name)
    if cached is not None:
        return cached
    role, name = _ROLE_NAMES[attr_name]
    llm = build_llm_from_config(role, name=name)
    _CACHE[attr_name] = llm
    return llm
