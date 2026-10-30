"""LLM clients and call helpers for document-process agents.

Clients are created lazily on first attribute access so importing this package
does not require ROUTER_AI_API_KEY at collection time.

Layout:
  config.py   — role settings from api.toml
  http.py     — Router AI HTTP clients
  factory.py  — build ChatOpenAI (gateway models + app-side LocalFallback)
  clients.py  — cached validation / classifier / answer / summarize / summarize_map
  fallback.py — LocalFallbackRunnable (schema / mid-stream; not transport)
  invoke.py   — timeouts, retries, structured/stream calls

Per-role settings: ``[default.document_process_llms.<role>]`` in api.toml.
"""

from __future__ import annotations

from typing import Any

from src.agents.llm.clients import get_cached_llm
from src.agents.llm.config import get_llm_timeout

__all__ = [
    "answer_llm",
    "classifier_llm",
    "get_llm_timeout",
    "summarize_llm",
    "summarize_map_llm",
    "validation_llm",
]


def __getattr__(name: str) -> Any:
    if name in {
        "validation_llm",
        "classifier_llm",
        "answer_llm",
        "summarize_llm",
        "summarize_map_llm",
    }:
        return get_cached_llm(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
