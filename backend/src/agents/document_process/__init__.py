"""Document-process agent package.

Importing this package (or submodules like ``models``) must stay side-effect free:
no LLM clients, no compiled graph. Use ``get_document_app()`` only when the runtime
graph is needed.

Runtime knobs: ``settings.document_process`` (``[default.document_process]`` in api.toml).
"""

from __future__ import annotations

from typing import Any

from src.agents.document_process.graph import build_document_workflow
from src.common.config.config import settings

__all__ = [
    "build_document_workflow",
    "get_document_app",
]


class _DocumentAppState:
    instance: Any | None = None


def get_document_app() -> Any:
    """Build and cache the compiled LangGraph app (imports LLM clients on first call)."""
    if _DocumentAppState.instance is not None:
        return _DocumentAppState.instance

    from src.agents.llm import (  # noqa: PLC0415
        answer_llm,
        classifier_llm,
        get_llm_timeout,
        summarize_llm,
        summarize_map_llm,
        validation_llm,
    )

    graph = build_document_workflow(
        validation_llm=validation_llm,
        classifier_llm=classifier_llm,
        summarize_llm=summarize_llm,
        summarize_map_llm=summarize_map_llm,
        answer_llm=answer_llm,
        validation_llm_timeout_s=get_llm_timeout("validation"),
        classifier_llm_timeout_s=get_llm_timeout("classifier"),
        summarize_llm_timeout_s=get_llm_timeout("summarize"),
        summarize_map_llm_timeout_s=get_llm_timeout("summarize_map"),
        answer_llm_timeout_s=get_llm_timeout("answer"),
        config=settings.document_process,
    )
    _DocumentAppState.instance = graph.compile()
    return _DocumentAppState.instance
