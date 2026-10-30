"""Document-process LangGraph wiring."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.document_process.nodes import (
    DocumentQAAnswerNode,
    DocumentRetrieveNode,
    DocumentSummarizeNode,
    IntentClassifierNode,
    PrepareContextDocNode,
)
from src.agents.document_process.state import AgentState
from src.common.schemas.models import Message, MessageFragment

# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────


def route_from_intent(state: AgentState) -> str:
    """Route based on document_intent classification."""
    intent = state.get("document_intent", "document_qa")
    if intent == "summarize":
        return "summarize"
    return "document_qa"


def route_from_prepare_context(state: AgentState) -> str:
    """After prepare_context, go to intent_classifier if request is valid."""
    if not state.get("is_valid_request"):
        return "invalid"
    return "classify"


def _cfg_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return default if value is None else int(value)


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_document_workflow(
    *,
    validation_llm: Any,
    config: Mapping[str, Any],
    classifier_llm: Any | None = None,
    summarize_llm: Any | None = None,
    summarize_map_llm: Any | None = None,
    answer_llm: Any | None = None,
    validation_llm_timeout_s: float | None = None,
    classifier_llm_timeout_s: float | None = None,
    summarize_llm_timeout_s: float | None = None,
    summarize_map_llm_timeout_s: float | None = None,
    answer_llm_timeout_s: float | None = None,
):
    classifier_llm = classifier_llm or validation_llm
    summarize_llm = summarize_llm or answer_llm or validation_llm
    summarize_map_llm = summarize_map_llm or summarize_llm
    answer_llm = answer_llm or validation_llm

    g = StateGraph(AgentState)

    g.add_node(
        "prepare_context",
        PrepareContextDocNode(
            validation_llm,
            history_limit=_cfg_int(config, "history_limit_messages", 10),
            validation_llm_timeout_s=validation_llm_timeout_s,
        ),
    )
    g.add_node(
        "intent_classifier",
        IntentClassifierNode(
            classifier_llm,
            classifier_llm_timeout_s=classifier_llm_timeout_s,
            snippet_chars=_cfg_int(config, "snippet_chars", 8000),
        ),
    )
    g.add_node(
        "document_summarize",
        DocumentSummarizeNode(
            summarize_llm,
            map_llm=summarize_map_llm,
            summarize_llm_timeout_s=summarize_llm_timeout_s,
            map_llm_timeout_s=summarize_map_llm_timeout_s,
            map_reduce_threshold=_cfg_int(config, "summarize_map_reduce_threshold", 150000),
            map_max_chunks=_cfg_int(config, "summarize_map_max_chunks", 10),
            map_concurrency=_cfg_int(config, "summarize_map_concurrency", 3),
            map_segment_max=_cfg_int(config, "summarize_map_segment_max", 100000),
            map_max_segments=_cfg_int(config, "summarize_map_max_segments", 250),
            max_document_chars=_cfg_int(config, "summarize_max_document_chars", 10_000_000),
        ),
    )
    retrieve_core_top_k = _cfg_int(config, "retrieve_core_top_k", 20)
    g.add_node(
        "document_retrieve",
        DocumentRetrieveNode(
            segment_max_size=_cfg_int(config, "segment_max_size", 5000),
            top_k=_cfg_int(config, "retrieve_top_k", 30),
            core_top_k=retrieve_core_top_k,
            chunk_threshold=_cfg_int(config, "chunk_threshold", 150000),
            full_embed_max_chars=_cfg_int(config, "retrieve_full_embed_max_chars", 1_000_000),
            keyword_candidate_limit=_cfg_int(config, "retrieve_keyword_candidates", 500),
            embed_batch_size=_cfg_int(config, "retrieve_embed_batch_size", 64),
            fallback_chars=_cfg_int(config, "retrieve_fallback_chars", 50000),
        ),
    )
    g.add_node(
        "document_qa_answer",
        DocumentQAAnswerNode(
            answer_llm,
            answer_llm_timeout_s=answer_llm_timeout_s,
        ),
    )

    g.add_edge(START, "prepare_context")

    g.add_conditional_edges(
        "prepare_context",
        route_from_prepare_context,
        {
            "invalid": "document_qa_answer",
            "classify": "intent_classifier",
        },
    )

    g.add_conditional_edges(
        "intent_classifier",
        route_from_intent,
        {
            "summarize": "document_summarize",
            "document_qa": "document_retrieve",
        },
    )

    g.add_edge("document_summarize", END)
    g.add_edge("document_retrieve", "document_qa_answer")
    g.add_edge("document_qa_answer", END)

    return g


async def astream_events(app, initial_state: AgentState) -> AsyncGenerator[MessageFragment | Message, Any]:
    """Async generator that yields MessageFragment / Message events from the graph."""
    async for update in app.astream(initial_state, stream_mode=["custom"]):
        _mode, payload = update
        for _node, delta in payload.items():
            if full_message := delta.get("output_message"):
                yield full_message
            else:
                evs = delta.get("events")
                if isinstance(evs, list):
                    for ev in evs:
                        yield ev
