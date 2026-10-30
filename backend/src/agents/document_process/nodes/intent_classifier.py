"""intent_classifier node."""

from __future__ import annotations

import datetime
import time
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from src.agents.document_process.document_utils import (
    aresolve_document_snippet,
    get_attached_documents,
)
from src.agents.document_process.nodes.base import BaseNode, _emit_ui_event, _next_oid
from src.agents.document_process.prompts import IntentClassificationOut
from src.agents.document_process.state import AgentState
from src.agents.llm.invoke import (
    _collect_token_usage,
    call_llm_structured_async_with_timeout,
)
from src.infrastructure.langfuse.integration import aspan
from src.infrastructure.langfuse.prompt_kit import promptstore
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_SUMMARY_INTENT_MARKERS = (
    "суммар",
    "резюм",
    "выжимк",
    "кратко излож",
    "краткое содерж",
    "основные тезис",
    "о чем документ",
    "о чём документ",
    "summar",
    "tl;dr",
)

_INTENT_LABELS = {
    "summarize": "Суммаризация документа",
    "document_qa": "Ответ по документу",
}


def _fallback_document_intent(question: str) -> tuple[str, bool]:
    """Infer only high-confidence summary requests; otherwise route to document QA."""
    normalized = " ".join(question.casefold().split())
    if any(marker in normalized for marker in _SUMMARY_INTENT_MARKERS):
        return "summarize", True
    return "document_qa", False


async def _build_doc_snippet(docs: list, snippet_chars: int) -> str:
    if not docs:
        return ""
    if len(docs) == 1:
        raw_snippet = await aresolve_document_snippet(docs[0], snippet_chars)
        return f"\nНачало документа (первые ~{snippet_chars} символов):\n---\n{raw_snippet}\n---\n"
    per_doc_chars = max(200, snippet_chars // len(docs))
    snippets: list[str] = []
    for doc in docs:
        raw_snippet = await aresolve_document_snippet(doc, per_doc_chars)
        if not raw_snippet:
            continue
        snippets.append(
            "\n".join(
                [
                    f"Документ: {doc.display_name}",
                    "---",
                    raw_snippet,
                    "---",
                ]
            )
        )
    doc_snippet = "\n\n".join(snippets)
    return f"\nНачало документов:\n{doc_snippet}\n" if doc_snippet else ""


def _intent_event_content(
    *,
    intent: str,
    classification_failed: bool,
    fallback_is_confident: bool,
) -> tuple[str, str]:
    label = _INTENT_LABELS.get(intent, intent)
    if classification_failed and not fallback_is_confident:
        return (
            "error",
            "### Не удалось точно определить тип запроса\nПродолжаю в режиме ответа по документу.\n\n---",
        )
    if classification_failed:
        return (
            "success",
            f"### Тип запроса: {label} (определён резервным правилом)\n\n---",
        )
    return "success", f"### Тип запроса: {label}\n\n---"


class IntentClassifierNode(BaseNode):
    def __init__(
        self,
        classifier_llm: Any,
        *,
        classifier_llm_timeout_s: float | None = None,
        snippet_chars: int,
    ):
        super().__init__("intent_classifier")
        self.classifier_llm = classifier_llm
        self.classifier_llm_timeout_s = classifier_llm_timeout_s
        self.snippet_chars = snippet_chars

    @aspan(
        as_type="generation",
        name="intent_classifier",
        prompt_name="intent_classification",
    )
    async def __call__(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        ordering_id = _next_oid(state)
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content="### Определяю тип запроса...",
        )

        docs = get_attached_documents(state)
        question = state.get("enriched_user_query") or state.get("normalized_user_query") or ""

        intent_hint = state.get("document_intent_hint")
        if intent_hint in {"summarize", "document_qa"}:
            state["document_intent"] = intent_hint
            state["intent_classification_failed"] = False
            _emit_ui_event(
                state,
                self.step,
                ordering_id,
                status="success",
                content=f"### Тип запроса: {_INTENT_LABELS[intent_hint]}\n\n---",
                t0=t0,
            )
            return state

        doc_snippet = await _build_doc_snippet(docs, self.snippet_chars)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = promptstore["intent_classification"]
        prompt.compile(
            date=current_date,
            schema=IntentClassificationOut.model_json_schema(),
            question=question,
            has_document=str(bool(docs)).lower(),
            doc_snippet=doc_snippet,
        )
        messages = prompt.format("langchain")

        token_usage = None
        intent, fallback_is_confident = _fallback_document_intent(question)
        classification_failed = False
        try:
            with get_usage_metadata_callback() as cb:
                classification = await call_llm_structured_async_with_timeout(
                    self.classifier_llm,
                    messages,
                    IntentClassificationOut,
                    timeout_s=self.classifier_llm_timeout_s,
                )
                token_usage = _collect_token_usage(cb)
            intent = classification.intent
            await logger.adebug(classification)
        except Exception as exc:
            classification_failed = True
            await logger.aerror(
                "Intent classification failed:",
                error_class=type(exc).__name__,
                detail=str(exc),
                exc_info=True,
            )

        state["document_intent"] = intent
        state["intent_classification_failed"] = classification_failed
        event_status, event_content = _intent_event_content(
            intent=intent,
            classification_failed=classification_failed,
            fallback_is_confident=fallback_is_confident,
        )
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status=event_status,
            content=event_content,
            token_usage=token_usage,
            t0=t0,
        )
        return state
