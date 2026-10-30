"""document_qa_answer node."""

from __future__ import annotations

import datetime
import time
from typing import Any, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.config import get_stream_writer

from src.agents.document_process.document_utils import build_document_context
from src.agents.document_process.nodes.base import (
    BaseNode,
    _chat_id_from_state,
    _emit_ui_event,
    _next_oid,
    ordering_ids,
    with_attachment_warnings,
)
from src.agents.document_process.state import AgentState
from src.agents.llm.invoke import (
    _collect_token_usage_parts,
    astream_with_retries,
)
from src.infrastructure.langfuse.integration import aspan
from src.infrastructure.langfuse.prompt_kit import promptstore
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class DocumentQAAnswerNode(BaseNode):
    def __init__(self, answer_llm: Any, *, answer_llm_timeout_s: float | None = None):
        super().__init__("document_qa_answer")
        self.answer_llm = answer_llm
        self.answer_llm_timeout_s = answer_llm_timeout_s

    @aspan(as_type="generation", name="document_qa_answer", prompt_name="answer_document")
    async def __call__(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        ordering_id = _next_oid(state)
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content="### Формирую ответ по документу...",
        )

        relevant_docs = state["relevant_docs"]
        doc_chunks = relevant_docs.document
        question = state.get("enriched_user_query") or state.get("normalized_user_query") or ""
        response_fragment_id = _next_oid(state)

        token_usage = None
        token_parts: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        failed = False
        if not state.get("is_valid_request", False):
            body = state.get("request_error_message") or (
                "Запрос не прошёл проверку безопасности. Попробуйте сформулировать его иначе."
            )
            failed = True
        elif not doc_chunks:
            body = "Информация по этому вопросу в документе не найдена."
        else:
            context = build_document_context(doc_chunks)

            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            prompt = promptstore["answer_document"]
            prompt.compile(
                question=question,
                context=context,
                date=current_date,
            )
            messages = prompt.format("langchain")
            history_msgs = state["messages"][-3:-1]
            messages = [messages[0], *history_msgs, messages[1]]

            try:
                with get_usage_metadata_callback() as cb:
                    body = ""
                    _emit_ui_event(
                        state,
                        self.step,
                        response_fragment_id,
                        fragment_type="response",
                        status="in_progress",
                        content=body,
                        streaming=True,
                    )
                    async for partial in astream_with_retries(
                        self.answer_llm,
                        messages,
                        idle_timeout_s=self.answer_llm_timeout_s,
                        accumulate_n=2,
                        stream_mode="delta",
                    ):
                        if not partial:
                            body = ""
                        body += partial if partial else ""
                        _emit_ui_event(
                            state,
                            self.step,
                            response_fragment_id,
                            fragment_type="response",
                            status="in_progress",
                            content=partial if partial else "",
                            streaming=True,
                        )
                    token_parts = _collect_token_usage_parts(cb)
                    token_usage = token_parts["total_tokens"]
            except Exception as exc:
                await logger.aerror(
                    "Document Q&A answer failed",
                    error_class=type(exc).__name__,
                    detail=str(exc),
                    exc_info=True,
                )
                body = "Я столкнулся с ошибкой при подготовке ответа, попробуйте задать вопрос ещё раз."
                failed = True

        # Don't decode \n escape sequences - keep them as literal \n for frontend rendering
        body = with_attachment_warnings(state, body.strip())
        await logger.adebug(body)
        status: Literal["error", "success"] = "error" if failed else "success"
        await logger.ainfo(
            "DocumentQA answer metrics",
            duration_s=round(time.perf_counter() - t0, 3),
            body_chars=len(body),
            token_usage=token_usage,
            prompt_tokens=token_parts["prompt_tokens"],
            completion_tokens=token_parts["completion_tokens"],
            failed=failed,
        )

        _emit_ui_event(
            state,
            self.step,
            response_fragment_id,
            fragment_type="response",
            status=status,
            content=body,
            token_usage=token_usage,
        )

        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status=status,
            content=("### Не удалось подготовить ответ" if failed else "### Ответ по документу подготовлен"),
            token_usage=None,
            t0=t0,
        )

        full = state["output_message"]
        full.status = status
        full.updated_at = time.time()
        full.accumulated_token_usage = sum((f.token_usage or 0) for f in full.fragments)
        full.thinking_duration_s = sum((f.duration_s or 0.0) for f in full.fragments if f.fragment_type == "think")

        get_stream_writer()({self.step: {"events": [full]}})

        ordering_ids.reset_for(_chat_id_from_state(state))

        return state
