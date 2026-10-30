"""prepare_context node."""

from __future__ import annotations

import datetime
import time
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from src.agents.document_process.models import RelevantDocs
from src.agents.document_process.nodes.base import BaseNode, _emit_ui_event, _next_oid
from src.agents.document_process.prompts import DocumentValidationOut
from src.agents.document_process.state import AgentState
from src.agents.document_process.text_utils import sanitize_md_links
from src.agents.llm.invoke import (
    _collect_token_usage,
    call_llm_structured_async_with_timeout,
    normalize_query,
)
from src.common.schemas.models import HistoryMessage, Message, MessageFragment
from src.infrastructure.langfuse.integration import aspan
from src.infrastructure.langfuse.prompt_kit import promptstore
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class PrepareContextDocNode(BaseNode):
    def __init__(
        self,
        validation_llm: Any,
        *,
        history_limit: int,
        validation_llm_timeout_s: float | None = None,
    ):
        super().__init__("prepare_context")
        self.validation_llm = validation_llm
        self.history_limit = history_limit
        self.validation_llm_timeout_s = validation_llm_timeout_s

    @aspan(as_type="generation", name="doc_prepare_context", prompt_name="validation")
    async def __call__(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        ordering_id = _next_oid(state)

        progress_msg = "### Анализирую запрос..."

        full = Message(
            message_id=state["message_id"],
            parent_id=state["input_message"].message_id,
            role="assistant",
            status="in_progress",
            fragments=[
                MessageFragment(
                    fragment_id=ordering_id,
                    fragment_type="think",
                    status="in_progress",
                    content=progress_msg,
                ),
            ],
        )
        get_stream_writer()({self.step: {"events": [full]}})
        state["output_message"] = full

        msgs = self._build_messages_from_history(
            state.get("income_history_messages", []),
            limit=self.history_limit,
        )
        msgs.append(HumanMessage(content=state["input_message"].fragments[0].content))

        for msg in msgs:
            msg.content = sanitize_md_links(msg.content)

        state["relevant_docs"] = RelevantDocs()
        state["messages"] = msgs

        user_text = state["input_message"].fragments[0].content or ""
        normalized_user_query = normalize_query(user_text)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        schema = DocumentValidationOut.model_json_schema()
        prompt = promptstore["validation"]
        prompt.compile(date=current_date, schema=schema)
        messages = prompt.format("langchain")
        messages.extend(msgs[-self.history_limit :])

        validation: DocumentValidationOut | None = None
        validation_failed = False
        token_usage = None
        enriched_query = normalized_user_query
        if normalized_user_query:
            try:
                with get_usage_metadata_callback() as cb:
                    validation = await call_llm_structured_async_with_timeout(
                        self.validation_llm,
                        messages,
                        DocumentValidationOut,
                        timeout_s=self.validation_llm_timeout_s,
                    )
                    token_usage = _collect_token_usage(cb)
            except Exception as exc:
                await logger.aerror(
                    "Document validation failed",
                    error_class=type(exc).__name__,
                    detail=str(exc),
                    exc_info=True,
                )
                validation = None
                validation_failed = True
        await logger.adebug(validation)

        if validation:
            enriched_query = f"{validation.enriched_query} [{normalized_user_query}]"

        is_valid, request_error_message = self._validation_result(
            normalized_user_query,
            validation,
            validation_failed=validation_failed,
        )

        state["normalized_user_query"] = normalized_user_query
        state["enriched_user_query"] = enriched_query
        state["is_valid_request"] = is_valid
        state["request_error_message"] = request_error_message

        status = "success" if is_valid else "error"
        result_msg = "### Запрос проанализирован\n\n---" if is_valid else "### Запрос отклонён"

        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status=status,
            content=result_msg,
            token_usage=token_usage,
            t0=t0,
        )
        return state

    @staticmethod
    def _validation_result(
        normalized_query: str,
        validation: DocumentValidationOut | None,
        *,
        validation_failed: bool,
    ) -> tuple[bool, str]:
        if not normalized_query:
            return False, "Запрос пуст. Введите вопрос и попробуйте ещё раз."
        if validation_failed or validation is None:
            return (
                False,
                "Не удалось проверить безопасность запроса. Попробуйте отправить его ещё раз.",
            )
        if validation is not None and validation.is_jailbreak:
            return False, "Запрос отклонён проверкой безопасности."
        return True, ""

    def _build_messages_from_history(
        self,
        history_messages: list[HistoryMessage],
        limit: int = 100,
    ) -> list[HumanMessage | AIMessage]:
        msgs: list[HumanMessage | AIMessage] = []
        for hm in history_messages:
            if not hm.fragments:
                continue
            if hm.role == "user":
                text = "\n".join(f.content for f in hm.fragments if f.fragment_type in ("request",) and f.content).strip()
                if text:
                    msgs.append(HumanMessage(content=text))
            elif hm.role == "assistant":
                response_text = "\n".join(
                    f.content for f in hm.fragments if f.fragment_type in ("response",) and f.content
                ).strip()
                if response_text:
                    msgs.append(AIMessage(content=response_text))
        return msgs[-limit:]
