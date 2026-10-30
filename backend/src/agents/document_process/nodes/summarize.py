"""document_summarize node."""

from __future__ import annotations

import asyncio
import datetime
import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.config import get_stream_writer
from tenacity import stop_after_attempt, wait_none

from src.agents.document_process.document_utils import (
    MapSegmentLimitExceededError,
    aresolve_document_snippet,
    aselect_map_segments,
    document_char_count,
    documents_in_focus,
)
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
    _collect_token_usage,
    _collect_token_usage_parts,
    ainvoke_with_timeout_retries,
    astream_with_retries,
)
from src.infrastructure.langfuse.integration import aspan
from src.infrastructure.langfuse.prompt_kit import promptstore
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Map/reduce chunks: one attempt per candidate — fail fast to fallback/next chunk.
_MAP_RETRY_KWARGS = {
    "stop": stop_after_attempt(1),
    "wait": wait_none(),
    "reraise": True,
}

_SENTENCE_END = tuple(".!?…»\"'|)]")
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "token_limit"})
# Prompt targets ~7–9.5k; soft provider wall ~10–12k. Below this floor,
# trailing non-sentence endings are usually fine list items.
_TRUNCATION_FALLBACK_MIN_CHARS = 9500


def _looks_truncated_summary(
    text: str,
    *,
    finish_reason: str | None = None,
) -> bool:
    """Detect a provider cutoff, with a conservative text fallback."""
    normalized_reason = (finish_reason or "").strip().lower()
    if normalized_reason in _TRUNCATED_FINISH_REASONS or normalized_reason.endswith(".length"):
        return True

    t = text.rstrip()
    if not t:
        return False
    if t.endswith(("…", "...")):
        return True
    if t.endswith(("ст.", "п.", "ч.")):
        return True
    if t.endswith(("(", "[", "{", "«", ",", ";", ":", "/", "\\", "—", "–")):
        return True
    # Soft provider cutoff historically ~10–12k (higher with reasoning off). Below
    # the fallback floor, a list item ending on a word/date is usually valid.
    if len(t) < _TRUNCATION_FALLBACK_MIN_CHARS:
        return False
    return t[-1] not in _SENTENCE_END


def _join_map_summaries_for_final(partial_summaries: list[str], *, budget_chars: int) -> str:
    """Join map/mid-reduce outputs; truncate evenly only if over the final input budget."""
    n = len(partial_summaries)
    if n == 0:
        return ""
    total = sum(len(s) for s in partial_summaries)
    if total <= budget_chars:
        return "\n\n---\n\n".join(f"### Фрагмент {idx}\n{summary}" for idx, summary in enumerate(partial_summaries, start=1))
    per = max(400, budget_chars // n)
    pieces: list[str] = []
    for idx, summary in enumerate(partial_summaries, start=1):
        text = summary.strip()
        if len(text) > per:
            text = text[: max(0, per - 1)].rstrip() + "…"
        pieces.append(f"### Фрагмент {idx}\n{text}")
    return "\n\n---\n\n".join(pieces)


@dataclass(frozen=True)
class _SummaryResult:
    body: str
    token_usage: int | None = None
    failed: bool = False


class DocumentSummarizeNode(BaseNode):
    def __init__(
        self,
        summarize_llm: Any,
        *,
        map_llm: Any | None = None,
        summarize_llm_timeout_s: float | None = None,
        map_llm_timeout_s: float | None = None,
        map_reduce_threshold: int,
        map_max_chunks: int,
        map_concurrency: int,
        map_segment_max: int,
        map_max_segments: int,
        max_document_chars: int,
    ):
        super().__init__("document_summarize")
        self.summarize_llm = summarize_llm
        # Map chunks use a cheaper model when provided; reduce/final stay on summarize_llm.
        self.map_llm = map_llm or summarize_llm
        self.summarize_llm_timeout_s = summarize_llm_timeout_s
        self.map_llm_timeout_s = map_llm_timeout_s if map_llm_timeout_s is not None else summarize_llm_timeout_s
        self.map_reduce_threshold = map_reduce_threshold
        # A reduce batch of one can never shrink the list and would loop forever.
        self.map_max_chunks = max(2, map_max_chunks)
        self.map_concurrency = max(1, map_concurrency)
        self.map_segment_max = map_segment_max
        self.map_max_segments = max(1, map_max_segments)
        self.max_document_chars = max_document_chars

    async def _stream_final_summary(
        self,
        state: AgentState,
        *,
        document_text: str,
        response_fragment_id: int,
    ) -> _SummaryResult:
        final_t0 = time.perf_counter()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = promptstore["summarize_document"]
        prompt.compile(date=current_date, document_text=document_text)
        messages = prompt.format("langchain")

        body = ""
        token_usage: int | None = None
        token_parts: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        failed = False
        finish_reason: str | None = None

        def _capture_finish_reason(reason: str | None) -> None:
            nonlocal finish_reason
            finish_reason = reason

        _emit_ui_event(
            state,
            self.step,
            response_fragment_id,
            fragment_type="response",
            status="in_progress",
            content="",
            streaming=True,
        )
        try:
            with get_usage_metadata_callback() as cb:
                async for partial in astream_with_retries(
                    self.summarize_llm,
                    messages,
                    idle_timeout_s=self.summarize_llm_timeout_s,
                    accumulate_n=2,
                    stream_mode="delta",
                    on_finish_reason=_capture_finish_reason,
                ):
                    if not partial:
                        # astream_with_retries uses an empty delta as an explicit
                        # restart marker after a failed streaming attempt.
                        body = ""
                    else:
                        body += partial
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
                "Summarization failed",
                error_class=type(exc).__name__,
                detail=str(exc),
                exc_info=True,
            )
            body = "Не удалось суммаризировать документ. Попробуйте ещё раз."
            failed = True

        looks_truncated = not failed and bool(body) and _looks_truncated_summary(body, finish_reason=finish_reason)
        await logger.ainfo(
            "DocumentSummarize final metrics",
            duration_s=round(time.perf_counter() - final_t0, 3),
            body_chars=len(body),
            token_usage=token_usage,
            prompt_tokens=token_parts["prompt_tokens"],
            completion_tokens=token_parts["completion_tokens"],
            finish_reason=finish_reason,
            looks_truncated=looks_truncated,
            appears_complete=not failed and bool(body) and not looks_truncated,
        )
        return _SummaryResult(
            body=body.strip(),
            token_usage=token_usage,
            failed=failed,
        )

    async def _map_one_chunk(
        self,
        *,
        segment: str,
        chunk_index: int,
        chunk_total: int,
        prompt_lock: asyncio.Lock,
    ) -> tuple[int, _SummaryResult]:
        async with prompt_lock:
            chunk_prompt = promptstore["summarize_document_chunk"]
            chunk_prompt.compile(
                document_text=segment,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            )
            chunk_messages = chunk_prompt.format("langchain")
        try:
            with get_usage_metadata_callback() as cb:
                # Leave total budget to invoke's default (timeout × 3) so a timed-out
                # primary still leaves wall-clock for LocalFallbackRunnable candidates.
                response = await ainvoke_with_timeout_retries(
                    self.map_llm,
                    chunk_messages,
                    timeout_s=self.map_llm_timeout_s,
                    retry_kwargs=_MAP_RETRY_KWARGS,
                )
                piece = getattr(response, "content", str(response))
                return chunk_index, _SummaryResult(
                    body=str(piece).strip(),
                    token_usage=_collect_token_usage(cb),
                )
        except Exception as exc:
            await logger.aerror(
                "Map summarization failed",
                chunk_index=chunk_index,
                error_class=type(exc).__name__,
                detail=str(exc),
                exc_info=True,
            )
            return chunk_index, _SummaryResult(body="", failed=True)

    async def _reduce_summaries_batch(
        self,
        *,
        partial_summaries: list[str],
        prompt_lock: asyncio.Lock,
    ) -> _SummaryResult:
        combined = "\n\n---\n\n".join(f"### Фрагмент {idx}\n{summary}" for idx, summary in enumerate(partial_summaries, start=1))
        async with prompt_lock:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            prompt = promptstore["summarize_document_reduce"]
            prompt.compile(date=current_date, document_text=combined)
            messages = prompt.format("langchain")
        try:
            with get_usage_metadata_callback() as cb:
                response = await ainvoke_with_timeout_retries(
                    self.summarize_llm,
                    messages,
                    timeout_s=self.summarize_llm_timeout_s,
                    retry_kwargs=_MAP_RETRY_KWARGS,
                )
                body = str(getattr(response, "content", response)).strip()
                return _SummaryResult(
                    body=body,
                    token_usage=_collect_token_usage(cb),
                )
        except Exception as exc:
            await logger.aerror(
                "Mid-level summarization reduce failed",
                error_class=type(exc).__name__,
                detail=str(exc),
                exc_info=True,
            )
            return _SummaryResult(
                body="",
                failed=True,
            )

    async def _reduce_summary_level(
        self,
        *,
        partial_summaries: list[str],
        prompt_lock: asyncio.Lock,
    ) -> list[_SummaryResult]:
        """Reduce independent batches concurrently while preserving their order."""
        batches = [
            partial_summaries[start : start + self.map_max_chunks]
            for start in range(0, len(partial_summaries), self.map_max_chunks)
        ]
        results: dict[int, _SummaryResult] = {}
        semaphore = asyncio.Semaphore(self.map_concurrency)

        async def _reduce_one(batch_index: int, batch: list[str]) -> None:
            if len(batch) == 1:
                results[batch_index] = _SummaryResult(body=batch[0])
                return
            async with semaphore:
                results[batch_index] = await self._reduce_summaries_batch(
                    partial_summaries=batch,
                    prompt_lock=prompt_lock,
                )

        async with asyncio.TaskGroup() as task_group:
            for batch_index, batch in enumerate(batches):
                task_group.create_task(_reduce_one(batch_index, batch))

        return [results[index] for index in range(len(batches))]

    async def _map_segments(
        self,
        state: AgentState,
        *,
        segments: list[str],
        ordering_id: int,
    ) -> tuple[list[str], int | None, bool]:
        """Map chunk summaries with bounded concurrency; keeps ≤ map_concurrency windows in flight."""
        map_t0 = time.perf_counter()
        total = len(segments)
        await logger.ainfo(
            "DocumentSummarize map start",
            segments=total,
            concurrency=self.map_concurrency,
            map_timeout_s=self.map_llm_timeout_s,
        )
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content=(f"### Суммаризирую документ...\nОбрабатываю {total} фрагментов (параллельно до {self.map_concurrency})"),
        )

        prompt_lock = asyncio.Lock()
        done = 0
        results: dict[int, _SummaryResult] = {}
        work_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        for index, segment in enumerate(segments, start=1):
            work_queue.put_nowait((index, segment))

        async def _worker() -> None:
            nonlocal done
            while True:
                try:
                    chunk_index, segment = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                chunk_t0 = time.perf_counter()
                result_index, result = await self._map_one_chunk(
                    segment=segment,
                    chunk_index=chunk_index,
                    chunk_total=total,
                    prompt_lock=prompt_lock,
                )
                results[result_index] = result
                done += 1
                await logger.ainfo(
                    "DocumentSummarize map chunk done",
                    chunk_index=result_index,
                    done=done,
                    total=total,
                    failed=result.failed,
                    body_chars=len(result.body),
                    token_usage=result.token_usage,
                    duration_s=round(time.perf_counter() - chunk_t0, 3),
                )
                _emit_ui_event(
                    state,
                    self.step,
                    ordering_id,
                    status="in_progress",
                    content=(f"### Суммаризирую документ...\nГотово фрагментов: {done}/{total}"),
                )

        # TaskGroup guarantees that cancellation or an unexpected worker error
        # cancels and joins every sibling before this node unwinds.
        async with asyncio.TaskGroup() as task_group:
            for _ in range(min(self.map_concurrency, total)):
                task_group.create_task(_worker())

        await logger.ainfo(
            "DocumentSummarize map complete",
            segments=total,
            duration_s=round(time.perf_counter() - map_t0, 3),
        )

        if total == 0:
            return [], None, False

        token_usage_total: int | None = None
        partial_summaries: list[str] = []
        failed = False
        for chunk_index in range(1, total + 1):
            result = results[chunk_index]
            partial_summaries.append(result.body)
            failed = failed or result.failed
            if result.token_usage is not None:
                token_usage_total = (token_usage_total or 0) + result.token_usage
        return partial_summaries, token_usage_total, failed

    def _empty_document_result(
        self,
        state: AgentState,
        *,
        response_fragment_id: int,
        ordering_id: int,
    ) -> _SummaryResult:
        body = "Документ не содержит текста для суммаризации."
        _emit_ui_event(
            state,
            self.step,
            response_fragment_id,
            fragment_type="response",
            status="in_progress",
            content=body,
            streaming=True,
        )
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content="### Документ пуст",
        )
        return _SummaryResult(body=body, failed=True)

    async def _summarize_document(
        self,
        state: AgentState,
        *,
        doc: Any,
        response_fragment_id: int,
        ordering_id: int,
    ) -> _SummaryResult:
        char_count = document_char_count(doc)
        await logger.ainfo(
            "DocumentSummarize start",
            char_count=char_count,
            map_reduce_threshold=self.map_reduce_threshold,
            map_segment_max=self.map_segment_max,
        )
        if char_count <= 0:
            return self._empty_document_result(
                state,
                response_fragment_id=response_fragment_id,
                ordering_id=ordering_id,
            )

        if char_count > self.max_document_chars:
            body = (
                "Файл слишком большой для суммаризации. "
                f"Допустимо не более {self.max_document_chars} символов "
                f"(в документе около {char_count}). "
                "Разделите документ на части или сократите его."
            )
            _emit_ui_event(
                state,
                self.step,
                response_fragment_id,
                fragment_type="response",
                status="in_progress",
                content=body,
                streaming=True,
            )
            _emit_ui_event(
                state,
                self.step,
                ordering_id,
                status="in_progress",
                content="### Документ слишком большой для суммаризации",
            )
            return _SummaryResult(body=body, failed=True)

        if char_count <= self.map_reduce_threshold:
            # Cap read to threshold — never pull an unexpectedly huge file here.
            return await self._stream_final_summary(
                state,
                document_text=await aresolve_document_snippet(doc, self.map_reduce_threshold),
                response_fragment_id=response_fragment_id,
            )

        # Full coverage for docs ≤ max_document_chars: windows of map_segment_max, ≤ map_max_segments.
        try:
            segments = await aselect_map_segments(
                doc,
                max_chunk_chars=self.map_segment_max,
                max_segments=self.map_max_segments,
                require_full_coverage=True,
            )
        except MapSegmentLimitExceededError as exc:
            await logger.aerror(
                "Full-coverage map segmentation exceeded its configured limit",
                total_segments=exc.total,
                max_segments=exc.limit,
                char_count=char_count,
            )
            return _SummaryResult(
                body=(
                    "Не удалось разбить документ на фрагменты без потери текста. "
                    "Попробуйте ещё раз или разделите документ на части."
                ),
                failed=True,
            )
        if not segments:
            return await self._stream_final_summary(
                state,
                document_text="",
                response_fragment_id=response_fragment_id,
            )
        if len(segments) == 1:
            return await self._stream_final_summary(
                state,
                document_text=segments[0],
                response_fragment_id=response_fragment_id,
            )

        partial_summaries, token_usage_total, failed = await self._map_segments(
            state,
            segments=segments,
            ordering_id=ordering_id,
        )
        if failed:
            return _SummaryResult(
                body=("Не удалось суммаризировать документ: ошибка при обработке одного из фрагментов. Попробуйте ещё раз."),
                token_usage=token_usage_total,
                failed=True,
            )
        if not partial_summaries:
            return await self._stream_final_summary(
                state,
                document_text="",
                response_fragment_id=response_fragment_id,
            )

        # Mid-reduce only when there are too many parts AND they exceed the final
        # input budget. Typical codes (~8×50k windows) skip mid-reduce so themes
        # are not dropped by an extra LLM compression pass.
        # Keep the join budget large enough for ~8–10 map outputs (~6–8k each).
        # A tight ~24k budget silently truncates each fragment and the final model
        # then claims late themes (МЗП, 23/35 ч, ЧАЭС, ст. 402) were "not in input".
        final_input_budget = max(self.map_reduce_threshold * 3 // 4, 90000)
        prompt_lock = asyncio.Lock()
        level = 1
        while len(partial_summaries) > self.map_max_chunks and sum(len(s) for s in partial_summaries) > final_input_budget:
            before_n = len(partial_summaries)
            before_chars = sum(len(s) for s in partial_summaries)
            await logger.ainfo(
                "DocumentSummarize mid-reduce",
                level=level,
                parts=before_n,
                chars=before_chars,
                batch_size=self.map_max_chunks,
            )
            _emit_ui_event(
                state,
                self.step,
                ordering_id,
                status="in_progress",
                content=(
                    f"### Суммаризирую документ...\nСжимаю промежуточные выжимки (уровень {level}, {len(partial_summaries)} шт.)"
                ),
            )
            next_level: list[str] = []
            level_results = await self._reduce_summary_level(
                partial_summaries=partial_summaries,
                prompt_lock=prompt_lock,
            )
            for mid_result in level_results:
                if mid_result.token_usage is not None:
                    token_usage_total = (token_usage_total or 0) + mid_result.token_usage
                if mid_result.failed:
                    return _SummaryResult(
                        body=("Не удалось обработать промежуточные выжимки документа. Попробуйте ещё раз."),
                        token_usage=token_usage_total,
                        failed=True,
                    )
                next_level.append(mid_result.body)
            if len(next_level) >= before_n and sum(len(s) for s in next_level) >= before_chars:
                await logger.awarning(
                    "DocumentSummarize mid-reduce did not shrink; stopping to avoid loop",
                    level=level,
                    parts=before_n,
                    next_parts=len(next_level),
                )
                partial_summaries = next_level
                break
            partial_summaries = next_level
            level += 1

        combined = _join_map_summaries_for_final(partial_summaries, budget_chars=final_input_budget)
        await logger.ainfo(
            "DocumentSummarize final pass",
            parts=len(partial_summaries),
            chars=len(combined),
            budget=final_input_budget,
        )
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content="### Суммаризирую документ...\nСобираю итоговую выжимку",
        )
        final_result = await self._stream_final_summary(
            state,
            document_text=combined,
            response_fragment_id=response_fragment_id,
        )
        if final_result.token_usage is not None:
            token_usage_total = (token_usage_total or 0) + final_result.token_usage
        return _SummaryResult(
            body=final_result.body,
            token_usage=token_usage_total,
            failed=failed or final_result.failed,
        )

    @aspan(
        as_type="generation",
        name="document_summarize",
        prompt_name="summarize_document",
    )
    async def __call__(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        ordering_id = _next_oid(state)

        docs = documents_in_focus(state)

        if docs:
            doc_names = ", ".join(d.display_name for d in docs)
            _emit_ui_event(
                state,
                self.step,
                ordering_id,
                status="in_progress",
                content=f"### Суммаризирую документ...\nДокумент: {doc_names}",
            )
        else:
            _emit_ui_event(
                state,
                self.step,
                ordering_id,
                status="error",
                content="### Документ не загружен",
                t0=t0,
            )
            full = state["output_message"]
            full.status = "error"
            full.updated_at = time.time()
            full.thinking_duration_s = sum(
                (fragment.duration_s or 0.0) for fragment in full.fragments if fragment.fragment_type == "think"
            )
            get_stream_writer()({self.step: {"events": [full]}})
            ordering_ids.reset_for(_chat_id_from_state(state))
            return state

        response_fragment_id = _next_oid(state)
        token_usage_total: int | None = None
        failed = False

        _emit_ui_event(
            state,
            self.step,
            response_fragment_id,
            fragment_type="response",
            status="in_progress",
            content="",
            streaming=True,
        )

        if len(docs) == 1:
            result = await self._summarize_document(
                state,
                doc=docs[0],
                response_fragment_id=response_fragment_id,
                ordering_id=ordering_id,
            )
            body = result.body
            token_usage_total = result.token_usage
            failed = result.failed
            await logger.adebug(body)
        else:
            sections: list[str] = []
            for doc in docs:
                result = await self._summarize_document(
                    state,
                    doc=doc,
                    response_fragment_id=response_fragment_id,
                    ordering_id=ordering_id,
                )
                section_body = result.body
                await logger.adebug(section_body)
                sections.append(f"### {doc.display_name}\n\n{section_body}")
                failed = failed or result.failed
                _emit_ui_event(
                    state,
                    self.step,
                    response_fragment_id,
                    fragment_type="response",
                    status="in_progress",
                    content="\n\n---\n\n".join(sections),
                )
                if result.token_usage is not None:
                    token_usage_total = (token_usage_total or 0) + result.token_usage
            body = "\n\n---\n\n".join(sections)

        status: Literal["error", "success"] = "error" if failed else "success"
        body = with_attachment_warnings(state, body)

        _emit_ui_event(
            state,
            self.step,
            response_fragment_id,
            fragment_type="response",
            status=status,
            content=body,
            token_usage=token_usage_total,
        )

        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status=status,
            content=("### Не удалось завершить суммаризацию" if failed else "### Суммаризация завершена"),
            token_usage=None,
            t0=t0,
        )

        full = state["output_message"]
        full.status = status
        full.updated_at = time.time()
        full.accumulated_token_usage = sum((f.token_usage or 0) for f in full.fragments)
        full.thinking_duration_s = sum((f.duration_s or 0.0) for f in full.fragments if f.fragment_type == "think")

        await logger.ainfo(
            "DocumentSummarize complete",
            duration_s=round(time.perf_counter() - t0, 3),
            body_chars=len(body),
            token_usage=token_usage_total,
            failed=failed,
        )
        get_stream_writer()({self.step: {"events": [full]}})

        ordering_ids.reset_for(_chat_id_from_state(state))

        return state
