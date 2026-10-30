import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.llm.fallback import LocalFallbackRunnable
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_JSON_RX = re.compile(r"\{.*\}", re.S)


def normalize_query(q: str) -> str:
    return " ".join((q or "").split())


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_RX.search(text)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


_RETRY_KWARGS: dict[str, Any] = {
    "stop": stop_after_attempt(3),  # initial call + retries
    "wait": wait_exponential(min=0.5, max=5),
    "reraise": True,
    "before_sleep": lambda retry_state: logger.warning("Retrying LLM call: %s", retry_state.outcome.exception()),
}

# Total structured-call budget is larger than one attempt. For a fallback group
# it is split between candidates so the primary cannot consume it all.
_TOTAL_TIMEOUT_FACTOR = 3


def _collect_token_usage(cb) -> int | None:
    parts = _collect_token_usage_parts(cb)
    return parts["total_tokens"]


def _collect_token_usage_parts(cb) -> dict[str, int | None]:
    """Return prompt/completion/total token sums from a usage metadata callback."""
    prompt = 0
    completion = 0
    total = 0
    found_prompt = False
    found_completion = False
    found_total = False
    try:
        for usage in cb.usage_metadata.values():
            for key in ("input_tokens", "prompt_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    prompt += int(value)
                    found_prompt = True
                    break
            for key in ("output_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    completion += int(value)
                    found_completion = True
                    break
            value = usage.get("total_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += int(value)
                found_total = True
    except Exception:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if not found_total and (found_prompt or found_completion):
        total = prompt + completion
        found_total = True
    return {
        "prompt_tokens": prompt if found_prompt else None,
        "completion_tokens": completion if found_completion else None,
        "total_tokens": total if found_total else None,
    }


async def _call_llm_structured_async_once(
    llm: Any, messages_or_prompt: Any, schema: type[BaseModel], *, enforce_json: bool
) -> BaseModel:
    if hasattr(llm, "with_structured_output"):
        runnable = llm.with_structured_output(schema)
        obj = await runnable.ainvoke(messages_or_prompt)
        return obj if isinstance(obj, schema) else schema.model_validate(obj)

    bound = llm
    if enforce_json and hasattr(llm, "bind"):
        with suppress(Exception):
            bound = llm.bind(response_format={"type": "json_object"})

    if hasattr(bound, "ainvoke"):
        raw = await bound.ainvoke(messages_or_prompt)
    elif hasattr(bound, "apredict"):
        raw = await bound.apredict(messages_or_prompt)
    elif callable(bound):
        raw = bound(messages_or_prompt)
    else:
        raise TypeError("llm must support ainvoke/apredict or be callable")

    if isinstance(raw, dict):
        data = raw
    else:
        text = getattr(raw, "content", raw if isinstance(raw, str) else str(raw))
        try:
            data = json.loads(text)
        except Exception:
            data = _extract_json_block(text)

    return schema.model_validate(data)


async def _ainvoke_structured_with_retries(
    llm: Any,
    *,
    retry_config: dict[str, Any],
    call_once: Callable[[Any], Awaitable[BaseModel]],
) -> BaseModel:
    async for attempt in AsyncRetrying(**retry_config):
        with attempt:
            return await call_once(llm)

    raise RuntimeError("AsyncRetrying exited without yielding a result")


async def call_llm_structured_async_with_timeout(
    llm: Any,
    messages_or_prompt: Any,
    schema: type[BaseModel],
    *,
    enforce_json: bool = True,
    timeout_s: float | None = None,
    total_timeout_s: float | None = None,
) -> BaseModel:
    """
    Structured LLM call with hard wall-clock timeout.
    Tenacity retries the current candidate; LocalFallbackRunnable then tries the
    next model (schema / mid-call failures the gateway does not see).
    """
    retry_config = dict(_RETRY_KWARGS)
    attempt_timeout_s = timeout_s if timeout_s is not None and timeout_s > 0 else None

    async def call_once(candidate: Any) -> BaseModel:
        coro = _call_llm_structured_async_once(candidate, messages_or_prompt, schema, enforce_json=enforce_json)
        if attempt_timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=attempt_timeout_s)

    budget_s = total_timeout_s
    if budget_s is None and attempt_timeout_s is not None:
        budget_s = attempt_timeout_s * _TOTAL_TIMEOUT_FACTOR

    async def call_with_retries() -> BaseModel:
        if isinstance(llm, LocalFallbackRunnable):
            candidate_budget_s = budget_s / len(llm.candidates) if budget_s is not None else None
            return await llm.ainvoke_with_retries(
                retry_config=retry_config,
                call_once=call_once,
                candidate_budget_s=candidate_budget_s,
            )
        return await _ainvoke_structured_with_retries(llm, retry_config=retry_config, call_once=call_once)

    if budget_s is None:
        return await call_with_retries()
    return await asyncio.wait_for(call_with_retries(), timeout=budget_s)


async def ainvoke_with_timeout_retries(
    llm: Any,
    messages_or_prompt: Any,
    *,
    timeout_s: float | None = None,
    total_timeout_s: float | None = None,
    retry_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Plain ainvoke with per-attempt timeout and retries (map/reduce style calls)."""
    retry_config = dict(_RETRY_KWARGS)
    if retry_kwargs:
        retry_config.update(retry_kwargs)
    attempt_timeout_s = timeout_s if timeout_s is not None and timeout_s > 0 else None

    async def call_once(candidate: Any) -> Any:
        coro = candidate.ainvoke(messages_or_prompt)
        if attempt_timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=attempt_timeout_s)

    budget_s = total_timeout_s
    if budget_s is None and attempt_timeout_s is not None:
        budget_s = attempt_timeout_s * _TOTAL_TIMEOUT_FACTOR

    async def call_with_retries() -> Any:
        if isinstance(llm, LocalFallbackRunnable):
            candidate_budget_s = budget_s / len(llm.candidates) if budget_s is not None else None
            return await llm.ainvoke_with_retries(
                retry_config=retry_config,
                call_once=call_once,
                candidate_budget_s=candidate_budget_s,
            )
        async for attempt in AsyncRetrying(**retry_config):
            with attempt:
                return await call_once(llm)
        raise RuntimeError("AsyncRetrying exited without yielding a result")

    if budget_s is None:
        return await call_with_retries()
    return await asyncio.wait_for(call_with_retries(), timeout=budget_s)


def _content_part_to_text(part: Any) -> str:
    """Extract visible text from one streamed content part; skip reasoning blocks."""
    if isinstance(part, str):
        return part

    part_type: str | None = None
    text: Any = None
    if isinstance(part, dict):
        part_type = part.get("type")
        text = part.get("text")
        if text is None:
            text = part.get("content")
    else:
        part_type = getattr(part, "type", None)
        text = getattr(part, "text", None)
        if text is None:
            text = getattr(part, "content", None)

    if part_type in {"reasoning", "thinking", "reasoning_content"}:
        return ""
    if isinstance(text, str):
        return text
    return ""


def _stream_chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict) and "content" in chunk:
        content = chunk["content"]
    elif hasattr(chunk, "content"):
        content = chunk.content
    else:
        raise TypeError("stream chunk must be str/dict(content=...) or have .content attribute")

    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_content_part_to_text(part) for part in content)

    raise TypeError("stream chunk content must be str or list[content parts]")


def _stream_chunk_finish_reason(chunk: Any) -> str | None:
    """Extract a provider finish reason without coupling callers to chunk classes."""
    candidates: tuple[Any, ...]
    if isinstance(chunk, dict):
        candidates = (
            chunk.get("response_metadata"),
            chunk.get("generation_info"),
            chunk,
        )
    else:
        candidates = (
            getattr(chunk, "response_metadata", None),
            getattr(chunk, "generation_info", None),
        )

    for metadata in candidates:
        if not isinstance(metadata, dict):
            continue
        reason = metadata.get("finish_reason") or metadata.get("stop_reason") or metadata.get("finishReason")
        if reason is not None:
            normalized = str(reason).strip()
            return normalized or None
    return None


async def _astream_once(runnable: Any, messages_or_prompt: Any) -> AsyncIterator[Any]:
    if hasattr(runnable, "astream"):
        async for chunk in runnable.astream(messages_or_prompt):
            yield chunk
        return
    raise TypeError("runnable must support astream(...) for streaming")


async def _stream_attempt(
    runnable: Any,
    messages_or_prompt: Any,
    *,
    idle_timeout_s: float | None,
    accumulate_n: int,
    stream_mode: Literal["accumulated", "delta"],
    on_finish_reason: Callable[[str | None], None] | None,
) -> AsyncIterator[str]:
    full_text = ""
    pending: list[str] = []
    pending_count = 0

    stream_iter = _astream_once(runnable, messages_or_prompt)
    stream_iter_anext = stream_iter.__anext__

    while True:
        try:
            if idle_timeout_s is not None and idle_timeout_s > 0:
                chunk = await asyncio.wait_for(stream_iter_anext(), timeout=idle_timeout_s)
            else:
                chunk = await stream_iter_anext()
        except StopAsyncIteration:
            break

        finish_reason = _stream_chunk_finish_reason(chunk)
        if finish_reason is not None and on_finish_reason is not None:
            on_finish_reason(finish_reason)

        delta = _stream_chunk_to_text(chunk)
        if not delta:
            continue

        pending.append(delta)
        pending_count += 1
        if pending_count < accumulate_n:
            continue

        batched_delta = "".join(pending)
        pending.clear()
        pending_count = 0
        if stream_mode == "delta":
            yield batched_delta
        else:
            full_text += batched_delta
            yield full_text

    if pending:
        batched_delta = "".join(pending)
        if stream_mode == "delta":
            yield batched_delta
        else:
            full_text += batched_delta
            yield full_text


async def astream_with_retries(
    runnable: Any,
    messages_or_prompt: Any,
    *,
    idle_timeout_s: float | None = None,
    retry_kwargs: dict | None = None,
    accumulate_n: int = 1,
    stream_mode: Literal["accumulated", "delta"] = "accumulated",
    on_finish_reason: Callable[[str | None], None] | None = None,
) -> AsyncIterator[str]:
    """
    Stream a model response with retries and an idle timeout.

    By default yields the *accumulated* content so callers can safely upsert UI
    fragments with full text each time. With `stream_mode="delta"`, yields only
    new text since the previous yield.

    - `idle_timeout_s` bounds time waiting for the next streamed chunk.
    - `accumulate_n` controls how many streamed chunks to buffer before yielding.
    - `on_finish_reason` receives provider termination metadata and is reset to
      ``None`` whenever a retry or fallback starts a new response.

    If a streaming attempt fails (including mid-stream), the helper retries the
    current candidate, then LocalFallbackRunnable's next model, and yields an
    empty string at the start of the next attempt to signal a restart.
    """
    if stream_mode not in {"accumulated", "delta"}:
        raise ValueError(f"Unsupported stream_mode: {stream_mode}")
    accumulate_n = max(1, accumulate_n)
    retry_config = dict(_RETRY_KWARGS)
    if retry_kwargs:
        retry_config.update(retry_kwargs)

    async def stream_once(candidate: Any) -> AsyncIterator[str]:
        if on_finish_reason is not None:
            # A retry/fallback starts a new response, so metadata from the failed
            # attempt must not determine whether the final response was truncated.
            on_finish_reason(None)
        async for partial in _stream_attempt(
            candidate,
            messages_or_prompt,
            idle_timeout_s=idle_timeout_s,
            accumulate_n=accumulate_n,
            stream_mode=stream_mode,
            on_finish_reason=on_finish_reason,
        ):
            yield partial

    if isinstance(runnable, LocalFallbackRunnable):
        async for partial in runnable.astream_with_retries(retry_config=retry_config, stream_once=stream_once):
            yield partial
        return

    attempt_no = 0
    async for attempt in AsyncRetrying(**retry_config):
        attempt_no += 1
        with attempt:
            if attempt_no > 1:
                yield ""

            async for partial in stream_once(runnable):
                yield partial

            return
