"""App-side model fallback for failures llm-gateway does not see.

The gateway walks ``models`` on transport / pre-content errors. A 200 with
unparseable structured output, or a stream that dies after the first token, is
the client's problem: retry the current candidate, then the next ChatOpenAI.

HTTP 4xx/5xx, connection failures, and timeouts are not retried on another
model here — the gateway already walked that ladder.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.exceptions import OutputParserException
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError
from tenacity import AsyncRetrying

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_RESTART_MARKER = ""

_SCHEMA_ERROR_TYPES = (ValidationError, json.JSONDecodeError, OutputParserException)
_TRANSPORT_ERROR_TYPES = (
    TimeoutError,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)


def should_fallback_to_next_model(
    exc: BaseException,
    *,
    delivered_content: bool = False,
) -> bool:
    """True for schema/parse errors and mid-stream disconnects after tokens."""
    if delivered_content:
        return True
    return _is_schema_or_parse_error(exc)


def _is_schema_or_parse_error(exc: BaseException, *, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(exc, _SCHEMA_ERROR_TYPES):
        return True
    if isinstance(exc, ValueError) and "json" in str(exc).lower():
        return True
    if isinstance(exc, _TRANSPORT_ERROR_TYPES):
        return False
    for inner in (exc.__cause__, exc.__context__):
        if inner is not None and inner is not exc and _is_schema_or_parse_error(inner, depth=depth + 1):
            return True
    return False


class LocalFallbackRunnable:
    """Run LLM candidates in order after retries for the previous one are exhausted."""

    def __init__(self, candidates: Sequence[Any], *, name: str | None = None) -> None:
        if not candidates:
            raise ValueError("LocalFallbackRunnable requires at least one candidate")

        self.candidates = tuple(candidates)
        self.name = name or "llm"

    async def ainvoke_with_retries(
        self,
        *,
        retry_config: dict[str, Any],
        call_once: Callable[[Any], Any],
        candidate_budget_s: float | None = None,
    ) -> Any:
        """Bound all retries of one candidate so a slow primary leaves time for a fallback."""
        for candidate_index, candidate in enumerate(self.candidates):
            try:
                async with asyncio.timeout(candidate_budget_s):
                    async for attempt in AsyncRetrying(**retry_config):
                        with attempt:
                            return await call_once(candidate)
            except Exception as exc:
                if candidate_index == len(self.candidates) - 1 or not should_fallback_to_next_model(exc):
                    raise
                await self._alog_retries_exhausted(candidate_index, exc)

        raise RuntimeError("No LLM candidates configured")

    async def astream_with_retries(
        self,
        *,
        retry_config: dict[str, Any],
        stream_once: Callable[[Any], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        attempts = 0
        for candidate_index, candidate in enumerate(self.candidates):
            delivered_content = False
            try:
                async for attempt in AsyncRetrying(**retry_config):
                    attempts += 1
                    with attempt:
                        # Consumers use an empty fragment as a signal to discard
                        # a partial response from the failed stream attempt.
                        if attempts > 1:
                            yield _RESTART_MARKER

                        async for partial in stream_once(candidate):
                            if partial:
                                delivered_content = True
                            yield partial
                        return
            except Exception as exc:
                if candidate_index == len(self.candidates) - 1 or not should_fallback_to_next_model(
                    exc,
                    delivered_content=delivered_content,
                ):
                    raise
                await self._alog_retries_exhausted(candidate_index, exc)

        raise RuntimeError("No LLM candidates configured")

    async def _alog_retries_exhausted(self, candidate_index: int, exc: BaseException) -> None:
        await logger.awarning(
            "LLM candidate exhausted retries; trying fallback",
            fallback_group=self.name,
            model=self._candidate_model(candidate_index),
            fallback_model=self._candidate_model(candidate_index + 1),
            error_class=type(exc).__name__,
        )

    def _candidate_model(self, candidate_index: int) -> str:
        candidate = self.candidates[candidate_index]
        value = getattr(candidate, "model_name", None) or getattr(candidate, "model", None) or type(candidate).__name__
        return str(value)
