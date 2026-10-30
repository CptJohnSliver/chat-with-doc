from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, ParamSpec, cast

from src.app.api.utils.sse import sse_pack
from src.infrastructure.logger import get_logger

from .base import BaseError, DomainException
from .exceptions import ServerException

P = ParamSpec("P")
logger = get_logger(__name__)

_STATUS_ERROR = "error"
_FRAGMENT_TYPE_RESPONSE = "response"


def wrap_exception(exc: Exception, *, fallback: type[DomainException] = ServerException) -> DomainException:
    if isinstance(exc, DomainException):
        return exc
    return fallback(source=exc)


def metrics_exception_type(domain: DomainException, *, original: Exception | None = None) -> str:
    """Имя типа исключения для structured-log и метрик.

    Клиенту при неожиданной ошибке отдаётся ``ServerException`` с общим текстом,
    но в логах и счётчиках нужен реальный класс исходной ошибки (``RuntimeError``,
    ``TimeoutError`` и т.д.). Для ``ServerException`` возвращает ``type(source)``;
    для остальных ``DomainException`` — ``domain.type``.
    """
    if isinstance(domain, ServerException):
        root = domain.source if domain.source is not None else original
        if root is not None:
            return type(root).__name__
    return domain.type


def build_error_terminal_packet(
    *,
    message_id: str,
    created_at: float,
    fragments: list[dict[str, Any]],
    error: BaseError,
    role: str = "assistant",
) -> dict[str, Any]:
    """Terminal SSE-пакет ``data.status=error`` для оркестратора (``parse_terminal_payload``)."""
    fragment_id = max((int(fragment.get("fragment_id", 0)) for fragment in fragments), default=0) + 1
    error_fragment = {
        "fragment_id": fragment_id,
        "fragment_type": _FRAGMENT_TYPE_RESPONSE,
        "status": _STATUS_ERROR,
        "content": error.user_message,
        "token_usage": None,
        "duration_s": None,
        "streaming": False,
    }
    return {
        "data": {
            "message_id": message_id,
            "role": role,
            "status": _STATUS_ERROR,
            "fragments": [*fragments, error_fragment],
            "created_at": created_at,
            "updated_at": time.time(),
            "thinking_duration_s": max(0.0, time.time() - created_at),
            "accumulated_token_usage": 0,
        }
    }


def build_sse_error_frame(error: BaseError) -> str:
    """SSE ``event: error`` для локальной отладки без оркестратора."""
    payload: dict[str, Any] = {
        "ok": False,
        **error.model_dump(exclude_none=True, mode="json"),
        "error": error.user_message,
    }
    return sse_pack(payload, event="error")


async def _normalize_stream_error(
    exc: Exception,
    *,
    log_event: str,
    fallback: type[DomainException],
    **log_context: Any,
) -> BaseError:
    domain_exc = wrap_exception(exc, fallback=fallback)
    await logger.aerror(
        log_event,
        **domain_exc.log_data,
        exc_info=not isinstance(exc, DomainException),
        **log_context,
    )
    return domain_exc.to_base()


async def build_terminal_error_packet(
    exc: Exception,
    *,
    log_event: str,
    fallback: type[DomainException] = ServerException,
    to_terminal_packet: Callable[[BaseError], dict[str, Any]],
    **log_context: Any,
) -> dict[str, Any]:
    """Terminal error dict для MCP и оркестратора (без SSE-обёртки)."""
    base = await _normalize_stream_error(
        exc,
        log_event=log_event,
        fallback=fallback,
        **log_context,
    )
    return to_terminal_packet(base)


async def _build_sse_error_frame(
    exc: Exception,
    *,
    log_event: str,
    fallback: type[DomainException],
    to_terminal_packet: Callable[[BaseError], dict[str, Any]] | None = None,
    **log_context: Any,
) -> str:
    base = await _normalize_stream_error(
        exc,
        log_event=log_event,
        fallback=fallback,
        **log_context,
    )
    if to_terminal_packet is None:
        return build_sse_error_frame(base)
    return sse_pack(to_terminal_packet(base))


def stream_exception_handler(
    *,
    log_event: str,
    fallback: type[DomainException] = ServerException,
    to_terminal_packet: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[Callable[P, AsyncIterator[str]]], Callable[P, AsyncIterator[str]]]:
    """Async-генератор стрима: при ошибке yield SSE-кадр вместо raise.

    С ``to_terminal_packet`` — terminal ``data.status=error`` для оркестратора.
    Без него — ``event: error`` (только локальная отладка).
    """

    def decorator(func: Callable[P, AsyncIterator[str]]) -> Callable[P, AsyncIterator[str]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[str]:
            try:
                async for chunk in func(*args, **kwargs):
                    yield chunk
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                build_packet = (lambda error: to_terminal_packet(error, *args, **kwargs)) if to_terminal_packet else None
                yield await _build_sse_error_frame(
                    exc,
                    log_event=log_event,
                    fallback=fallback,
                    to_terminal_packet=build_packet,
                )

        return wrapper

    return decorator


def mcp_log_context(*args: object, **kwargs: object) -> dict[str, object]:
    """Поля для structured-log MCP-стрима: ``message_id`` из kwargs или dict-контекста в args."""
    if "message_id" in kwargs:
        return {"message_id": kwargs["message_id"]}
    for arg in args:
        if isinstance(arg, dict) and "message_id" in arg:
            return {"message_id": arg["message_id"]}
    return {}


def mcp_exception_handler(
    *,
    log_event: str,
    fallback: type[DomainException] = ServerException,
    to_terminal_packet: Callable[..., dict[str, Any]],
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """Оборачивает async MCP-стрим: при ошибке возвращает terminal error dict вместо raise.

    Обёрнутая функция возвращает terminal-пакет ``{"data": {"status": "success"|"error", ...}}``.
    Декоратор возвращает ``(final_packet, finish_reason)``, где ``finish_reason`` ∈ ``{"success", "error"}``.

    При ошибке:
    - вызывает ``build_terminal_error_packet`` (wrap + log + ``to_terminal_packet``);
    - если в ``kwargs`` передан ``on_event``, шлёт error-пакет в callback (как SSE-декоратор yield'ит кадр).

    ``to_terminal_packet(error, *args, **kwargs)`` — сборщик terminal error dict; те же args/kwargs, что у функции.

    ``CancelledError`` пробрасывается без преобразования.
    """

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[dict[str, Any], str]:
            try:
                final = await func(*args, **kwargs)
                return final, "success"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_context = mcp_log_context(*args, **kwargs)
                final = await build_terminal_error_packet(
                    exc,
                    log_event=log_event,
                    fallback=fallback,
                    to_terminal_packet=lambda error: to_terminal_packet(error, *args, **kwargs),
                    **log_context,
                )
                on_event = kwargs.get("on_event")
                if on_event is not None:
                    callback = cast(Callable[[dict[str, Any]], Any], on_event)
                    maybe_awaitable = callback(final)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
                return final, "error"

        return wrapper

    return decorator
