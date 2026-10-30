from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from src.agents.document_process.graph import astream_events
from src.agents.document_process.models import AttachedDocument
from src.agents.document_process.state import AgentState
from src.app.api.utils.document_text import DocumentFileTooLargeError
from src.app.api.utils.sse import sse_pack
from src.app.api.utils.stub_response import (
    PROCESSING_THINK,
    STUB_ANONYMIZATION_PENDING,
    STUB_FILE_LOAD_FAILED,
    STUB_SUMMARIZE,
    STUB_TOO_LONG,
    STUB_UPLOAD_AND_ASK,
    STUB_UPLOAD_REQUIRED,
    iter_stub_sse,
)
from src.app.schemas.loaded_document import S3LoadedDocument
from src.common.config.config import settings
from src.common.exceptions import BaseError, build_error_terminal_packet, stream_exception_handler
from src.common.schemas.models import HistoryMessage, Message, MessageFragment
from src.infrastructure.document_store import document_local_store
from src.infrastructure.logger import get_logger
from src.infrastructure.s3_services.async_client import AnonymizingFileIsNotReadyError

logger = get_logger(__name__)

_SSE_QUEUE_MAX_SIZE = max(1, int(settings.get("SSE_QUEUE_MAX_SIZE", 100)))


async def _yield_until_done(task: asyncio.Task[Any], *, ping_interval: float) -> AsyncIterator[str]:
    """Yield SSE comment pings until `task` completes."""
    while not task.done():
        done, _pending = await asyncio.wait({task}, timeout=ping_interval)
        if not done:
            yield ": ping\n\n"
    await task


def _pack_graph_event(event: Message | MessageFragment) -> str:
    payload = event.model_dump()
    if isinstance(event, Message) and event.status in {"success", "error"}:
        payload = {"data": payload}
    return sse_pack(payload)


@dataclass
class PrepareData:
    current_text: str
    attached_documents: list[AttachedDocument]
    income_history_messages: list[HistoryMessage]
    document_intent_hint: str | None = None
    attachment_warnings: list[str] = field(default_factory=list)


# Как завершилась попытка загрузить один файл.
_OK = "ok"
_PENDING = "pending"
_TOO_LARGE = "too_large"
_FAILED = "failed"


@dataclass
class _LoadResult:
    """Итог загрузки набора файлов: что доехало и что с остальными."""

    documents: list[S3LoadedDocument] = field(default_factory=list)
    pending: int = 0
    too_large: int = 0
    failed: int = 0
    # Самое свежее вложение истории ещё обезличивается. Отдельный признак нужен
    # потому, что «не готов» и «не прочитался» требуют разного: пропускать можно
    # второе, а первое означает, что документ вот-вот появится.
    freshest_pending: bool = False

    def stub(self) -> str | None:
        """Заглушка, когда не доехало ничего.

        Порядок важен: «ещё обрабатывается» лечится повтором, остальное — нет.
        """
        if self.pending:
            return STUB_ANONYMIZATION_PENDING
        if self.too_large:
            return STUB_TOO_LONG
        if self.failed:
            return STUB_FILE_LOAD_FAILED
        return None


# Намерение, пришедшее параметром MCP. Оркестратор уже знает, саммари нужно или
# ответ на вопрос, — незачем угадывать это второй раз внутри навыка.
_INTENT_BY_NAME = {"summarize": "summarize", "qa": "document_qa", "document_qa": "document_qa"}


def _document_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "файл"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "файла"
    return "файлов"


class DocumentService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        history_messages: list[HistoryMessage],
        thread_id: str,
        message_id: UUID,
        s3_loader: Callable[[UUID, str | None], Awaitable[S3LoadedDocument]],
        intent: str | None = None,
    ) -> None:
        self.message_id = message_id
        self._intent_hint = _INTENT_BY_NAME.get((intent or "").strip().casefold())
        self._graph = graph
        self._history_messages = history_messages
        self._thread_id = thread_id
        self._s3_loader = s3_loader
        self._stream_created_at = time.time()
        self._stream_fragments: list[dict] = []

    def _reset_stream_error_context(self) -> None:
        self._stream_created_at = time.time()
        self._stream_fragments = []

    def _terminal_error_packet(self, error: BaseError) -> dict:
        return build_error_terminal_packet(
            message_id=str(self.message_id),
            created_at=self._stream_created_at,
            fragments=list(self._stream_fragments),
            error=error,
        )

    async def execute(self) -> AgentState | str:
        data_or_stub = await self._prepare()
        if isinstance(data_or_stub, str):
            return data_or_stub
        return self._build_state(data_or_stub)

    async def _load_one(self, file_id: UUID) -> tuple[S3LoadedDocument | None, str]:
        try:
            return await self._s3_loader(file_id, self._thread_id), _OK
        except DocumentFileTooLargeError as exc:
            await logger.awarning(f"Document {file_id} too large: {exc.size_bytes} > {exc.limit_bytes}")
            return None, _TOO_LARGE
        except AnonymizingFileIsNotReadyError:
            await logger.ainfo(f"Anonymized file {file_id} is not ready yet")
            return None, _PENDING
        except Exception as exc:
            await logger.aerror(f"Failed to load file {file_id}: {exc!s}")
            return None, _FAILED

    async def _load_current(self, file_ids: list[UUID], seen: set[str]) -> _LoadResult:
        """Все файлы текущего сообщения, а не последний.

        Пользователь приложил два документа — значит спросил про оба. Раньше
        доезжал только последний, и ответ молча описывал один из двух.
        """
        result = _LoadResult()
        for file_id in file_ids:
            key = str(file_id)
            if key in seen:
                continue
            seen.add(key)
            loaded, kind = await self._load_one(file_id)
            if loaded is not None:
                result.documents.append(loaded)
            elif kind == _PENDING:
                result.pending += 1
            elif kind == _TOO_LARGE:
                result.too_large += 1
            else:
                result.failed += 1
        return result

    async def _load_from_history(
        self,
        history: list[HistoryMessage],
        seen: set[str],
        *,
        limit: int,
    ) -> _LoadResult:
        """Документы предыдущих ходов — от свежих к старым, до лимита.

        Нужны, чтобы вопрос про ранее загруженный документ на него отвечал:
        «посмотри доверенность — даёт ли она право подписать этот договор»
        требует и доверенности, и договора, приложенного ходом раньше.
        """
        result = _LoadResult()
        if limit <= 0:
            return result
        for message in reversed(history):
            for file_id in reversed(message.file_ids or []):
                if len(result.documents) >= limit:
                    return result
                key = str(file_id)
                if key in seen:
                    continue
                seen.add(key)
                loaded, kind = await self._load_one(file_id)
                if loaded is not None:
                    result.documents.append(loaded)
                elif kind == _PENDING:
                    result.pending += 1
                    # Идти дальше вглубь истории нельзя: там лежит документ постарше,
                    # и ответ по нему выглядел бы как ответ по только что приложенному.
                    if not result.documents:
                        result.freshest_pending = True
                        return result
                elif kind == _TOO_LARGE:
                    result.too_large += 1
                else:
                    result.failed += 1
        return result

    @staticmethod
    def _warnings(current: _LoadResult) -> list[str]:
        """Про потерянные файлы текущего сообщения надо сказать вслух.

        Молчаливый пропуск неотличим для пользователя от «прочитал и не нашёл».
        """
        parts: list[str] = []
        if current.too_large:
            parts.append(f"{current.too_large} {_document_word(current.too_large)} превышает допустимый размер")
        if current.failed:
            parts.append(f"{current.failed} {_document_word(current.failed)} не удалось прочитать")
        if not parts:
            return []
        return [f"Учтены не все приложенные файлы: {', '.join(parts)}."]

    async def _prepare(self) -> PrepareData | str:
        history = self._history_messages
        if not history:
            return STUB_UPLOAD_AND_ASK

        user_message = history.pop()
        current_text = user_message.fragments[0].content or "" if user_message.fragments else ""
        has_text = bool(current_text and current_text.strip())

        seen: set[str] = set()
        current = await self._load_current(list(user_message.file_ids or []), seen)

        # Часть файлов ещё обезличивается: отвечать по половине приложенного —
        # тот же дефект, от которого уходим. Повтор запроса даст полный ответ.
        if current.pending:
            return STUB_ANONYMIZATION_PENDING

        max_documents = self._config_int("max_attached_documents", 4)
        history_limit = min(
            self._config_int("history_documents_limit", 2),
            max(0, max_documents - len(current.documents)),
        )
        historical = await self._load_from_history(history, seen, limit=history_limit)

        # В текущем сообщении файлов нет, а свежайшее вложение диалога ещё
        # обезличивается: просим повторить. Иначе ответ уехал бы по документу
        # из прошлых ходов — пользователь спрашивает не про него.
        if not current.documents and historical.freshest_pending:
            return STUB_ANONYMIZATION_PENDING

        # Происхождение сохраняем: дальше по конвейеру отличить документ этого хода
        # от подобранного из истории уже неоткуда, а вопрос задан именно про первый.
        loaded = [(doc, True) for doc in current.documents] + [(doc, False) for doc in historical.documents]
        attached_documents = [
            self._build_attached_document(doc, index, from_current_message=is_current)
            for index, (doc, is_current) in enumerate(loaded)
        ]
        has_doc = bool(attached_documents)

        if not has_doc:
            stub = current.stub() or historical.stub()
            if stub:
                return stub
            return STUB_UPLOAD_REQUIRED if has_text else STUB_UPLOAD_AND_ASK

        document_intent_hint = self._intent_hint
        if not has_text:
            current_text = STUB_SUMMARIZE
            document_intent_hint = document_intent_hint or "summarize"

        return PrepareData(
            current_text=current_text,
            attached_documents=attached_documents,
            income_history_messages=history,
            document_intent_hint=document_intent_hint,
            attachment_warnings=self._warnings(current),
        )

    @staticmethod
    def _config_int(key: str, default: int) -> int:
        value = settings.get("document_process", {}).get(key, default)
        return default if value is None else int(value)

    def _build_state(self, data: PrepareData) -> AgentState:
        user_message = self._build_user_message(data.current_text)
        state: AgentState = {
            "input_message": user_message,
            "meta": {"configurable": {"thread_id": self._thread_id}},
            "chat_id": self._thread_id,
            "message_id": self.message_id,
            "attached_documents": data.attached_documents,
            "income_history_messages": data.income_history_messages,
        }
        if data.document_intent_hint:
            state["document_intent_hint"] = data.document_intent_hint
        if data.attachment_warnings:
            state["attachment_warnings"] = data.attachment_warnings
        return state

    async def stream_response(self) -> AsyncIterator[str]:
        """Open SSE immediately, prepare (with pings), then stream the graph."""
        yield ": connected\n\n"
        # Real data frame (not just an SSE comment) so proxies keep the stream alive
        # while we wait for S3 / graph start — same shape as prepare_context.
        yield sse_pack(self._build_processing_message().model_dump())
        async for chunk in self._stream_response_body():
            yield chunk

    @stream_exception_handler(
        log_event="document_stream_error",
        to_terminal_packet=lambda error, self, *_args, **_kwargs: self._terminal_error_packet(error),
    )
    async def _stream_response_body(self) -> AsyncIterator[str]:
        self._reset_stream_error_context()
        ping_interval = max(0.1, float(settings.get("PING_INTERVAL", 10)))
        prepare_task: asyncio.Task[AgentState | str] | None = None
        prepared_state: AgentState | None = None
        try:
            prepare_task = asyncio.create_task(self.execute())
            async for ping in _yield_until_done(prepare_task, ping_interval=ping_interval):
                yield ping
            result = prepare_task.result()

            if isinstance(result, str):
                async for event in iter_stub_sse(self.message_id, result, connected=False):
                    yield event
                return

            prepared_state = result
            async for event in self.stream(result, connected=False):
                yield event
        finally:
            await self._cleanup_stream_response(
                prepare_task=prepare_task,
                prepared_state=prepared_state,
            )

    async def _cleanup_stream_response(
        self,
        *,
        prepare_task: asyncio.Task[AgentState | str] | None,
        prepared_state: AgentState | None,
    ) -> None:
        if prepare_task is not None and not prepare_task.done():
            prepare_task.cancel()
            await asyncio.gather(prepare_task, return_exceptions=True)
        state = prepared_state
        if state is None and prepare_task is not None and prepare_task.done() and not prepare_task.cancelled():
            with suppress(Exception):
                completed_result = prepare_task.result()
                if not isinstance(completed_result, str):
                    state = completed_result
        if state is not None:
            docs = state.get("attached_documents") or []
            document_local_store.release_attached_local_docs([doc.local_doc_id for doc in docs])

    async def stream(self, state: AgentState, *, connected: bool = True) -> AsyncIterator[str]:
        if connected:
            yield ": connected\n\n"
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_SSE_QUEUE_MAX_SIZE)
        stop = asyncio.Event()

        async def produce_events() -> None:
            cancelled = False
            try:
                async for frame in self._graph_event_frames(state):
                    await q.put(frame)
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                stop.set()
                if not cancelled:
                    await q.put(None)

        async def heartbeat() -> None:
            interval = max(0.1, float(settings.get("PING_INTERVAL", 10)))
            try:
                while not stop.is_set():
                    await asyncio.sleep(interval)
                    if stop.is_set():
                        break
                    await q.put(": ping\n\n")
            except asyncio.CancelledError:
                raise

        t_events = asyncio.create_task(produce_events())
        t_ping = asyncio.create_task(heartbeat())
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            raise
        finally:
            t_events.cancel()
            t_ping.cancel()
            await asyncio.gather(t_events, t_ping, return_exceptions=True)
            docs = state.get("attached_documents") or []
            document_local_store.release_attached_local_docs([doc.local_doc_id for doc in docs])

    @stream_exception_handler(
        log_event="document_graph_stream_error",
        to_terminal_packet=lambda error, self, *_args, **_kwargs: self._terminal_error_packet(error),
    )
    async def _graph_event_frames(self, state: AgentState) -> AsyncIterator[str]:
        async for event in astream_events(self._graph, state):
            if isinstance(event, MessageFragment):
                self._stream_fragments.append(event.model_dump(mode="json"))
            elif isinstance(event, Message):
                self._stream_fragments = [fragment.model_dump(mode="json") for fragment in event.fragments]
            yield _pack_graph_event(event)

    def _build_processing_message(self) -> Message:
        """First visible think fragment while acquire/prepare are still running."""
        return Message(
            message_id=self.message_id,
            parent_id=None,
            role="assistant",
            status="in_progress",
            fragments=[
                MessageFragment(
                    fragment_id=1,
                    fragment_type="think",
                    status="in_progress",
                    content=PROCESSING_THINK,
                )
            ],
        )

    @staticmethod
    def _build_user_message(content: str, parent_id: UUID | None = None) -> Message:
        return Message(
            parent_id=parent_id,
            role="user",
            status="success",
            fragments=[
                MessageFragment(
                    fragment_id=1,
                    fragment_type="request",
                    content=content,
                    status="success",
                )
            ],
        )

    @staticmethod
    def _build_attached_document(
        loaded: S3LoadedDocument,
        file_index: int = 0,
        *,
        from_current_message: bool = True,
    ) -> AttachedDocument:
        document = AttachedDocument(
            file_id=loaded.file_id,
            filename=loaded.filename,
            text="",
            file_index=file_index,
            char_count=loaded.char_count,
            local_doc_id=loaded.local_doc_id,
            from_current_message=from_current_message,
        )
        return document
