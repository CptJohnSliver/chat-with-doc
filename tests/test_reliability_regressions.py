from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from src.agents.document_process import document_utils
from src.agents.document_process.models import AttachedDocument, RelevantDocs
from src.agents.document_process.nodes import base, intent_classifier, retrieve, summarize
from src.agents.document_process.nodes.intent_classifier import IntentClassifierNode
from src.agents.document_process.nodes.retrieve import DocumentRetrieveNode
from src.agents.document_process.nodes.summarize import (
    DocumentSummarizeNode,
    _SummaryResult,
    _looks_truncated_summary,
)
from src.app.api.utils import document_text
from src.app.api.utils.document_text import InvalidDocumentTextError
from src.app.schemas.document import DocumentRequest
from src.app.services import document as document_service_module
from src.app.services.document import DocumentService
from src.common.config.config import settings
from src.common.schemas.models import HistoryMessage, HistoryMessageFragment, Message
from src.infrastructure.document_store.local_store import DocumentLocalStore


def _cfg() -> object:
    return settings.document_process


def _summarize_node(llm: object = None, **overrides: object) -> DocumentSummarizeNode:
    cfg = _cfg()
    kwargs = {
        "map_reduce_threshold": int(cfg.get("summarize_map_reduce_threshold")),
        "map_max_chunks": int(cfg.get("summarize_map_max_chunks")),
        "map_concurrency": int(cfg.get("summarize_map_concurrency")),
        "map_segment_max": int(cfg.get("summarize_map_segment_max")),
        "map_max_segments": int(cfg.get("summarize_map_max_segments")),
        "max_document_chars": int(cfg.get("summarize_max_document_chars")),
    }
    kwargs.update(overrides)
    return DocumentSummarizeNode(llm, **kwargs)


def _intent_node(llm: object = None, **overrides: object) -> IntentClassifierNode:
    cfg = _cfg()
    kwargs = {"snippet_chars": int(cfg.get("snippet_chars"))}
    kwargs.update(overrides)
    return IntentClassifierNode(llm, **kwargs)


def _discard_event(_event: object) -> None:
    return None


def _output_state() -> dict:
    return {
        "chat_id": "chat",
        "output_message": Message(
            message_id=uuid4(),
            parent_id=None,
            role="assistant",
            status="in_progress",
            fragments=[],
        ),
    }


@pytest.mark.asyncio
async def test_summary_stream_restart_discards_failed_partial(monkeypatch) -> None:
    class FakePrompt:
        def compile(self, **_kwargs: object) -> None:
            return None

        def format(self, _target: str) -> list:
            return []

    async def fake_stream(*_args: object, **_kwargs: object):
        yield "failed-primary"
        yield ""
        yield "fallback"

    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    monkeypatch.setattr(summarize, "promptstore", {"summarize_document": FakePrompt()})
    monkeypatch.setattr(summarize, "astream_with_retries", fake_stream)

    result = await _summarize_node()._stream_final_summary(
        _output_state(),
        document_text="document",
        response_fragment_id=1,
    )

    assert result.body == "fallback"


@pytest.mark.asyncio
async def test_summary_truncation_is_measured_without_continuation(
    monkeypatch,
) -> None:
    class FakePrompt:
        def compile(self, **_kwargs: object) -> None:
            return None

        def format(self, _target: str) -> list:
            return []

    main_body = "Основная выжимка"
    calls = 0
    metrics = AsyncMock()

    async def fake_stream(_llm: object, _messages: list, **kwargs: object):
        nonlocal calls
        calls += 1
        on_finish_reason = kwargs["on_finish_reason"]
        assert callable(on_finish_reason)
        on_finish_reason("length")
        yield main_body

    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    monkeypatch.setattr(summarize, "promptstore", {"summarize_document": FakePrompt()})
    monkeypatch.setattr(summarize, "astream_with_retries", fake_stream)
    monkeypatch.setattr(summarize.logger, "ainfo", metrics)

    result = await _summarize_node()._stream_final_summary(
        _output_state(),
        document_text="document",
        response_fragment_id=1,
    )

    assert calls == 1
    assert result == _SummaryResult(body=main_body)
    assert metrics.await_args.kwargs["looks_truncated"] is True
    assert metrics.await_args.kwargs["appears_complete"] is False


def test_summary_truncation_detection_uses_reason_and_text() -> None:
    complete_list = "- Правило применяется.\n" * 500 + "- Срок — 31 декабря 2026."
    cutoff_after_word = "- Правило применяется.\n" * 500 + "Норма применяется к работникам"
    cutoff_after_heading = (
        "- Правило применяется.\n" * 500 + "### 5. Сроки и суммы:"
    )
    complete_table = "- Правило применяется.\n" * 500 + "| Срок | 31 декабря 2026 |"

    assert not _looks_truncated_summary(complete_list, finish_reason="stop")
    assert _looks_truncated_summary(complete_list, finish_reason="length")
    assert _looks_truncated_summary(cutoff_after_word, finish_reason="stop")
    assert _looks_truncated_summary(cutoff_after_word, finish_reason="")
    assert _looks_truncated_summary(cutoff_after_word)
    assert _looks_truncated_summary(cutoff_after_heading, finish_reason="stop")
    assert not _looks_truncated_summary(complete_table, finish_reason="stop")
    assert not _looks_truncated_summary("- Срок — 31 декабря 2026")


def test_map_reduce_batch_size_is_clamped_to_progress() -> None:
    assert _summarize_node(map_max_chunks=1).map_max_chunks == 2
    assert _summarize_node(map_max_chunks=0).map_max_chunks == 2


def test_summary_full_coverage_at_ten_million_chars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = DocumentLocalStore(root=tmp_path)
    local_doc_id, _workspace = store.allocate_workspace()
    max_document_chars = int(settings.document_process.get("summarize_max_document_chars"))
    max_segments = int(settings.document_process.get("summarize_map_max_segments"))
    max_chunk_chars = int(settings.document_process.get("summarize_map_segment_max"))
    soft_cut_lookback = 500
    minimum_window = max_chunk_chars - soft_cut_lookback
    required_at_worst_cut = math.ceil(max_document_chars / minimum_window)

    # Keep at least 20% headroom over the worst normal soft-cut case.
    assert max_segments >= math.ceil(required_at_worst_cut * 1.2)

    block = "a" * minimum_window + "\n"
    text = block * (max_document_chars // len(block))
    text += "b" * (max_document_chars - len(text))
    store.extracted_text_path(local_doc_id).write_text(text, encoding="utf-8")
    monkeypatch.setattr(document_utils, "document_local_store", store)

    segments = document_utils.select_map_segments(
        AttachedDocument(
            file_id="file-1",
            filename="ten-million.txt",
            file_index=0,
            local_doc_id=local_doc_id,
            char_count=len(text),
        ),
        max_chunk_chars=max_chunk_chars,
        max_segments=max_segments,
        require_full_coverage=True,
    )

    assert max_document_chars == 10_000_000
    assert len(segments) <= max_segments
    assert segments[0].startswith("a")
    assert segments[-1].endswith("b")
    assert sum(len(segment) for segment in segments) == len(text) - text.count("\n")


def test_intent_classifier_snippet_is_bounded() -> None:
    assert int(settings.document_process.get("snippet_chars")) == 8_000


@pytest.mark.asyncio
async def test_intent_classifier_failure_keeps_explicit_summary_intent(
    monkeypatch,
) -> None:
    class FakePrompt:
        def compile(self, **_kwargs: object) -> None:
            return None

        def format(self, _target: str) -> list:
            return []

    async def fail_classification(*_args: object, **_kwargs: object):
        raise TimeoutError("classifier timeout")

    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    monkeypatch.setattr(
        intent_classifier,
        "promptstore",
        {"intent_classification": FakePrompt()},
    )
    monkeypatch.setattr(
        intent_classifier,
        "call_llm_structured_async_with_timeout",
        fail_classification,
    )
    state = _output_state()
    state.update(
        normalized_user_query="Сделай краткую выжимку документа",
        attached_documents=[
            AttachedDocument(file_id="file-1", filename="document.txt", file_index=0, text="content")
        ],
    )

    await IntentClassifierNode.__call__.__wrapped__(_intent_node(), state)

    assert state["document_intent"] == "summarize"
    assert state["intent_classification_failed"] is True
    assert "резервным правилом" in state["output_message"].fragments[-1].content


@pytest.mark.asyncio
async def test_upload_only_intent_hint_skips_classifier(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    classifier = AsyncMock()
    state = _output_state()
    state.update(
        normalized_user_query="Суммаризируй документ",
        document_intent_hint="summarize",
        attached_documents=[
            AttachedDocument(file_id="file-1", filename="document.txt", file_index=0, text="content")
        ],
    )

    await IntentClassifierNode.__call__.__wrapped__(
        _intent_node(classifier),
        state,
    )

    assert state["document_intent"] == "summarize"
    classifier.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_mid_reduce_failure_aborts_without_forwarding_raw_partials(
    monkeypatch,
) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)

    async def fake_select(*_args: object, **_kwargs: object) -> list[str]:
        return ["segment"] * 11

    monkeypatch.setattr(summarize, "aselect_map_segments", fake_select)
    node = _summarize_node(
        map_reduce_threshold=1,
        map_max_chunks=10,
        max_document_chars=100,
    )
    raw_partial = "RAW_PARTIAL" * 300
    node._map_segments = AsyncMock(return_value=([raw_partial] * 11, 17, False))
    node._reduce_summaries_batch = AsyncMock(
        return_value=_SummaryResult(body="", failed=True)
    )
    node._stream_final_summary = AsyncMock()

    result = await node._summarize_document(
        _output_state(),
        doc=AttachedDocument(file_id="file-1", filename="document.txt", file_index=0, text="abc"),
        response_fragment_id=1,
        ordering_id=2,
    )

    assert result.failed is True
    assert "RAW_PARTIAL" not in result.body
    node._stream_final_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_mid_reduce_uses_summarize_llm(monkeypatch) -> None:
    class FakePrompt:
        def compile(self, **_kwargs: object) -> None:
            return None

        def format(self, _target: str) -> list:
            return []

    summarize_llm = object()
    map_llm = object()
    invoke = AsyncMock(return_value=SimpleNamespace(content="strong reduce"))
    monkeypatch.setattr(
        summarize,
        "promptstore",
        {"summarize_document_reduce": FakePrompt()},
    )
    monkeypatch.setattr(summarize, "ainvoke_with_timeout_retries", invoke)
    node = _summarize_node(
        summarize_llm,
        map_llm=map_llm,
        summarize_llm_timeout_s=91,
        map_llm_timeout_s=17,
    )

    result = await node._reduce_summaries_batch(
        partial_summaries=["one", "two"],
        prompt_lock=asyncio.Lock(),
    )

    assert result.body == "strong reduce"
    assert invoke.await_args.args[0] is summarize_llm
    assert invoke.await_args.kwargs["timeout_s"] == 91


@pytest.mark.asyncio
async def test_mid_reduce_batches_use_bounded_concurrency() -> None:
    node = _summarize_node(map_max_chunks=10, map_concurrency=2)
    active = 0
    max_active = 0
    calls = 0

    async def fake_reduce(
        *,
        partial_summaries: list[str],
        prompt_lock: asyncio.Lock,
    ) -> _SummaryResult:
        nonlocal active, max_active, calls
        del prompt_lock
        calls += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _SummaryResult(body=f"reduced-{len(partial_summaries)}")

    node._reduce_summaries_batch = fake_reduce  # type: ignore[method-assign]

    results = await node._reduce_summary_level(
        partial_summaries=["part"] * 25,
        prompt_lock=asyncio.Lock(),
    )

    assert calls == 3
    assert max_active == 2
    assert [result.body for result in results] == [
        "reduced-10",
        "reduced-10",
        "reduced-5",
    ]


@pytest.mark.asyncio
async def test_map_failure_aborts_without_reduce(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)

    async def fake_select(*_args: object, **_kwargs: object) -> list[str]:
        return ["segment-a", "segment-b", "segment-c"]

    monkeypatch.setattr(summarize, "aselect_map_segments", fake_select)
    node = _summarize_node(
        map_reduce_threshold=1,
        map_max_chunks=10,
        max_document_chars=100,
    )
    node._map_segments = AsyncMock(
        return_value=(["partial-a", "partial-b"], 3, True)
    )
    node._reduce_summaries_batch = AsyncMock()
    node._stream_final_summary = AsyncMock()

    result = await node._summarize_document(
        _output_state(),
        doc=AttachedDocument(file_id="file-1", filename="document.txt", file_index=0, text="abc"),
        response_fragment_id=1,
        ordering_id=2,
    )
    assert result.failed is True
    assert "partial-" not in result.body
    node._reduce_summaries_batch.assert_not_awaited()
    node._stream_final_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_map_workers_are_cancelled_and_joined_with_parent(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    node = _summarize_node(map_concurrency=3)
    all_started = asyncio.Event()
    never_finish = asyncio.Event()
    started = 0
    cancelled = 0

    async def blocking_map(**kwargs: object):
        nonlocal started, cancelled
        started += 1
        if started == 3:
            all_started.set()
        try:
            await never_finish.wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return int(kwargs["chunk_index"]), None

    monkeypatch.setattr(node, "_map_one_chunk", blocking_map)
    task = asyncio.create_task(
        node._map_segments(
            _output_state(),
            segments=["a", "b", "c", "d"],
            ordering_id=1,
        )
    )
    await asyncio.wait_for(all_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled == 3


@pytest.mark.asyncio
async def test_empty_document_is_rejected_and_workspace_released(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class EmptyClient:
        async def metadata(self, *, key: str) -> None:
            return None

        async def download_anonymized_to_file(self, **kwargs: object) -> None:
            Path(kwargs["output_path"]).write_bytes(b"")

    store = DocumentLocalStore(root=tmp_path)
    monkeypatch.setattr(document_text, "document_local_store", store)

    with pytest.raises(InvalidDocumentTextError):
        await document_text.load_anonymized_document("file", EmptyClient())

    assert store._workspaces == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_empty_document_does_not_reach_summary_llm(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    node = _summarize_node()
    node._stream_final_summary = AsyncMock()

    result = await node._summarize_document(
        _output_state(),
        doc=AttachedDocument(file_id="file-1", filename="empty.txt", file_index=0, char_count=0),
        response_fragment_id=1,
        ordering_id=2,
    )

    assert result.failed is True
    assert "не содержит текста" in result.body
    node._stream_final_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_failure_uses_keyword_hit_with_adjacent_segments(
    monkeypatch,
) -> None:
    """Сохраняет соседей найденного по словам фрагмента при сбое векторизации."""
    segments = [f"Общее положение {index}." for index in range(100)]
    segments[89] = "Условие, к которому относится специальное правило."
    segments[90] = "ИСКЛЮЧЕНИЕ применяется к специальной категории работников."
    segments[91] = "Ограничение и срок действия специального правила."

    scans = 0

    def fake_iter(_doc: AttachedDocument, *, max_chunk_chars: int):
        nonlocal scans
        scans += 1
        del max_chunk_chars
        yield from segments

    async def failed_vector_search(*_args: object, **_kwargs: object):
        raise ConnectionError("embeddings unavailable")

    monkeypatch.setattr(
        document_utils,
        "iter_map_chunks_for_document",
        fake_iter,
    )
    monkeypatch.setattr(document_utils, "vector_search", failed_vector_search)
    stream_writer = MagicMock(return_value=_discard_event)
    monkeypatch.setattr(base, "get_stream_writer", stream_writer)

    document = AttachedDocument(
        file_id="file-1",
        filename="law.txt",
        file_index=0,
        char_count=1_000_000,
    )
    node = DocumentRetrieveNode(
        segment_max_size=100,
        top_k=3,
        core_top_k=1,
        chunk_threshold=100,
        full_embed_max_chars=100,
        keyword_candidate_limit=8,
        embed_batch_size=8,
        fallback_chars=100,
    )
    state = _output_state()
    state.update(attached_documents=[document], normalized_user_query="исключение")

    await node(state)

    chunks = state["relevant_docs"].document
    assert {chunk.segment_index for chunk in chunks} == {89, 90, 91}
    assert chunks[0].segment_index == 90
    assert chunks[0].selection_reason == "search"
    assert all(chunk.selection_reason == "neighbor" for chunk in chunks[1:])
    assert scans == 1


@pytest.mark.asyncio
async def test_adjacent_retrieval_keeps_full_semantic_seed_budget(
    monkeypatch,
) -> None:
    """Передаёт все самостоятельные попадания и соседний текст в общий отбор."""
    segments = [f"Положение {index}." for index in range(40)]
    requested_top_k: list[int] = []

    monkeypatch.setattr(
        document_utils,
        "segment_document",
        lambda *_args, **_kwargs: segments,
    )

    async def fake_vector_search(
        _query: str,
        chunks: list[str],
        *,
        top_k: int,
        batch_size: int,
    ) -> list[tuple[int, str, float]]:
        del batch_size
        requested_top_k.append(top_k)
        return [
            (index, chunks[index], 1.0 - index * 0.001)
            for index in range(top_k)
        ]

    monkeypatch.setattr(document_utils, "vector_search", fake_vector_search)

    hits, total, adjacent = await document_utils.retrieve_scored_segments(
        AttachedDocument(
            file_id="file-1",
            filename="law.txt",
            file_index=0,
            text="document",
        ),
        "положение",
        max_segment_chars=100,
        keyword_candidate_limit=40,
        embed_batch_size=8,
        top_k=30,
        full_embed_max_chars=1_000,
    )

    assert requested_top_k == [30]
    assert total == len(segments)
    assert len(hits) == 30
    assert {index for index, _text, _score in hits} == set(range(30))
    assert set(adjacent) == set(range(31))
    assert adjacent[30] == segments[30]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "retrieval", "no_hits"])
async def test_retrieval_preserves_same_named_file_identities(
    monkeypatch, mode: str
) -> None:
    """Сохраняет источник каждой части во всех ветках чтения документов."""
    stream_writer = MagicMock(return_value=_discard_event)
    monkeypatch.setattr(base, "get_stream_writer", stream_writer)
    current_document = AttachedDocument(
        file_id="file-current",
        filename="договор.txt",
        text="Оплата в течение 5 дней.",
        file_index=0,
        from_current_message=True,
    )
    history_document = AttachedDocument(
        file_id="file-history",
        filename="договор.txt",
        text="Оплата в течение 30 дней.",
        file_index=1,
        from_current_message=False,
    )
    if mode == "retrieval":
        search = AsyncMock(side_effect=[
            ([(0, current_document.text, 0.7)], 1, {0: current_document.text}),
            ([(0, history_document.text, 0.9)], 1, {0: history_document.text}),
        ])
    else:
        search = AsyncMock(return_value=([], 1, {}))
    monkeypatch.setattr(retrieve, "retrieve_scored_segments", search)
    chunk_threshold = 1_000 if mode == "full" else 1
    node = DocumentRetrieveNode(
        segment_max_size=1_000,
        top_k=2,
        core_top_k=2,
        chunk_threshold=chunk_threshold,
        full_embed_max_chars=1_000,
        keyword_candidate_limit=10,
        embed_batch_size=10,
        fallback_chars=1_000,
    )
    state = _output_state()
    state.update(
        attached_documents=[current_document, history_document],
        normalized_user_query="Какой срок оплаты?",
    )

    state = await node(state)

    chunks = state["relevant_docs"].document
    assert len(chunks) == 2
    by_file_id = {chunk.file_id: chunk for chunk in chunks}
    assert by_file_id["file-current"].text == current_document.text
    assert by_file_id["file-history"].text == history_document.text
    assert by_file_id["file-current"].from_current_message is True
    assert by_file_id["file-history"].from_current_message is False
    assert all(chunk.doc_name == "договор.txt" for chunk in chunks)
    expected_reason = {"full": "full", "retrieval": "search", "no_hits": "fallback"}[mode]
    expected_index = 0 if mode == "retrieval" else None
    assert all(chunk.selection_reason == expected_reason for chunk in chunks)
    assert all(chunk.segment_index == expected_index for chunk in chunks)
    if mode == "full":
        search.assert_not_awaited()
    else:
        assert search.await_count == 2


@pytest.mark.asyncio
async def test_all_empty_documents_mark_retrieval_as_error(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    state = _output_state()
    state.update(
        attached_documents=[
            AttachedDocument(file_id="file-1", filename="empty.txt", file_index=0, char_count=0),
        ],
        relevant_docs=RelevantDocs(),
        normalized_user_query="Что написано в документе?",
        is_valid_request=True,
    )
    node = DocumentRetrieveNode(
        segment_max_size=5_000,
        top_k=30,
        core_top_k=20,
        chunk_threshold=150_000,
        full_embed_max_chars=1_000_000,
        keyword_candidate_limit=500,
        embed_batch_size=64,
        fallback_chars=50_000,
    )

    await node(state)

    assert state["is_valid_request"] is False
    assert state["request_error_message"] == "Документ не содержит текста для поиска."
    assert state["relevant_docs"].document == []
    assert state["output_message"].fragments[-1].status == "error"


@pytest.mark.asyncio
async def test_cancelled_download_releases_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    class CancelledClient:
        async def metadata(self, *, key: str) -> None:
            return None

        async def download_anonymized_to_file(self, **_kwargs: object) -> None:
            raise asyncio.CancelledError

    store = DocumentLocalStore(root=tmp_path)
    monkeypatch.setattr(document_text, "document_local_store", store)

    with pytest.raises(asyncio.CancelledError):
        await document_text.load_anonymized_document("file", CancelledClient())

    assert store._workspaces == {}
    assert list(tmp_path.iterdir()) == []


def test_release_defers_rmtree_while_reader_is_active(tmp_path: Path) -> None:
    store = DocumentLocalStore(root=tmp_path)
    local_doc_id, workspace = store.allocate_workspace()
    extracted = workspace / "extracted.txt"
    extracted.write_text("hello world " * 20, encoding="utf-8")

    windows = store.iter_extracted_text_windows(local_doc_id, max_chars=8)
    assert next(windows)

    store.release_local_doc(local_doc_id)
    assert local_doc_id in store._pending_delete
    assert workspace.exists()
    # Generator still holds the reader lease — further reads must succeed.
    assert next(windows)

    windows.close()
    assert local_doc_id not in store._workspaces
    assert not workspace.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_status", ["available", "missing", "failed"])
async def test_successful_decode_keeps_source_id_without_raw_download(
    tmp_path: Path, monkeypatch, metadata_status: str
) -> None:
    """Сохраняет исходный ID независимо от доступности метаданных файла."""
    class TextClient:
        async def metadata(self, *, key: str) -> SimpleNamespace | None:
            """Имитирует доступность метаданных загруженного файла."""
            if metadata_status == "failed":
                raise ConnectionError("metadata unavailable")
            metadata = None
            if metadata_status == "available":
                metadata = SimpleNamespace(original_filename="договор.txt")
            return metadata

        async def download_anonymized_to_file(self, **kwargs: object) -> None:
            """Записывает текст вместо сетевого скачивания."""
            Path(kwargs["output_path"]).write_bytes("текст".encode())

    store = DocumentLocalStore(root=tmp_path)
    monkeypatch.setattr(document_text, "document_local_store", store)

    source_id = uuid4()
    source_file_id = str(source_id)
    client = TextClient()
    loaded = await document_text.load_anonymized_document(source_file_id, client)
    workspace = store.get_workspace(loaded.local_doc_id)

    assert loaded.file_id == source_file_id
    assert loaded.local_doc_id != source_file_id
    expected_filename = "договор.txt" if metadata_status == "available" else ""
    assert loaded.filename == expected_filename
    assert [path.name for path in workspace.iterdir()] == ["extracted.txt"]
    store.release_local_doc(loaded.local_doc_id)


def test_binary_document_is_rejected() -> None:
    with pytest.raises(InvalidDocumentTextError):
        document_text._decode_text_bytes(b"valid prefix\x00binary payload")
    with pytest.raises(InvalidDocumentTextError):
        document_text._decode_text_bytes(bytes(range(32)) * 20)


def test_cp1251_fallback_is_streamed_without_read_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.bin"
    extracted = tmp_path / "extracted.txt"
    expected = "Привет, мир!\n" * 100_000
    source.write_bytes(expected.encode("cp1251"))

    def forbid_read_bytes(_path: Path) -> bytes:
        raise AssertionError("cp1251 fallback must not load the whole file")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    count = document_text._fallback_decode_to_extracted(source, extracted)

    assert count == len(expected)
    assert extracted.read_text(encoding="utf-8") == expected


def test_stale_cleanup_removes_orphans_but_preserves_active(tmp_path: Path) -> None:
    store = DocumentLocalStore(root=tmp_path)
    _active_id, active_path = store.allocate_workspace()
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    old = time.time() - 10
    os.utime(orphan, (old, old))

    removed = store.cleanup_stale_workspaces(max_age_seconds=1)

    assert removed == 1
    assert active_path.exists()
    assert not orphan.exists()


def test_keyword_prefilter_is_deterministic_and_reserves_coverage() -> None:
    segments = [f"shared term segment {index}" for index in range(1000)]

    first, total = document_utils.keyword_prefilter_segments(
        "shared",
        iter(segments),
        candidate_limit=40,
    )
    second, _ = document_utils.keyword_prefilter_segments(
        "shared",
        iter(segments),
        candidate_limit=40,
    )

    assert total == 1000
    assert first == second
    assert len(first) == 40
    assert any(index >= 30 for index, _text, _score in first)
    assert first[-1][0] == 999


@pytest.mark.asyncio
async def test_stream_opens_before_document_loader_runs() -> None:
    loader = AsyncMock()
    service = DocumentService(
        graph=MagicMock(),
        history_messages=[
            HistoryMessage(
                role="user",
                fragments=[
                    HistoryMessageFragment(fragment_type="request", content="question")
                ],
                file_ids=[uuid4()],
            )
        ],
        thread_id="thread",
        message_id=uuid4(),
        s3_loader=loader,
    )

    stream = service.stream_response()
    assert await anext(stream) == ": connected\n\n"
    first_data = await anext(stream)
    assert "Обрабатываю запрос" in first_data
    assert "in_progress" in first_data
    assert "event: error" not in first_data
    loader.assert_not_awaited()
    await stream.aclose()


def test_stub_message_closes_processing_think() -> None:
    from src.app.api.utils.stub_response import PROCESSING_THINK, build_stub_message

    message = build_stub_message(uuid4(), "pending text")
    assert message.status == "success"
    assert message.fragments[0].fragment_type == "think"
    assert message.fragments[0].status == "success"
    assert message.fragments[0].content == PROCESSING_THINK
    assert message.fragments[1].fragment_type == "response"
    assert message.fragments[1].status == "success"
    assert message.fragments[1].content == "pending text"


@pytest.mark.asyncio
async def test_empty_graph_stream_terminates_without_waiting_for_ping(
    monkeypatch,
) -> None:
    async def no_events(*_args: object, **_kwargs: object):
        if False:
            yield None

    monkeypatch.setattr(document_service_module, "astream_events", no_events)
    service = DocumentService(MagicMock(), [], "thread", uuid4(), AsyncMock())

    chunks = await asyncio.wait_for(
        anext_then_collect(service.stream({"attached_documents": []})),
        timeout=1,
    )

    assert chunks == [": connected\n\n"]


@pytest.mark.asyncio
async def test_graph_error_does_not_leak_exception_text(monkeypatch) -> None:
    async def failed_events(*_args: object, **_kwargs: object):
        raise RuntimeError("secret internal URL")
        yield

    monkeypatch.setattr(document_service_module, "astream_events", failed_events)
    service = DocumentService(MagicMock(), [], "thread", uuid4(), AsyncMock())

    chunks = await anext_then_collect(service.stream({"attached_documents": []}))

    body = "".join(chunks)
    assert "secret internal URL" not in body
    assert "event: error" not in body
    terminal = next((parsed for chunk in chunks if (parsed := _parse_terminal_payload(chunk))), None)
    assert terminal is not None
    assert terminal["status"] == "error"
    assert "Ошибка сервера" in body


def _parse_terminal_payload(event_text: str) -> dict | None:
    import json

    data_lines: list[str] = []
    for line in event_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            _, value = line.split(":", 1)
            data_lines.append(value.lstrip())
    data_str = "\n".join(data_lines).strip()
    if not data_str:
        return None
    obj = json.loads(data_str)
    payload = obj.get("data") if isinstance(obj, dict) else None
    if isinstance(payload, dict) and payload.get("status") in {"success", "error"}:
        return payload
    return None


async def anext_then_collect(stream) -> list[str]:
    return [chunk async for chunk in stream]


def test_request_history_is_bounded() -> None:
    message = HistoryMessage(
        role="user",
        fragments=[HistoryMessageFragment(fragment_type="request", content="q")],
    )
    with pytest.raises(ValidationError):
        DocumentRequest(history_messages=[message] * 101)


@pytest.mark.asyncio
async def test_stream_error_body_is_read_before_raise() -> None:
    """Regression: stream + unread .json()/.text masked real HTTP errors as ResponseNotRead."""
    from src.infrastructure.s3_services.async_client import (
        S3ServiceAsyncClient,
        S3ServiceClientError,
    )

    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(
        500,
        request=request,
        stream=httpx.ByteStream(b'{"detail":"upstream boom"}'),
        headers={"content-type": "application/json"},
    )
    with pytest.raises(S3ServiceClientError, match="upstream boom"):
        await S3ServiceAsyncClient._raise_for_status(response)
