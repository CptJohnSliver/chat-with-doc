from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from gigachat.exceptions import RequestEntityTooLargeError
from src.agents.document_process import document_utils
from src.agents.document_process.models import AttachedDocument, RelevantDocs
from src.agents.document_process.nodes import base, qa_answer, summarize
from src.agents.document_process.nodes.prepare_context import PrepareContextDocNode
from src.agents.document_process.nodes.qa_answer import DocumentQAAnswerNode
from src.agents.document_process.nodes.summarize import (
    DocumentSummarizeNode,
    _SummaryResult,
)
from src.agents.document_process.prompts import DocumentValidationOut
from src.common.config.config import settings
from src.common.schemas.models import Message, MessageFragment
from src.infrastructure.embeddings import gigachat as embeddings_module
from src.infrastructure.embeddings.gigachat import GigaChatEmbeddings


def _summarize_node(llm: object = None, **overrides: object) -> DocumentSummarizeNode:
    cfg = settings.document_process
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


class _FakeEmbeddingsClient:
    instances: list[_FakeEmbeddingsClient] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.batch_sizes: list[int] = []
        self.instances.append(self)

    async def aembeddings(self, documents: list[str], *, model: str) -> SimpleNamespace:
        self.batch_sizes.append(len(documents))
        data = [
            SimpleNamespace(embedding=[float(index + 1), 1.0])
            for index in range(len(documents))
        ]
        return SimpleNamespace(data=data)

    async def aclose(self) -> None:
        return None


class _PayloadLimitedClient(_FakeEmbeddingsClient):
    async def aembeddings(self, documents: list[str], *, model: str) -> SimpleNamespace:
        self.batch_sizes.append(len(documents))
        if len(documents) > 2:
            raise RequestEntityTooLargeError(
                "https://example.test/embeddings",
                413,
                b"payload too large",
                httpx.Headers(),
            )
        data = [SimpleNamespace(embedding=[1.0, 1.0]) for _ in documents]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_embeddings_use_large_batches_and_only_sdk_retries(monkeypatch) -> None:
    _FakeEmbeddingsClient.instances.clear()
    monkeypatch.setattr(embeddings_module, "GigaChatAsyncClient", _FakeEmbeddingsClient)

    embedder = GigaChatEmbeddings(
        model_name="test",
        credentials="secret",
        ca_bundle_file=__file__,
    )
    result = await embedder.aembed_texts([f"text-{index}" for index in range(130)])

    client = _FakeEmbeddingsClient.instances[-1]
    assert len(result) == 130
    assert client.batch_sizes == [64, 64, 2]
    assert client.kwargs["max_retries"] == 5


@pytest.mark.asyncio
async def test_oversized_embedding_batch_is_split_without_truncating_all_texts(
    monkeypatch,
) -> None:
    _PayloadLimitedClient.instances.clear()
    monkeypatch.setattr(embeddings_module, "GigaChatAsyncClient", _PayloadLimitedClient)

    embedder = GigaChatEmbeddings(
        model_name="test",
        credentials="secret",
        ca_bundle_file=__file__,
    )
    result = await embedder.aembed_texts(["text"] * 5)

    client = _PayloadLimitedClient.instances[-1]
    assert len(result) == 5
    assert client.batch_sizes == [5, 2, 3, 1, 2]


def test_validation_is_fail_closed() -> None:
    assert (
        PrepareContextDocNode._validation_result(
            "safe question",
            None,
            validation_failed=True,
        )[0]
        is False
    )
    assert (
        PrepareContextDocNode._validation_result(
            "safe question",
            None,
            validation_failed=False,
        )[0]
        is False
    )
    assert (
        PrepareContextDocNode._validation_result(
            "unsafe question",
            DocumentValidationOut(
                context_analysis="unsafe",
                is_jailbreak=True,
                enriched_query="unsafe question",
            ),
            validation_failed=False,
        )[0]
        is False
    )


def test_select_map_segments_reads_document_once(monkeypatch) -> None:
    calls = 0

    def fake_iter(doc: AttachedDocument, *, max_chunk_chars: int):
        nonlocal calls
        calls += 1
        yield from ["segment-0", "segment-1", "segment-2", "segment-3", "segment-4"]

    monkeypatch.setattr(document_utils, "iter_map_chunks_for_document", fake_iter)

    selected = document_utils.select_map_segments(
        AttachedDocument(file_id="file-1", filename="document.txt", file_index=0),
        max_chunk_chars=100,
        max_segments=3,
    )

    assert calls == 1
    assert selected == ["segment-0", "segment-2", "segment-4"]


@pytest.mark.asyncio
async def test_document_disk_helpers_run_outside_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_resolve(doc: AttachedDocument, max_chars: int) -> str:
        worker_threads.append(threading.get_ident())
        return "snippet"

    def fake_select(
        doc: AttachedDocument,
        *,
        max_chunk_chars: int,
        max_segments: int,
        require_full_coverage: bool = False,
    ) -> list[str]:
        worker_threads.append(threading.get_ident())
        return ["segment"]

    monkeypatch.setattr(document_utils, "resolve_document_snippet", fake_resolve)
    monkeypatch.setattr(document_utils, "select_map_segments", fake_select)
    doc = AttachedDocument(file_id="file-1", filename="document.txt", file_index=0)

    assert await document_utils.aresolve_document_snippet(doc, 100) == "snippet"
    assert await document_utils.aselect_map_segments(
        doc,
        max_chunk_chars=100,
        max_segments=10,
    ) == ["segment"]
    assert worker_threads
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)


def _state() -> dict:
    input_message = Message(
        message_id=uuid4(),
        parent_id=None,
        role="user",
        status="success",
        fragments=[
            MessageFragment(
                fragment_id=1,
                fragment_type="request",
                status="success",
                content="question",
            )
        ],
    )
    return {
        "chat_id": "test-chat",
        "message_id": uuid4(),
        "input_message": input_message,
        "output_message": Message(
            message_id=uuid4(),
            parent_id=input_message.message_id,
            role="assistant",
            status="in_progress",
            fragments=[],
        ),
        "relevant_docs": RelevantDocs(),
        "messages": [],
    }


def _discard_event(event: object) -> None:
    return None


@pytest.mark.asyncio
async def test_invalid_request_finishes_with_error_status(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    monkeypatch.setattr(qa_answer, "get_stream_writer", lambda: _discard_event)
    state = _state()
    state.update(
        is_valid_request=False,
        request_error_message="Validation unavailable",
    )

    node = DocumentQAAnswerNode(answer_llm=None)
    await DocumentQAAnswerNode.__call__.__wrapped__(node, state)

    assert state["output_message"].status == "error"
    assert state["output_message"].fragments[-1].status == "error"


@pytest.mark.asyncio
async def test_partial_summary_failure_finishes_with_error_status(monkeypatch) -> None:
    monkeypatch.setattr(base, "get_stream_writer", lambda: _discard_event)
    monkeypatch.setattr(summarize, "get_stream_writer", lambda: _discard_event)
    state = _state()
    state["attached_documents"] = [
        AttachedDocument(file_id="file-1", filename="document.txt", text="content", file_index=0)
    ]

    node = _summarize_node()
    node._summarize_document = AsyncMock(
        return_value=_SummaryResult(body="partial result", failed=True)
    )
    await DocumentSummarizeNode.__call__.__wrapped__(node, state)

    assert state["output_message"].status == "error"
    assert state["output_message"].fragments[-1].status == "error"
