"""Проверяет общий отбор найденных частей и соседнего контекста."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.agents.document_process.models import AttachedDocument, Chunk
from src.agents.document_process.nodes import base, retrieve
from src.agents.document_process.nodes.retrieve import DocumentRetrieveNode
from src.agents.document_process.state import AgentState
from src.common.schemas.models import Message

_SearchResult = tuple[list[tuple[int, str, float]], int, dict[int, str]]


def _document(file_id: str, *, file_index: int = 0, current: bool = True) -> AttachedDocument:
    """Создаёт одноимённое вложение для проверки общего отбора."""
    document = AttachedDocument(
        file_id=file_id,
        filename="договор.txt",
        file_index=file_index,
        char_count=200_000,
        from_current_message=current,
    )
    return document


def _scattered_result(count: int = 30, *, score_start: float = 1.0) -> _SearchResult:
    """Размещает независимые найденные условия между соседними пояснениями."""
    hits = []
    segments = {}
    for rank in range(count):
        index = rank * 3 + 1
        text = f"Самостоятельное условие {rank + 1}."
        score = score_start - rank * 0.01
        hits.append((index, text, score))
        segments[index - 1] = f"Пояснение перед условием {rank + 1}."
        segments[index] = text
        segments[index + 1] = f"Пояснение после условия {rank + 1}."
    result = hits, count * 3, segments
    return result


async def _run_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    documents: list[AttachedDocument],
    search_results: list[_SearchResult],
    *,
    top_k: int = 30,
    core_top_k: int = 20,
) -> list[Chunk]:
    """Запускает настоящий узел с заранее заданными результатами поиска."""
    search = AsyncMock(side_effect=search_results)
    writer = MagicMock()
    writer_factory = MagicMock(return_value=writer)
    monkeypatch.setattr(retrieve, "retrieve_scored_segments", search)
    monkeypatch.setattr(base, "get_stream_writer", writer_factory)
    node = DocumentRetrieveNode(
        segment_max_size=5_000,
        top_k=top_k,
        core_top_k=core_top_k,
        chunk_threshold=150_000,
        full_embed_max_chars=1_000_000,
        keyword_candidate_limit=500,
        embed_batch_size=64,
        fallback_chars=50_000,
    )
    message_id = uuid4()
    output_message = Message(
        message_id=message_id,
        parent_id=None,
        role="assistant",
        status="in_progress",
        fragments=[],
    )
    state: AgentState = {
        "chat_id": "protected-retrieval",
        "output_message": output_message,
        "attached_documents": documents,
        "normalized_user_query": "Какие условия предусмотрены договором?",
    }

    state = await node(state)

    assert search.await_count == len(documents)
    chunks = state["relevant_docs"].document
    return chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("score_start", [1.0, -0.1])
async def test_neighbors_keep_twenty_independent_hits(
    monkeypatch: pytest.MonkeyPatch, score_start: float
) -> None:
    """Сохраняет двадцать найденных условий при обоих знаках оценки поиска."""
    document = _document("file-1")
    search_result = _scattered_result(score_start=score_start)

    chunks = await _run_retrieval(monkeypatch, [document], [search_result])

    found = {chunk.segment_index for chunk in chunks if chunk.selection_reason == "search"}
    neighbors = [chunk for chunk in chunks if chunk.selection_reason == "neighbor"]
    assert found == set(range(1, 60, 3))
    assert len(neighbors) == 10
    assert len(chunks) == 30
    assert "Самостоятельное условие 12." in {chunk.text for chunk in chunks}
    assert {0, 2}.issubset({chunk.segment_index for chunk in neighbors})


@pytest.mark.asyncio
async def test_global_protection_survives_combining_same_named_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сохраняет слабее оценённые условия после объединения разных файлов."""
    current = _document("file-current")
    historical = _document("file-history", file_index=1, current=False)
    hits, total, segments = _scattered_result(count=20)
    ranked_hits = []
    for rank, (index, text, _score) in enumerate(hits):
        score = 1.0 - rank * 0.01 if rank < 5 else 0.4 - rank * 0.01
        ranked_hits.append((index, text, score))
    search_result = ranked_hits, total, segments

    chunks = await _run_retrieval(
        monkeypatch, [current, historical], [search_result, search_result]
    )

    keys = {(chunk.file_id, chunk.segment_index) for chunk in chunks}
    found = [chunk for chunk in chunks if chunk.selection_reason == "search"]
    assert len(chunks) == len(keys) == 30
    assert len(found) == 20
    assert ("file-current", 16) in keys
    assert ("file-history", 16) in keys
    assert all(chunk.from_current_message for chunk in chunks if chunk.file_id == "file-current")
    assert all(
        not chunk.from_current_message for chunk in chunks if chunk.file_id == "file-history"
    )
    assert {chunk.doc_name for chunk in chunks} == {"договор.txt"}


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_neighbors", [0, 1])
async def test_unused_neighbor_slots_return_to_independent_hits(
    monkeypatch: pytest.MonkeyPatch, extra_neighbors: int
) -> None:
    """Заполняет свободные места следующими самостоятельными результатами поиска."""
    document = _document("file-1")
    hits, total, _segments = _scattered_result()
    lookup = {index: text for index, text, _score in hits}
    if extra_neighbors:
        lookup[0] = "Заголовок первого условия."
    search_result = hits, total, lookup

    chunks = await _run_retrieval(monkeypatch, [document], [search_result])

    found = [chunk for chunk in chunks if chunk.selection_reason == "search"]
    neighbors = [chunk for chunk in chunks if chunk.selection_reason == "neighbor"]
    assert len(chunks) == 30
    assert len(found) == 30 - extra_neighbors
    assert len(neighbors) == extra_neighbors
    assert "Самостоятельное условие 25." in {chunk.text for chunk in found}


@pytest.mark.asyncio
async def test_shared_neighbor_is_added_once_at_document_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Добавляет общий сосед один раз и не создаёт части за границами документа."""
    document = _document("file-1")
    hits = [(0, "Первое условие.", 0.9), (2, "Последнее условие.", 0.8)]
    lookup = {0: "Первое условие.", 1: "Общее пояснение.", 2: "Последнее условие."}
    search_result = hits, 3, lookup

    chunks = await _run_retrieval(monkeypatch, [document], [search_result], top_k=4, core_top_k=2)

    reasons = {chunk.segment_index: chunk.selection_reason for chunk in chunks}
    assert len(chunks) == 3
    assert reasons == {0: "search", 1: "neighbor", 2: "search"}


@pytest.mark.asyncio
async def test_neighbor_found_independently_keeps_search_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сохраняет поисковое происхождение соседа без повторного расхода места."""
    document = _document("file-1")
    indices = [1, 10, 20, 2, 3, 11, 12, 21]
    hits = []
    lookup = {index: f"Текст части {index}." for index in range(23)}
    for rank, index in enumerate(indices):
        score = 1.0 - rank * 0.01
        hits.append((index, lookup[index], score))
    search_result = hits, 23, lookup

    chunks = await _run_retrieval(monkeypatch, [document], [search_result], top_k=6, core_top_k=3)

    reasons = {chunk.segment_index: chunk.selection_reason for chunk in chunks}
    assert len(chunks) == 6
    assert set(reasons) == {0, 1, 2, 9, 10, 20}
    assert reasons[2] == "search"
    assert reasons[0] == reasons[9] == "neighbor"


@pytest.mark.asyncio
async def test_few_hits_do_not_prevent_adding_available_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Дополняет редкое попадание соседним заголовком и исключением."""
    document = _document("file-1")
    hits = [(1, "Штраф составляет 100 рублей в день.", 0.9)]
    lookup = {
        0: "Ответственность за просрочку оплаты.",
        1: "Штраф составляет 100 рублей в день.",
        2: "При банковском сбое штраф не начисляется.",
    }
    search_result = hits, 3, lookup

    chunks = await _run_retrieval(monkeypatch, [document], [search_result])

    assert {chunk.segment_index for chunk in chunks} == {0, 1, 2}
    assert len(chunks) == 3
    assert sum(chunk.selection_reason == "search" for chunk in chunks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("document_count", [21, 30, 35])
async def test_each_document_keeps_a_hit_when_total_budget_allows(
    monkeypatch: pytest.MonkeyPatch, document_count: int
) -> None:
    """Сохраняет представленность файлов перед добавлением соседних частей."""
    documents = []
    search_results = []
    for index in range(document_count):
        file_id = f"file-{index}"
        document = _document(file_id, file_index=index)
        score = 1.0 - index * 0.01
        hits = [(1, "Основное условие.", score)]
        lookup = {0: "Заголовок.", 1: "Основное условие.", 2: "Пояснение."}
        search_result = hits, 3, lookup
        documents.append(document)
        search_results.append(search_result)

    chunks = await _run_retrieval(monkeypatch, documents, search_results)

    found = [chunk for chunk in chunks if chunk.selection_reason == "search"]
    expected_count = min(document_count, 30)
    assert len(chunks) == 30
    assert len(found) == expected_count
    assert {chunk.file_id for chunk in found} == {
        f"file-{index}" for index in range(expected_count)
    }
    assert len({(chunk.file_id, chunk.segment_index) for chunk in chunks}) == 30
