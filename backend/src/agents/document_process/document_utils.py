from __future__ import annotations

import asyncio
import hashlib
import heapq
from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.infrastructure.document_store import document_local_store
from src.infrastructure.embeddings import GigaChatEmbeddings
from src.infrastructure.logger import get_logger

if TYPE_CHECKING:
    from .models import AttachedDocument, Chunk
    from .state import AgentState


logger = get_logger(__name__)


class MapSegmentLimitExceededError(ValueError):
    def __init__(self, total: int, limit: int) -> None:
        self.total = total
        self.limit = limit
        super().__init__(f"Map segmentation produced {total} windows; limit is {limit}")


def _recursive_splitter(chunk_size: int, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    size = max(1, chunk_size)
    overlap = chunk_overlap if chunk_overlap is not None else min(200, max(0, size // 10))
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def segment_document(text: str, *, max_size: int = 3000) -> list[str]:
    """Split in-memory text with RecursiveCharacterTextSplitter (medium-doc RAG)."""
    if not text or not text.strip():
        return []
    chunks = _recursive_splitter(max_size).split_text(text)
    return [c.strip() for c in chunks if c.strip()] or [text.strip()]


def resolve_document_text(doc: AttachedDocument) -> str:
    """Load the full document text (memory or disk). Prefer for medium docs only."""
    if doc.text:
        return doc.text
    if doc.local_doc_id:
        return document_local_store.read_extracted_text(doc.local_doc_id)
    return ""


def iter_map_chunks_for_document(
    doc: AttachedDocument,
    *,
    max_chunk_chars: int,
) -> Iterator[str]:
    """Yield map windows covering the whole document without loading it all at once."""
    if doc.local_doc_id:
        yield from document_local_store.iter_extracted_text_windows(doc.local_doc_id, max_chars=max_chunk_chars)
        return

    text = doc.text or ""
    max_chunk_chars = max(1, max_chunk_chars)
    for start in range(0, len(text), max_chunk_chars):
        chunk = text[start : start + max_chunk_chars].strip()
        if chunk:
            yield chunk


def select_map_segments(
    doc: AttachedDocument,
    *,
    max_chunk_chars: int,
    max_segments: int,
    require_full_coverage: bool = False,
) -> list[str]:
    """
    Return up to max_segments windows for map-summarize.

    For documents within summarize_max_document_chars this covers the whole text
    (non-overlapping windows). With require_full_coverage, fail instead of silently
    sampling if the configured invariant is violated.
    """
    max_segments = max(1, max_segments)
    segments = list(iter_map_chunks_for_document(doc, max_chunk_chars=max_chunk_chars))
    total = len(segments)
    if not segments:
        return []
    if total <= max_segments:
        return segments
    if require_full_coverage:
        raise MapSegmentLimitExceededError(total, max_segments)
    if max_segments == 1:
        return [segments[0]]

    wanted = {round(i * (total - 1) / (max_segments - 1)) for i in range(max_segments)}
    return [segments[idx] for idx in sorted(wanted)]


async def aselect_map_segments(
    doc: AttachedDocument,
    *,
    max_chunk_chars: int,
    max_segments: int,
    require_full_coverage: bool = False,
) -> list[str]:
    """Select map segments without running disk I/O or splitting on the event loop."""
    return await asyncio.to_thread(
        select_map_segments,
        doc,
        max_chunk_chars=max_chunk_chars,
        max_segments=max_segments,
        require_full_coverage=require_full_coverage,
    )


def _keyword_score(query: str, chunk: str) -> float:
    query_terms = set(query.lower().split())
    chunk_lower = chunk.lower()
    chunk_words = max(len(chunk_lower.split()), 1)
    score = 0.0
    for term in query_terms:
        if len(term) < 2:
            continue
        count = chunk_lower.count(term)
        if count > 0:
            score += count / chunk_words
    return score


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)


async def vector_search(
    query: str,
    chunks: list[str],
    top_k: int = 10,
    *,
    batch_size: int = 64,
) -> list[tuple[int, str, float]]:
    """Embed chunks in batches and keep only top_k similarities (bounded memory)."""
    if not chunks:
        return []
    top_k = max(1, min(top_k, len(chunks)))
    batch_size = max(1, batch_size)

    async with GigaChatEmbeddings(model_name="EmbeddingsGigaR") as embedder:
        query_norm = _normalize_vec(np.array(await embedder.aembed_text(query), dtype=np.float32))

        heap: list[tuple[float, int]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings = await embedder.aembed_texts(batch, batch_size=batch_size)
            for offset, embedding in enumerate(batch_embeddings):
                idx = start + offset
                chunk_norm = _normalize_vec(np.array(embedding, dtype=np.float32))
                similarity = float(np.dot(chunk_norm, query_norm))
                if len(heap) < top_k:
                    heapq.heappush(heap, (similarity, idx))
                elif similarity > heap[0][0]:
                    heapq.heapreplace(heap, (similarity, idx))

    ranked = sorted(heap, key=lambda item: item[0], reverse=True)
    results = [(idx, chunks[idx], similarity) for similarity, idx in ranked]
    logger.debug(f"Vector search: query={query[:50]}..., chunks={len(chunks)}, batch_size={batch_size}, results={len(results)}")
    return results


class _CandidateNeighborTracker:
    """Bounded previous/current/next lookup for candidates kept by the heaps."""

    def __init__(self) -> None:
        self._active: dict[int, dict[int, str]] = {}
        self._first: dict[int, str] = {}
        self._last: dict[int, str] = {}
        self._first_idx: int | None = None
        self._last_idx: int | None = None
        self._previous: tuple[int, str] | None = None

    def before_segment(self, idx: int, text: str) -> None:
        if self._first_idx is None:
            self._first_idx = idx
            self._first[idx] = text
        if self._previous is None:
            return
        previous_idx, _previous_text = self._previous
        if context := self._active.get(previous_idx):
            context[idx] = text
        if previous_idx == self._first_idx:
            self._first[idx] = text

    def after_segment(
        self,
        idx: int,
        text: str,
        *,
        active_indices: set[int],
    ) -> None:
        if idx in active_indices:
            context = {idx: text}
            if self._previous is not None:
                previous_idx, previous_text = self._previous
                context[previous_idx] = previous_text
            self._active[idx] = context
        for stale_idx in self._active.keys() - active_indices:
            del self._active[stale_idx]

        self._last_idx = idx
        self._last = {idx: text}
        if self._previous is not None:
            previous_idx, previous_text = self._previous
            self._last[previous_idx] = previous_text
        self._previous = (idx, text)

    def selected_lookup(
        self,
        selected: list[tuple[int, str, float]],
    ) -> dict[int, str]:
        lookup: dict[int, str] = {}
        for idx, text, _score in selected:
            lookup[idx] = text
            lookup.update(self._active.get(idx, {}))
            if idx == self._first_idx:
                lookup.update(self._first)
            if idx == self._last_idx:
                lookup.update(self._last)
        return lookup


def _keyword_prefilter_segments_with_neighbors(
    query: str,
    segments: Iterator[str],
    *,
    candidate_limit: int,
) -> tuple[list[tuple[int, str, float]], int, dict[int, str]]:
    """
    Stream segments once; keep at most candidate_limit windows.

    Combines top keyword hits with stable priority sampling across the whole
    document. Also retains immediate neighbors for the bounded active candidate
    set, so selected hits do not require a second scan of a disk-backed document.
    """
    limit = max(1, candidate_limit)
    # Half the budget is lexical hits; the rest is deterministic whole-document
    # coverage so paraphrases without shared tokens still get embedded.
    keyword_limit = max(1, limit // 2)
    keyword_heap: list[tuple[float, int, str]] = []
    coverage_heap: list[tuple[int, int, str, float]] = []
    total = 0
    query_seed = query.casefold().encode("utf-8", errors="ignore")
    first_segment: tuple[int, str, float] | None = None
    last_segment: tuple[int, str, float] | None = None
    neighbor_tracker = _CandidateNeighborTracker()

    for idx, segment in enumerate(segments):
        total += 1
        neighbor_tracker.before_segment(idx, segment)
        score = _keyword_score(query, segment)
        current_segment = (idx, segment, score)
        if first_segment is None:
            first_segment = current_segment
        last_segment = current_segment
        if score > 0:
            item = (score, idx, segment)
            if len(keyword_heap) < keyword_limit:
                heapq.heappush(keyword_heap, item)
            elif score > keyword_heap[0][0]:
                heapq.heapreplace(keyword_heap, item)

        # Stable priority sampling provides bounded, reproducible coverage of
        # the entire stream without retaining every segment in memory.
        digest = hashlib.blake2b(
            query_seed + idx.to_bytes(8, "big") + segment.encode("utf-8", errors="ignore"),
            digest_size=8,
        ).digest()
        priority = int.from_bytes(digest, "big")
        coverage_item = (-priority, idx, segment, score)
        if len(coverage_heap) < limit:
            heapq.heappush(coverage_heap, coverage_item)
        elif priority < -coverage_heap[0][0]:
            heapq.heapreplace(coverage_heap, coverage_item)

        neighbor_tracker.after_segment(
            idx,
            segment,
            active_indices={item[1] for item in keyword_heap} | {item[1] for item in coverage_heap},
        )

    keyword_ranked = sorted(keyword_heap, key=lambda item: (item[0], -item[1]), reverse=True)
    selected: list[tuple[int, str, float]] = []
    seen: set[int] = set()
    for score, idx, text in keyword_ranked:
        selected.append((idx, text, score))
        seen.add(idx)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        coverage = sorted(
            ((idx, text, score) for _priority, idx, text, score in coverage_heap),
            key=lambda item: item[0],
        )
        edge_coverage = [edge for edge in (first_segment, last_segment) if edge is not None]
        for idx, text, score in [*edge_coverage, *coverage]:
            if idx in seen:
                continue
            selected.append((idx, text, score))
            seen.add(idx)
            if len(selected) >= limit:
                break

    selected.sort(key=lambda item: item[0])
    return selected, total, neighbor_tracker.selected_lookup(selected)


def keyword_prefilter_segments(
    query: str,
    segments: Iterator[str],
    *,
    candidate_limit: int,
) -> tuple[list[tuple[int, str, float]], int]:
    """Public prefilter API without the internal adjacent-segment lookup."""
    selected, total, _neighbors = _keyword_prefilter_segments_with_neighbors(
        query,
        segments,
        candidate_limit=candidate_limit,
    )
    return selected, total


def _keyword_fallback_hits(
    candidates: list[tuple[int, str, float]],
    *,
    limit: int,
) -> list[tuple[int, str, float]]:
    """Prefer lexical hits, then add deterministic whole-document coverage."""
    limit = max(1, min(limit, len(candidates)))
    ranked = sorted(candidates, key=lambda item: (-item[2], item[0]))
    selected = [candidate for candidate in ranked if candidate[2] > 0][:limit]
    seen = {idx for idx, _text, _score in selected}

    if len(selected) < limit:
        remaining = [candidate for candidate in candidates if candidate[0] not in seen]
        slots = limit - len(selected)
        positions = [0] if slots == 1 else [round(index * (len(remaining) - 1) / (slots - 1)) for index in range(slots)]
        for position in positions:
            candidate = remaining[position]
            if candidate[0] in seen:
                continue
            selected.append(candidate)
            seen.add(candidate[0])

    return selected


def _neighbor_texts(
    hits: list[tuple[int, str, float]],
    segment_lookup: dict[int, str],
) -> dict[int, str]:
    """Сохраняет только непосредственных соседей найденных частей."""
    neighbors: dict[int, str] = {}
    for index, _text, _score in hits:
        for neighbor_index in (index - 1, index + 1):
            text = segment_lookup.get(neighbor_index)
            if text:
                neighbors[neighbor_index] = text
    return neighbors


async def retrieve_scored_segments(
    doc: AttachedDocument,
    query: str,
    *,
    max_segment_chars: int,
    keyword_candidate_limit: int,
    embed_batch_size: int,
    top_k: int,
    full_embed_max_chars: int,
) -> tuple[list[tuple[int, str, float]], int, dict[int, str]]:
    """Находит основные части документа и сохраняет соседний текст для общего отбора."""

    char_count = document_char_count(doc)
    use_full_embed = char_count <= max(0, full_embed_max_chars)

    if use_full_embed:

        def _split_all() -> tuple[
            list[tuple[int, str, float]],
            int,
            dict[int, str],
        ]:
            text = resolve_document_text(doc)
            segments = segment_document(text, max_size=max_segment_chars)
            candidates = [(idx, segment, _keyword_score(query, segment)) for idx, segment in enumerate(segments)]
            return candidates, len(segments), dict(enumerate(segments))

        candidates, total_segments, adjacent_segments = await asyncio.to_thread(_split_all)
    else:

        def _prefilter() -> tuple[
            list[tuple[int, str, float]],
            int,
            dict[int, str],
        ]:
            segments = iter_map_chunks_for_document(doc, max_chunk_chars=max_segment_chars)
            return _keyword_prefilter_segments_with_neighbors(
                query,
                segments,
                candidate_limit=keyword_candidate_limit,
            )

        # Disk scan + keyword scoring are sync/CPU — keep them off the event loop.
        candidates, total_segments, adjacent_segments = await asyncio.to_thread(_prefilter)

    if not candidates:
        return [], total_segments, {}

    seed_limit = max(1, top_k)
    candidate_texts = [text for _idx, text, _score in candidates]
    try:
        vector_hits = await vector_search(
            query,
            candidate_texts,
            top_k=seed_limit,
            batch_size=embed_batch_size,
        )
    except Exception as exc:
        await logger.awarning(
            "Embedding retrieval failed; using keyword/coverage fallback",
            error_class=type(exc).__name__,
            detail=str(exc),
            candidates=len(candidates),
        )
        keyword_hits = _keyword_fallback_hits(candidates, limit=seed_limit)
        neighbors = _neighbor_texts(keyword_hits, adjacent_segments)
        return keyword_hits, total_segments, neighbors

    scored: list[tuple[int, str, float]] = []
    for candidate_index, text, similarity in vector_hits:
        if similarity <= 0:
            continue
        idx, _text, _kw = candidates[candidate_index]
        scored.append((idx, text, similarity))
    if not scored and vector_hits:
        # Soft fallback: keep best cosine matches even when all are ≤ 0 so we do
        # not silently drop to a document-prefix stub after spending embed budget.
        for candidate_index, text, similarity in vector_hits:
            idx, _text, _kw = candidates[candidate_index]
            scored.append((idx, text, similarity))
    neighbors = _neighbor_texts(scored, adjacent_segments)
    retrieval = scored, total_segments, neighbors
    return retrieval


def get_attached_documents(state: AgentState) -> list[AttachedDocument]:
    return list(state.get("attached_documents") or [])


def documents_in_focus(state: AgentState) -> list[AttachedDocument]:
    """Документы, про которые идёт речь: приложенные к текущему вопросу.

    Для краткого содержания это единственно верный набор. Пересказ документа
    из прошлого хода пользователь уже видел, а получить хотел про тот, который
    только что приложил, — на это и жаловались.

    Когда в текущем сообщении файлов нет (вопрос по ранее загруженному), история
    остаётся единственным источником, и берутся все документы.
    """
    documents = get_attached_documents(state)
    current = [document for document in documents if document.from_current_message]
    return current or documents


def document_char_count(doc: AttachedDocument) -> int:
    if doc.char_count:
        return doc.char_count
    if doc.text:
        return len(doc.text)
    if doc.local_doc_id:
        return document_local_store.extracted_byte_size(doc.local_doc_id)
    return 0


def resolve_document_snippet(doc: AttachedDocument, max_chars: int) -> str:
    if doc.text:
        return doc.text[:max_chars].strip()
    if doc.local_doc_id:
        return document_local_store.read_extracted_snippet(doc.local_doc_id, max_chars)
    return ""


async def aresolve_document_snippet(doc: AttachedDocument, max_chars: int) -> str:
    """Read a document snippet without blocking the event loop."""
    return await asyncio.to_thread(resolve_document_snippet, doc, max_chars)


def resolve_no_hit_fallback(
    doc: AttachedDocument,
    *,
    full_embed_max_chars: int,
    fallback_chars: int,
) -> str:
    """
    Context when retrieve found no usable chunks.

    Medium docs (≤ full_embed_max_chars): full text (main parity / embeddable size).
    Larger docs: beginning + middle + end regions so the answer is not stuck on the prefix.
    """
    char_count = document_char_count(doc)
    if char_count <= max(0, full_embed_max_chars):
        return resolve_document_snippet(doc, full_embed_max_chars)

    region = max(1, fallback_chars)
    if doc.text:
        text = doc.text
        if len(text) <= region * 3:
            return text.strip()
        mid_start = max(0, (len(text) - region) // 2)
        parts = [
            text[:region],
            text[mid_start : mid_start + region],
            text[-region:],
        ]
        return "\n\n...\n\n".join(part.strip() for part in parts if part.strip())

    if not doc.local_doc_id:
        return ""

    head = document_local_store.read_extracted_snippet(doc.local_doc_id, region)
    # Re-read size under a lease for mid/tail offsets via window iteration.
    windows = list(document_local_store.iter_extracted_text_windows(doc.local_doc_id, max_chars=region))
    if not windows:
        return head
    if len(windows) == 1:
        return windows[0]
    mid = windows[len(windows) // 2]
    tail = windows[-1]
    parts = [head or windows[0], mid, tail]
    # Deduplicate if the file is barely larger than one region.
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part[:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return "\n\n...\n\n".join(unique)


async def aresolve_no_hit_fallback(
    doc: AttachedDocument,
    *,
    full_embed_max_chars: int,
    fallback_chars: int,
) -> str:
    return await asyncio.to_thread(
        resolve_no_hit_fallback,
        doc,
        full_embed_max_chars=full_embed_max_chars,
        fallback_chars=fallback_chars,
    )


#: Подписи происхождения документа в контексте модели.
SOURCE_CURRENT = "приложен к текущему вопросу"
SOURCE_HISTORY = "приложен ранее в этом диалоге"

_SELECTION_LABELS = {
    "search": "найдено поиском",
    "neighbor": "соседняя часть найденного текста",
    "full": "полный текст документа",
    "fallback": "резервное чтение документа",
}


def _chunk_position(chunk: Chunk) -> int:
    """Определяет порядок частей по исходным номерам."""
    position = chunk.segment_index if chunk.segment_index is not None else -1
    return position


def build_document_context(doc_chunks: list[Chunk]) -> str:
    """Группирует части по идентификатору файла и собирает контекст для модели."""
    grouped: OrderedDict[str, list[Chunk]] = OrderedDict()
    for chunk in doc_chunks:
        grouped.setdefault(chunk.file_id, []).append(chunk)

    origins = {bool(getattr(chunk, "from_current_message", True)) for chunk in doc_chunks}
    mark_source = len(origins) > 1

    parts: list[str] = []
    for doc_idx, chunks in enumerate(grouped.values(), 1):
        chunks.sort(key=_chunk_position)
        doc_name = chunks[0].doc_name or "Документ"
        lines = [f"=== ДОКУМЕНТ {doc_idx} ===", f"Название: {doc_name}"]
        if mark_source:
            is_current = any(getattr(chunk, "from_current_message", True) for chunk in chunks)
            lines.append(f"Источник: {SOURCE_CURRENT if is_current else SOURCE_HISTORY}")
        lines.append("")
        for frag_idx, chunk in enumerate(chunks, 1):
            lines.append(f"--- ФРАГМЕНТ D{doc_idx}-F{frag_idx} ---")
            if chunk.segment_index is not None:
                lines.append(f"Номер части в исходном тексте: {chunk.segment_index + 1}")
            lines.append(f"Причина включения: {_SELECTION_LABELS[chunk.selection_reason]}")
            lines.append("Текст:")
            lines.append(chunk.text)
            lines.append("")
        parts.append("\n".join(lines))

    context = "\n".join(parts)
    return context
