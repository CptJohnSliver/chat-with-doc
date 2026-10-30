"""Отбирает найденные части документов и соседний контекст."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.agents.document_process.document_utils import (
    aresolve_document_snippet,
    aresolve_no_hit_fallback,
    document_char_count,
    get_attached_documents,
    retrieve_scored_segments,
)
from src.agents.document_process.models import AttachedDocument, Chunk, RelevantDocs
from src.agents.document_process.nodes.base import BaseNode, _emit_ui_event, _next_oid
from src.agents.document_process.state import AgentState
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_EMPTY_DOCUMENT_MESSAGE = "Документ не содержит текста для поиска."


@dataclass(frozen=True)
class _SearchHit:
    """Связывает оценённую часть с исходным документом."""

    document: AttachedDocument
    segment_index: int
    text: str
    score: float

    @property
    def key(self) -> tuple[str, int]:
        """Определяет часть по файлу и исходному номеру."""
        key = self.document.file_id, self.segment_index
        return key

    def as_chunk(self) -> Chunk:
        """Сохраняет положение и происхождение найденной части."""
        chunk = Chunk(
            file_id=self.document.file_id,
            doc_name=self.document.display_name,
            text=self.text,
            segment_index=self.segment_index,
            selection_reason="search",
            from_current_message=self.document.from_current_message,
        )
        return chunk


def _hit_score(hit: _SearchHit) -> float:
    """Извлекает собственную оценку найденной части для сортировки."""
    score = hit.score
    return score


class DocumentRetrieveNode(BaseNode):
    """Собирает контекст с защищёнными попаданиями и отдельным лимитом соседей."""

    def __init__(
        self,
        *,
        segment_max_size: int,
        top_k: int,
        core_top_k: int,
        chunk_threshold: int,
        full_embed_max_chars: int,
        keyword_candidate_limit: int,
        embed_batch_size: int,
        fallback_chars: int,
    ):
        """Задаёт ограничения поиска и общего отбора частей."""
        super().__init__("document_retrieve")
        self.segment_max_size = segment_max_size
        self.top_k = max(1, top_k)
        bounded_core_top_k = min(core_top_k, self.top_k)
        self.core_top_k = max(1, bounded_core_top_k)
        self.chunk_threshold = chunk_threshold
        self.full_embed_max_chars = full_embed_max_chars
        self.keyword_candidate_limit = keyword_candidate_limit
        self.embed_batch_size = embed_batch_size
        self.fallback_chars = fallback_chars

    def _select_core_hits(
        self,
        ranked: list[_SearchHit],
        best_per_doc: list[_SearchHit],
    ) -> list[_SearchHit]:
        """Защищает представителей файлов и добирает лучшие самостоятельные попадания."""
        protected = best_per_doc[: self.top_k]
        seen = {hit.key for hit in protected}
        for hit in ranked:
            if len(protected) >= self.core_top_k:
                break
            if hit.key not in seen:
                protected.append(hit)
                seen.add(hit.key)
        protected.sort(key=_hit_score, reverse=True)
        return protected

    def _select_neighbors(
        self,
        protected: list[_SearchHit],
        core_by_key: dict[tuple[str, int], _SearchHit],
        adjacent: dict[tuple[str, int], str],
        seen: set[tuple[str, int]],
    ) -> list[Chunk]:
        """Добавляет соседей защищённых попаданий без повторов и дальнейшего расширения."""
        neighbors: list[Chunk] = []
        remaining_slots = self.top_k - len(seen)
        limit = min(self.top_k - self.core_top_k, remaining_slots)
        if limit <= 0:
            return neighbors
        for center in protected:
            for index in (center.segment_index - 1, center.segment_index + 1):
                key = center.document.file_id, index
                text = adjacent.get(key)
                if not text or key in seen:
                    continue
                core = core_by_key.get(key)
                if core is not None:
                    chunk = core.as_chunk()
                else:
                    chunk = Chunk(
                        file_id=center.document.file_id,
                        doc_name=center.document.display_name,
                        text=text,
                        segment_index=index,
                        selection_reason="neighbor",
                        from_current_message=center.document.from_current_message,
                    )
                neighbors.append(chunk)
                seen.add(key)
                if len(neighbors) >= limit:
                    return neighbors
        return neighbors

    def _select_chunks(
        self,
        ranked: list[_SearchHit],
        best_per_doc: list[_SearchHit],
        adjacent: dict[tuple[str, int], str],
    ) -> list[Chunk]:
        """Распределяет общий лимит между попаданиями и соседями всех файлов."""
        protected = self._select_core_hits(ranked, best_per_doc)
        selected = [hit.as_chunk() for hit in protected]
        seen = {hit.key for hit in protected}
        core_by_key = {hit.key: hit for hit in ranked}
        neighbors = self._select_neighbors(protected, core_by_key, adjacent, seen)
        selected.extend(neighbors)
        for hit in ranked:
            if len(selected) >= self.top_k:
                return selected
            if hit.key not in seen:
                chunk = hit.as_chunk()
                selected.append(chunk)
                seen.add(hit.key)
        return selected

    async def _retrieve_chunks(self, docs: list[AttachedDocument], question: str) -> tuple[list[Chunk], int, bool]:
        """Собирает результаты поиска по файлам для единого окончательного отбора."""
        ranked: list[_SearchHit] = []
        best_per_doc: list[_SearchHit] = []
        adjacent: dict[tuple[str, int], str] = {}
        total_segments = 0
        used_full_embed = False
        for doc in docs:
            scored, count, segment_lookup = await retrieve_scored_segments(
                doc,
                question,
                max_segment_chars=self.segment_max_size,
                keyword_candidate_limit=self.keyword_candidate_limit,
                embed_batch_size=self.embed_batch_size,
                top_k=self.top_k,
                full_embed_max_chars=self.full_embed_max_chars,
            )
            total_segments += count
            used_full_embed = used_full_embed or document_char_count(doc) <= self.full_embed_max_chars
            doc_hits: list[_SearchHit] = []
            for index, text, score in scored:
                hit = _SearchHit(document=doc, segment_index=index, text=text, score=score)
                doc_hits.append(hit)
            doc_hits.sort(key=_hit_score, reverse=True)
            if doc_hits:
                best_per_doc.append(doc_hits[0])
            ranked.extend(doc_hits)
            for index, text in segment_lookup.items():
                adjacent[doc.file_id, index] = text
        ranked.sort(key=_hit_score, reverse=True)
        best_per_doc.sort(key=_hit_score, reverse=True)
        chunks = self._select_chunks(ranked, best_per_doc, adjacent)
        retrieval = chunks, total_segments, used_full_embed
        return retrieval

    async def _non_empty_documents(
        self,
        state: AgentState,
    ) -> tuple[list[AttachedDocument], list[AttachedDocument]]:
        attached_docs = get_attached_documents(state)
        docs = [doc for doc in attached_docs if document_char_count(doc) > 0]
        if len(docs) != len(attached_docs):
            await logger.awarning(
                "DocumentRetrieve skipped empty documents",
                skipped=len(attached_docs) - len(docs),
            )
        return attached_docs, docs

    def _finish_without_attachment(
        self,
        state: AgentState,
        *,
        ordering_id: int,
        t0: float,
    ) -> AgentState:
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="success",
            content="### Документ не загружен",
            t0=t0,
        )
        return state

    def _finish_with_empty_documents(
        self,
        state: AgentState,
        *,
        ordering_id: int,
        t0: float,
    ) -> AgentState:
        relevant_docs = state.get("relevant_docs") or RelevantDocs()
        relevant_docs.document = []
        state["relevant_docs"] = relevant_docs
        state["is_valid_request"] = False
        state["request_error_message"] = _EMPTY_DOCUMENT_MESSAGE
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="error",
            content="### Документ не содержит текста",
            t0=t0,
        )
        return state

    async def __call__(self, state: AgentState) -> AgentState:
        """Находит части документов и сохраняет контекст для ответа."""
        t0 = time.perf_counter()
        ordering_id = _next_oid(state)
        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="in_progress",
            content="### Ищу по документу...",
        )

        attached_docs, docs = await self._non_empty_documents(state)
        question = state.get("enriched_user_query") or state.get("normalized_user_query") or ""

        if not attached_docs:
            return self._finish_without_attachment(
                state,
                ordering_id=ordering_id,
                t0=t0,
            )
        if not docs:
            return self._finish_with_empty_documents(
                state,
                ordering_id=ordering_id,
                t0=t0,
            )

        relevant_chunks: list[Chunk] = []
        total_segments = 0

        needs_chunking = any(document_char_count(doc) > self.chunk_threshold for doc in docs)

        if needs_chunking:
            relevant_chunks, total_segments, used_full_embed = await self._retrieve_chunks(docs, question)

            mode = "full-embed" if used_full_embed else "keyword-prefilter"
            await logger.ainfo(
                f"DocumentRetrieve (chunked/{mode}): {total_segments} segments scanned, "
                f"keyword≤{self.keyword_candidate_limit}, top_k={self.top_k}, "
                f"relevant={len(relevant_chunks)}"
            )

            if not relevant_chunks:
                for doc in docs:
                    doc_text = await aresolve_no_hit_fallback(
                        doc,
                        full_embed_max_chars=self.full_embed_max_chars,
                        fallback_chars=self.fallback_chars,
                    )
                    if doc_text.strip():
                        chunk = Chunk(
                            file_id=doc.file_id,
                            doc_name=doc.display_name,
                            text=doc_text,
                            selection_reason="fallback",
                            from_current_message=doc.from_current_message,
                        )
                        relevant_chunks.append(chunk)
        else:
            for doc in docs:
                doc_text = await aresolve_document_snippet(doc, self.chunk_threshold)
                if doc_text.strip():
                    chunk = Chunk(
                        file_id=doc.file_id,
                        doc_name=doc.display_name,
                        text=doc_text,
                        selection_reason="full",
                        from_current_message=doc.from_current_message,
                    )
                    relevant_chunks.append(chunk)

            await logger.ainfo(f"DocumentRetrieve (full): document size ≤ {self.chunk_threshold}, passing full text to LLM")

        relevant_docs = state.get("relevant_docs") or RelevantDocs()
        relevant_docs.document = relevant_chunks
        state["relevant_docs"] = relevant_docs

        _emit_ui_event(
            state,
            self.step,
            ordering_id,
            status="success",
            content="### Поиск по документу завершён\n\n---",
            t0=t0,
        )
        return state
