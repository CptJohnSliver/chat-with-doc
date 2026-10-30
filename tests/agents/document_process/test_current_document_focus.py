"""Ответ и саммари идут по документу текущего вопроса, а не по документу из истории.

Жалоба ИФТ: пользователь загрузил договор подряда, получил саммари, затем приложил
доверенность со словами «проверь риски» — и получил ответ про договор подряда.
Механика была такая: к документу текущего хода подмешиваются документы прошлых ходов
(`history_documents_limit`), в контексте они неразличимы, а рисков в доверенности нет —
модель отвечала по тому документу, где нашла ответ.
"""

from __future__ import annotations

import pytest

from src.agents.document_process.document_utils import (
    SOURCE_CURRENT,
    SOURCE_HISTORY,
    build_document_context,
    documents_in_focus,
)
from src.agents.document_process.models import AttachedDocument, Chunk


def doc(file_id: str, filename: str, *, current: bool) -> AttachedDocument:
    """Создаёт вложение с заданным происхождением."""
    document = AttachedDocument(
        file_id=file_id,
        filename=filename,
        file_index=0,
        char_count=10,
        from_current_message=current,
    )
    return document


class TestDocumentContext:
    def test_mixed_origins_are_marked(self) -> None:
        context = build_document_context(
            [
                Chunk(file_id="file-1", doc_name="доверенность.pdf", text="Доверенность", from_current_message=True),
                Chunk(file_id="file-2", doc_name="подряд.pdf", text="Договор подряда", from_current_message=False),
            ]
        )

        assert f"Название: доверенность.pdf\nИсточник: {SOURCE_CURRENT}" in context
        assert f"Название: подряд.pdf\nИсточник: {SOURCE_HISTORY}" in context

    def test_documents_of_one_origin_are_not_marked(self) -> None:
        """Выбирать не из чего — разметка была бы лишним шумом в контексте."""
        context = build_document_context(
            [
                Chunk(file_id="file-1", doc_name="первый.pdf", text="раз", from_current_message=True),
                Chunk(file_id="file-2", doc_name="второй.pdf", text="два", from_current_message=True),
            ]
        )

        assert "Источник:" not in context

    def test_chunks_without_origin_behave_as_before(self) -> None:
        """Совместимость: кусок без признака — обычный документ текущего хода."""
        chunk = Chunk(file_id="file-1", doc_name="файл.pdf", text="текст")
        context = build_document_context([chunk])

        assert "Источник:" not in context
        assert "Название: файл.pdf" in context

    @pytest.mark.parametrize("filename", ["договор.txt", ""])
    @pytest.mark.parametrize("second_is_current", [False, True])
    def test_same_names_keep_documents_separate(
        self, filename: str, second_is_current: bool
    ) -> None:
        """Разделяет одноимённые документы и сохраняет происхождение их условий."""
        current_chunk = Chunk(
            file_id="file-current",
            doc_name=filename,
            text="Оплата в течение 5 дней.",
            from_current_message=True,
        )
        second_chunk = Chunk(
            file_id="file-second",
            doc_name=filename,
            text="Оплата в течение 30 дней.",
            from_current_message=second_is_current,
        )

        context = build_document_context([current_chunk, second_chunk])

        current_block, second_block = context.split("=== ДОКУМЕНТ 2 ===")
        assert context.count("=== ДОКУМЕНТ ") == 2
        assert "5 дней" in current_block
        assert "30 дней" not in current_block
        assert "30 дней" in second_block
        assert "5 дней" not in second_block
        assert "D1-F1" in current_block
        assert "D2-F1" in second_block
        if second_is_current:
            assert "Источник:" not in context
        else:
            assert f"Источник: {SOURCE_CURRENT}" in current_block
            assert f"Источник: {SOURCE_HISTORY}" in second_block

    def test_interleaved_chunks_of_one_file_stay_together(self) -> None:
        """Собирает части одного файла вместе при чередовании результатов поиска."""
        first_chunk = Chunk(file_id="file-1", doc_name="договор.txt", text="Первое условие.")
        other_chunk = Chunk(file_id="file-2", doc_name="договор.txt", text="Другой договор.")
        last_chunk = Chunk(file_id="file-1", doc_name="договор.txt", text="Второе условие.")

        context = build_document_context([first_chunk, other_chunk, last_chunk])

        first_block, second_block = context.split("=== ДОКУМЕНТ 2 ===")
        assert context.count("=== ДОКУМЕНТ ") == 2
        assert "D1-F1" in first_block
        assert "D1-F2" in first_block
        assert "Первое условие." in first_block
        assert "Второе условие." in first_block
        assert "Другой договор." not in first_block
        assert "Другой договор." in second_block

    def test_source_positions_and_reasons_follow_document_order(self) -> None:
        """Показывает исходную последовательность и причину включения частей."""
        later = Chunk(
            file_id="file-1",
            doc_name="договор.txt",
            text="Позднее пояснение.",
            segment_index=47,
            selection_reason="neighbor",
        )
        earlier = Chunk(
            file_id="file-1",
            doc_name="договор.txt",
            text="Ранее найденное условие.",
            segment_index=3,
            selection_reason="search",
        )

        context = build_document_context([later, earlier])

        first_fragment, second_fragment = context.split("--- ФРАГМЕНТ D1-F2 ---")
        assert "Номер части в исходном тексте: 4" in first_fragment
        assert "Причина включения: найдено поиском" in first_fragment
        assert "Ранее найденное условие." in first_fragment
        assert "Позднее пояснение." not in first_fragment
        assert "Номер части в исходном тексте: 48" in second_fragment
        assert "Причина включения: соседняя часть найденного текста" in second_fragment
        assert "Позднее пояснение." in second_fragment

    @pytest.mark.parametrize("reason", ["full", "fallback"])
    def test_context_does_not_invent_unknown_source_positions(self, reason: str) -> None:
        """Не присваивает номер части тексту без известной позиции в нарезке."""
        chunk = Chunk(
            file_id="file-1",
            doc_name="договор.txt",
            text="Текст без известного номера части.",
            selection_reason=reason,
        )

        context = build_document_context([chunk])

        assert "Номер части в исходном тексте:" not in context
        assert "Текст без известного номера части." in context


class TestSummaryFocus:
    def test_summary_covers_only_the_freshly_attached_document(self) -> None:
        """Пересказ документа из прошлого хода пользователь уже видел."""
        documents = [
            doc("file-1", "доверенность.pdf", current=True),
            doc("file-2", "подряд.pdf", current=False),
        ]

        assert [item.filename for item in documents_in_focus({"attached_documents": documents})] == ["доверенность.pdf"]

    def test_without_fresh_attachments_history_is_the_only_source(self) -> None:
        documents = [doc("file-1", "подряд.pdf", current=False)]

        assert documents_in_focus({"attached_documents": documents}) == documents

    def test_no_documents_at_all(self) -> None:
        assert documents_in_focus({"attached_documents": []}) == []
