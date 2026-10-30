"""Предупреждение о непрочитанных вложениях должно попадать в тело ответа.

Оркестратор забирает только фрагмент `response`; think-фрагменты до него не
доходят. Значит сказать про потерянный файл можно только в самом ответе.
"""

from __future__ import annotations

from src.agents.document_process.nodes.base import with_attachment_warnings


def test_warning_is_appended_to_the_answer() -> None:
    state = {"attachment_warnings": ["Учтены не все приложенные файлы: 1 файл не удалось прочитать."]}

    body = with_attachment_warnings(state, "Цена квартиры — 14 500 000 рублей.")

    assert body.startswith("Цена квартиры")
    assert "1 файл не удалось прочитать" in body


def test_answer_without_warnings_is_untouched() -> None:
    assert with_attachment_warnings({}, "Ответ") == "Ответ"
    assert with_attachment_warnings({"attachment_warnings": []}, "Ответ") == "Ответ"


def test_warning_survives_empty_body() -> None:
    state = {"attachment_warnings": ["Учтены не все приложенные файлы: 1 файл не удалось прочитать."]}

    assert with_attachment_warnings(state, "").startswith("⚠️")
