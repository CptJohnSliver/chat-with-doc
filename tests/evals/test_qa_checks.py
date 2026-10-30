"""Проверки замера должны ловить ровно то, на что жаловался заказчик.

Тексты ниже — с реальных скриншотов ИФТ 08.08.2026. Если детектор перестанет их
ловить, замер начнёт показывать зелёный отчёт на тех же дефектах.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

QA_DIR = Path(__file__).resolve().parents[2] / "backend" / "evals" / "qa"
sys.path.insert(0, str(QA_DIR))

from checks import CHECKS, run_checks  # noqa: E402


def check(name: str, answer: str, arg=True, case: dict | None = None) -> tuple[bool, str]:
    return CHECKS[name](answer, arg, case or {})


# ─────────────────────────────────────────────────────────────────────────────
# Жаргон и служебная разметка (скриншоты 8–10)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer",
    [
        "Поскольку лицевой токен Карабеков Расим отсутствует в тексте…",
        "Данный идентификатор в тексте не упоминается.",
        "Ответ построен по найденным чанкам документа.",
    ],
)
def test_jargon_is_caught(answer: str) -> None:
    ok, detail = check("no_jargon", answer)
    assert not ok
    assert detail


def test_clean_legal_answer_passes_jargon_check() -> None:
    answer = "Цена квартиры — 14 500 000 рублей. Расчёты производятся через аккредитив."
    assert check("no_jargon", answer)[0]


@pytest.mark.parametrize(
    "answer",
    [
        "В тексте предоставленного фрагмента D1–F1 упоминаются исключительно…",
        "=== ДОКУМЕНТ 1 ===\nНазвание: ДКП",
        "В предоставленных фрагментах исключений не указано.",
        "Согласно фрагменту D2-F3 договор заключён 15.06.2026.",
    ],
)
def test_fragment_markup_is_caught(answer: str) -> None:
    assert not check("no_fragment_markup", answer)[0]


def test_word_fragment_in_normal_sense_is_allowed() -> None:
    answer = "Договор представлен полностью, без утраченных фрагментов текста."
    assert check("no_fragment_markup", answer)[0]


@pytest.mark.parametrize(
    "answer",
    [
        "Текст обезличен, поэтому фамилии заменены.",
        "Персональные данные маскированы системой.",
        "Лицевой токен не удалось расшифровать.",
    ],
)
def test_anonymization_talk_is_caught(answer: str) -> None:
    assert not check("no_anonymization_talk", answer)[0]


def test_tokens_themselves_are_allowed() -> None:
    """[PER_1] в ответе — норма: восстановление идёт ниже по конвейеру."""
    answer = "Продавец — [PER_1], покупатель — [PER_2]."
    assert check("no_anonymization_talk", answer)[0]
    assert check("no_jargon", answer)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Отсутствие сущности и противоречия
# ─────────────────────────────────────────────────────────────────────────────


def test_absence_claim_is_caught() -> None:
    answer = "В предоставленном документе сведений о лицах с обозначением Карабеков Расим нет."
    ok, detail = check("no_absence_claim", answer, ["Карабеков"])
    assert not ok
    assert "Карабеков" in detail


def test_absence_claim_about_other_entity_is_ignored() -> None:
    answer = "Сведений о залоге в документе нет. Продавец — Карабеков Расим."
    assert check("no_absence_claim", answer, ["Карабеков"])[0]


def test_contradiction_within_one_answer_is_caught() -> None:
    answer = (
        "Лицо Карабеков Расим в тексте не упоминается. "
        "Карабеков Расим Мирзеханович указан как продавец. "
        "Таких лиц в документе не найдено."
    )
    ok, detail = check("no_contradiction", answer, ["Карабеков"])
    assert not ok
    assert "противоречие" in detail


def test_consistent_answer_passes_contradiction_check() -> None:
    answer = "Карабеков Расим Мирзеханович указан в договоре как продавец."
    assert check("no_contradiction", answer, ["Карабеков"])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Даты (скриншот 1: две «даты заключения» с разными значениями)
# ─────────────────────────────────────────────────────────────────────────────


def test_conflicting_dates_are_caught() -> None:
    answer = "Дата заключения: 15.06.2026\nДата заключения договора: 25.06.2026"
    ok, detail = check("no_conflicting_dates", answer)
    assert not ok
    assert "15.06.2026" in detail and "25.06.2026" in detail


def test_different_labels_with_different_dates_are_fine() -> None:
    answer = "Дата заключения: 15.06.2026\nДата регистрации права: 20.11.2021"
    assert check("no_conflicting_dates", answer)[0]


def test_invented_dates_are_caught() -> None:
    case = {"_source_text": "Договор от 15.06.2026, право зарегистрировано 20.11.2021."}
    ok, detail = check("no_invented_dates", "Договор заключён 15.06.2026, изменён 01.01.2027.", True, case)
    assert not ok
    assert "01.01.2027" in detail


def test_dates_from_document_pass() -> None:
    case = {"_source_text": "Договор от 15.06.2026."}
    assert check("no_invented_dates", "Договор заключён 15.06.2026.", True, case)[0]


@pytest.mark.parametrize(
    ("source", "answer_date"),
    [
        ("Договор заключён 15 июня 2026 года.", "15.06.2026"),
        ("Двадцатое мая две тысячи двадцать шестого года", "20.05.2026"),
        ("по девятнадцатое мая две тысячи двадцать седьмого года", "19.05.2027"),
        ("Десятое июня две тысячи двадцать шестого года", "10.06.2026"),
        ("двадцать первое марта две тысячи двадцать шестого года", "21.03.2026"),
    ],
)
def test_dates_written_in_words_are_not_inventions(source: str, answer_date: str) -> None:
    """Документы пишут даты словами; пересказ модели цифрами — не выдумка."""
    case = {"_source_text": source}
    assert check("no_invented_dates", f"Дата — {answer_date}.", True, case)[0]


def test_word_dates_do_not_whitewash_a_real_invention() -> None:
    case = {"_source_text": "Договор заключён 15 июня 2026 года."}
    ok, detail = check("no_invented_dates", "Срок истекает 01.01.2030.", True, case)
    assert not ok
    assert "01.01.2030" in detail


# ─────────────────────────────────────────────────────────────────────────────
# Покрытие документов и признание нехватки данных
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_second_document_is_caught() -> None:
    case = {"document_anchors": {"ДКП": ["14 500 000"], "выписка": ["50:20:0030204:312"]}}
    ok, detail = check("covers_documents", "Цена квартиры 14 500 000 рублей.", True, case)
    assert not ok
    assert "выписка" in detail


def test_both_documents_covered() -> None:
    case = {"document_anchors": {"ДКП": ["14 500 000"], "выписка": ["50:20:0030204:312"]}}
    answer = "Цена 14 500 000 рублей; участок 50:20:0030204:312."
    assert check("covers_documents", answer, True, case)[0]


def test_honest_answer_about_missing_document_passes() -> None:
    answer = (
        "Проверить объём полномочий представителя исключительно по тексту договора нельзя, "
        "так как содержание доверенности не представлено."
    )
    assert check("declares_missing_document", answer, ["доверенн"])[0]


def test_confident_answer_without_document_is_caught() -> None:
    answer = "Да, вы вправе подписать договор по этой доверенности."
    assert not check("declares_missing_document", answer, ["доверенн"])[0]


def test_insufficiency_without_naming_the_document_is_caught() -> None:
    """«Установить нельзя» без ответа «что приложить» заказчика не устраивает."""
    ok, detail = check("declares_missing_document", "По имеющимся данным установить нельзя.", ["доверенн"])
    assert not ok
    assert "доверенн" in detail


# ─────────────────────────────────────────────────────────────────────────────
# Сборка проверок
# ─────────────────────────────────────────────────────────────────────────────


def test_hygiene_checks_run_without_being_declared() -> None:
    """Гигиена — требование к любому ответу, кейс её не объявляет."""
    results = run_checks("Согласно фрагменту D1–F1 …", {"checks": {}})
    names = {r["check"] for r in results}
    assert {"no_jargon", "no_fragment_markup", "no_anonymization_talk"} <= names
    assert not next(r for r in results if r["check"] == "no_fragment_markup")["ok"]


def test_case_can_disable_a_default_check() -> None:
    results = run_checks("Согласно фрагменту D1–F1 …", {"checks": {"no_fragment_markup": False}})
    assert "no_fragment_markup" not in {r["check"] for r in results}


def test_unknown_check_fails_loudly() -> None:
    results = run_checks("текст", {"checks": {"нет_такой_проверки": True}})
    unknown = next(r for r in results if r["check"] == "нет_такой_проверки")
    assert unknown == {"check": "нет_такой_проверки", "ok": False, "detail": "неизвестная проверка"}
