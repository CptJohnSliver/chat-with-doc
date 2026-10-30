"""Детерминированные проверки ответов chat_w_doc.

Ни одна проверка не зовёт LLM: замер должен быть воспроизводимым и дешёвым.
Каждая проверка возвращает (ok, detail); detail попадает в отчёт, поэтому пишем
в него то, что видно глазами — найденную строку, а не «не прошло».
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from typing import Any

Result = tuple[bool, str]

# ─────────────────────────────────────────────────────────────────────────────
# Словари дефектов из ИФТ 08.08.2026
# ─────────────────────────────────────────────────────────────────────────────

# Внутренняя кухня модели, попавшая в ответ юристу (скриншоты 8–10).
JARGON = (
    "токен",
    "placeholder",
    "плейсхолдер",
    "идентификатор",
    "чанк",
    "эмбеддинг",
    "промпт",
    "контекстное окно",
)

# Разметка контекста из answer_document.yaml, протекающая в ответ.
FRAGMENT_MARKUP = (
    re.compile(r"===\s*ДОКУМЕНТ\s*\d*\s*==="),
    re.compile(r"---\s*ФРАГМЕНТ", re.IGNORECASE),
    re.compile(r"фрагмент[а-яё]*\s+[A-ZА-Я]?\d+\s*[–—-]\s*[A-ZА-Я]?\d+", re.IGNORECASE),
    re.compile(r"\bD\d+\s*[–—-]\s*F\d+\b"),
    re.compile(r"в\s+предоставленн[а-яё]+\s+фрагмент", re.IGNORECASE),
)

# Разговоры про обезличивание. Сами токены [PER_1] в ответе допустимы —
# они восстанавливаются ниже по конвейеру; недопустимо их обсуждать.
ANONYMIZATION_TALK = (
    re.compile(r"обезлич", re.IGNORECASE),
    re.compile(r"маскир", re.IGNORECASE),
    re.compile(r"анонимиз", re.IGNORECASE),
    re.compile(r"лицев[а-яё]+\s+токен", re.IGNORECASE),
    re.compile(r"замен[а-яё]*\s+персональн", re.IGNORECASE),
)

# Утверждения об отсутствии сущности в документе. Только явное отрицание: оборот
# «сведений о …» сам по себе не отрицание, он бывает и в «сверьте сведения о лице».
ABSENCE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"не упомина",
        r"не упомянут",
        r"отсутству",
        r"не найден",
        r"не содержится",
        r"не указан",
        r"не значится",
        r"нет сведени",
        r"сведени[йя][^.]{0,80}?\b(?:нет|не найден|не содерж|отсутству)",
    )
)

DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")

# Даты в документах пишут словами — «15 июня 2026 года», «Двадцатое мая две тысячи
# двадцать шестого года». Без разбора этих форм проверка «дата не из документа»
# ловит собственный пересказ модели вместо выдумки.
_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_DAY_UNITS = {
    "перв": 1, "втор": 2, "трет": 3, "четвёрт": 4, "четверт": 4, "пят": 5, "шест": 6,
    "седьм": 7, "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12,
    "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16,
    "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20,
    "тридцат": 30,
}
_YEAR_UNITS = {
    "двадцат": 20, "двадцать перв": 21, "двадцать втор": 22, "двадцать трет": 23,
    "двадцать четвёрт": 24, "двадцать четверт": 24, "двадцать пят": 25,
    "двадцать шест": 26, "двадцать седьм": 27, "двадцать восьм": 28,
    "двадцать девят": 29, "тридцат": 30,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_ALT = "|".join(sorted(_DAY_UNITS, key=len, reverse=True))
_YEAR_ALT = "|".join(sorted(_YEAR_UNITS, key=len, reverse=True))

_DIGIT_DAY_DATE_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})[а-яё]*\s+(\d{{4}})", re.IGNORECASE)
_WORD_DATE_RE = re.compile(
    rf"\b(?:(?:двадцать|тридцать)\s+)?({_DAY_ALT})[а-яё]*\s+({_MONTH_ALT})[а-яё]*\s+"
    rf"дв(?:е|ух)\s+тысяч[а-яё]*\s+({_YEAR_ALT})[а-яё]*",
    re.IGNORECASE,
)
_COMPOUND_DAY_RE = re.compile(rf"\b(двадцать|тридцать)\s+({_DAY_ALT})[а-яё]*\s+(?:{_MONTH_ALT})", re.IGNORECASE)
LABELLED_DATE_RE = re.compile(r"([А-ЯЁа-яё][А-ЯЁа-яё \-]{2,48}?)\s*[:—–-]\s*(\d{2}\.\d{2}\.\d{4})")

_DATE_LABEL_NOISE = re.compile(r"\b(документа|договора|настоящего|указанн[а-яё]+|данного|его|№)\b", re.IGNORECASE)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _norm_label(label: str) -> str:
    cleaned = _DATE_LABEL_NOISE.sub(" ", _norm(label))
    return " ".join(cleaned.split())


def _known_dates(source: str) -> set[str]:
    """Все даты документа в виде ДД.ММ.ГГГГ, включая записанные словами."""
    dates = set(DATE_RE.findall(source))

    for day, month, year in _DIGIT_DAY_DATE_RE.findall(source):
        dates.add(f"{int(day):02d}.{_MONTHS[month.casefold()]:02d}.{year}")

    for match in _WORD_DATE_RE.finditer(source):
        day_word, month_word, year_word = (part.casefold() for part in match.groups())
        day = _DAY_UNITS[day_word]
        prefix = _COMPOUND_DAY_RE.match(match.group(0))
        if prefix:  # «двадцать первое», «тридцать первое»
            day = (20 if prefix.group(1).casefold() == "двадцать" else 30) + _DAY_UNITS[prefix.group(2).casefold()]
        year = 2000 + _YEAR_UNITS[year_word]
        dates.add(f"{day:02d}.{_MONTHS[month_word]:02d}.{year}")

    return dates


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", text) if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Проверки
# ─────────────────────────────────────────────────────────────────────────────


def no_jargon(answer: str, _arg: Any, _case: dict) -> Result:
    low = _norm(answer)
    hits = [word for word in JARGON if word in low]
    return (not hits, f"жаргон: {', '.join(hits)}" if hits else "")


def no_fragment_markup(answer: str, _arg: Any, _case: dict) -> Result:
    for pattern in FRAGMENT_MARKUP:
        match = pattern.search(answer)
        if match:
            return False, f"разметка контекста: «{match.group(0).strip()}»"
    return True, ""


def no_anonymization_talk(answer: str, _arg: Any, _case: dict) -> Result:
    for pattern in ANONYMIZATION_TALK:
        match = pattern.search(answer)
        if match:
            return False, f"разговор про обезличивание: «{match.group(0)}»"
    return True, ""


def no_absence_claim(answer: str, entities: Sequence[str], _case: dict) -> Result:
    """Не утверждать, что сущности нет в документе.

    Нужна там, где сущность в документе есть, но скрыта за токеном обезличивания:
    ответ «такого лица в документе нет» формально верен для маскированного текста
    и ложен для документа.
    """
    for sentence in _sentences(answer):
        low = _norm(sentence)
        if not any(marker.search(low) for marker in ABSENCE):
            continue
        for entity in entities:
            if _norm(entity) in low:
                return False, f"утверждение об отсутствии: «{sentence[:160]}»"
    return True, ""


def no_contradiction(answer: str, entities: Sequence[str], _case: dict) -> Result:
    """В одном ответе не может быть и «нет такого лица», и утверждения о его роли."""
    for entity in entities:
        entity_low = _norm(entity)
        denies = False
        asserts = False
        for sentence in _sentences(answer):
            low = _norm(sentence)
            if entity_low not in low:
                continue
            if any(marker.search(low) for marker in ABSENCE):
                denies = True
            else:
                asserts = True
        if denies and asserts:
            return False, f"противоречие про «{entity}»: и отрицание, и утверждение"
    return True, ""


def covers_documents(answer: str, _arg: Any, case: dict) -> Result:
    """Каждый приложенный документ должен быть отражён в ответе.

    Опора — не имя файла (модель его не обязана цитировать), а якоря из
    `document_anchors`: реквизит или сумма, которые есть только в этом документе.
    """
    anchors: dict[str, list[str]] = case.get("document_anchors") or {}
    missing = [name for name, variants in anchors.items() if not any(v in answer for v in variants)]
    return (not missing, f"документы не отражены: {', '.join(missing)}" if missing else "")


def no_conflicting_dates(answer: str, _arg: Any, _case: dict) -> Result:
    """Один и тот же реквизит не может иметь двух разных дат.

    Жалоба заказчика: «Дата заключения 15.06.2026» и «Дата заключения договора
    25.06.2026» в одном саммари.
    """
    by_label: dict[str, set[str]] = {}
    for raw_label, date in LABELLED_DATE_RE.findall(answer):
        label = _norm_label(raw_label)
        if len(label) < 4:
            continue
        by_label.setdefault(label, set()).add(date)
    for label, dates in by_label.items():
        if len(dates) > 1:
            return False, f"«{label}» — {' и '.join(sorted(dates))}"
    return True, ""


def no_invented_dates(answer: str, _arg: Any, case: dict) -> Result:
    """Все даты ответа должны встречаться в исходных документах."""
    source = case.get("_source_text", "")
    if not source:
        return True, ""
    known = _known_dates(source)
    invented = sorted({d for d in DATE_RE.findall(answer) if d not in known})
    return (not invented, f"даты не из документа: {', '.join(invented)}" if invented else "")


def must_contain_all(answer: str, needles: Sequence[str], _case: dict) -> Result:
    missing = [n for n in needles if _norm(n) not in _norm(answer)]
    return (not missing, f"нет: {', '.join(missing)}" if missing else "")


def must_contain_any(answer: str, groups: Sequence[Sequence[str]], _case: dict) -> Result:
    low = _norm(answer)
    for group in groups:
        if not any(_norm(v) in low for v in group):
            return False, f"нет ни одного из: {', '.join(group)}"
    return True, ""


def must_not_contain(answer: str, needles: Sequence[str], _case: dict) -> Result:
    low = _norm(answer)
    hits = [n for n in needles if _norm(n) in low]
    return (not hits, f"найдено запрещённое: {', '.join(hits)}" if hits else "")


def must_match(answer: str, patterns: Sequence[str], _case: dict) -> Result:
    missing = [p for p in patterns if not re.search(p, answer, re.IGNORECASE)]
    return (not missing, f"не совпало: {', '.join(missing)}" if missing else "")


def must_not_match(answer: str, patterns: Sequence[str], _case: dict) -> Result:
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return False, f"совпало запрещённое «{match.group(0)[:80]}»"
    return True, ""


_INSUFFICIENT = (
    "установить нельзя",
    "определить нельзя",
    "невозможно установить",
    "нельзя сделать вывод",
    "не представлен",
    "не приложен",
    "не загружен",
    "не регулируется",
    "не содержит",
    "не содержится",
    "отсутствует в представленн",
    "требуется",
    "необходим",
    "нужен",
    "нужна",
    "приложите",
    "загрузите",
)


def declares_missing_document(answer: str, needed: Any, _case: dict) -> Result:
    """Вопрос требует документа, которого нет, — это надо сказать прямо и назвать его.

    Заказчику мало «установить нельзя»: из ответа должно быть понятно, что именно
    приложить. Эталон формулировки — со скриншота 4 отзыва.
    """
    low = _norm(answer)
    if not any(marker in low for marker in _INSUFFICIENT):
        return False, "нет признания, что данных не хватает"
    names = needed if isinstance(needed, (list, tuple)) else []
    missing = [name for name in names if _norm(name) not in low]
    if missing:
        return False, f"не названо, чего не хватает: {', '.join(missing)}"
    return True, ""


def max_chars(answer: str, limit: int, _case: dict) -> Result:
    return (len(answer) <= int(limit), f"{len(answer)} > {limit}")


def min_chars(answer: str, limit: int, _case: dict) -> Result:
    return (len(answer) >= int(limit), f"{len(answer)} < {limit}")


CHECKS: dict[str, Callable[[str, Any, dict], Result]] = {
    "no_jargon": no_jargon,
    "no_fragment_markup": no_fragment_markup,
    "no_anonymization_talk": no_anonymization_talk,
    "no_absence_claim": no_absence_claim,
    "no_contradiction": no_contradiction,
    "covers_documents": covers_documents,
    "no_conflicting_dates": no_conflicting_dates,
    "no_invented_dates": no_invented_dates,
    "must_contain_all": must_contain_all,
    "must_contain_any": must_contain_any,
    "must_not_contain": must_not_contain,
    "must_match": must_match,
    "must_not_match": must_not_match,
    "declares_missing_document": declares_missing_document,
    "max_chars": max_chars,
    "min_chars": min_chars,
}

# Проверки-гигиена включены везде: они описывают требования заказчика к любому
# ответу, а не к конкретному кейсу. Кейс может отключить их явным false.
DEFAULT_CHECKS: tuple[str, ...] = (
    "no_jargon",
    "no_fragment_markup",
    "no_anonymization_talk",
)


def run_checks(answer: str, case: dict) -> list[dict[str, Any]]:
    """Прогнать проверки кейса. Возвращает список результатов по одной на проверку."""
    declared: dict[str, Any] = dict(case.get("checks") or {})
    for name in DEFAULT_CHECKS:
        declared.setdefault(name, True)

    results: list[dict[str, Any]] = []
    for name, arg in declared.items():
        if arg is False or arg is None:
            continue
        check = CHECKS.get(name)
        if check is None:
            results.append({"check": name, "ok": False, "detail": "неизвестная проверка"})
            continue
        try:
            ok, detail = check(answer, arg, case)
        except Exception as exc:  # проверка не должна ронять замер
            ok, detail = False, f"проверка упала: {type(exc).__name__}: {exc}"
        results.append({"check": name, "ok": ok, "detail": detail})
    return results


def unknown_checks(cases: Iterable[dict]) -> set[str]:
    """Имена проверок, которых нет в реестре, — для валидации набора."""
    return {name for case in cases for name in (case.get("checks") or {}) if name not in CHECKS}
