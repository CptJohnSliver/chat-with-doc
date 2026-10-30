#!/usr/bin/env python3
"""Замер качества ответов chat_w_doc.

Гоняет НАСТОЯЩИЙ граф `document_process` с настоящими промптами, но без s3:
документы берутся из `docs/` как готовый текст — ровно в том виде, в каком навык
видит их в проде (обезличенными). Так замер меряет промпты и узлы, а не
инфраструктуру.

    export ROUTER_AI_API_KEY=...
    python3 evals/qa/run_eval.py --label baseline
    python3 evals/qa/run_eval.py --only hy- --workers 2

Отчёты — `evals/qa/reports/<label>.md` и `.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

QA_DIR = Path(__file__).resolve().parent
BACKEND_DIR = QA_DIR.parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONPATH", str(BACKEND_DIR))

from checks import run_checks, unknown_checks  # noqa: E402

INTENT_RE = re.compile(r"Тип запроса:\s*([^\n(]+)")
INTENT_BY_LABEL = {
    "суммаризация документа": "summarize",
    "ответ по документу": "document_qa",
}


# ─────────────────────────────────────────────────────────────────────────────
# Набор
# ─────────────────────────────────────────────────────────────────────────────


def load_cases(dataset: Path) -> list[dict[str, Any]]:
    files = sorted(dataset.glob("*.json")) if dataset.is_dir() else [dataset]
    cases: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        group = payload.get("group") or path.stem
        for case in payload.get("cases", []):
            case.setdefault("group", group)
            case["_file"] = path.name
            cases.append(case)
    return cases


def read_document(name: str) -> str:
    path = QA_DIR / "docs" / name
    if not path.exists():
        raise FileNotFoundError(f"нет документа {path}")
    return path.read_text(encoding="utf-8")


def _fake_s3_loader(name_by_id: dict[str, str]):
    """Подменяет s3-service: текст берётся из docs/, а не скачивается.

    Всё остальное идёт настоящим путём — через DocumentService._prepare, который
    и решает, сколько документов доедет до графа. Обрезка «берём последний файл»
    живёт именно там, поэтому обходить её замером нельзя.
    """
    from src.app.schemas.loaded_document import S3LoadedDocument  # noqa: PLC0415
    from src.infrastructure.document_store import document_local_store  # noqa: PLC0415

    async def loader(file_id: Any, _thread_id: str | None) -> Any:
        name = name_by_id[str(file_id)]
        text = read_document(name)
        local_doc_id, _workspace = document_local_store.allocate_workspace()
        document_local_store.extracted_text_path(local_doc_id).write_text(text, encoding="utf-8")
        source_file_id = str(file_id)
        char_count = len(text)
        document = S3LoadedDocument(
            file_id=source_file_id,
            filename=name,
            char_count=char_count,
            local_doc_id=local_doc_id,
        )
        return document

    return loader


def build_history(case: dict[str, Any]) -> tuple[list[Any], dict[str, str], str]:
    """История сообщений с вложениями — вход DocumentService."""
    from src.common.schemas.models import HistoryMessage, HistoryMessageFragment  # noqa: PLC0415

    name_by_id: dict[str, str] = {}
    source_parts: list[str] = []

    def file_ids(names: list[str]) -> list[Any]:
        ids = []
        for name in names:
            file_id = uuid4()
            name_by_id[str(file_id)] = name
            source_parts.append(read_document(name))
            ids.append(file_id)
        return ids

    history: list[Any] = []
    for turn in case.get("history") or []:
        fragment_type = "request" if turn["role"] == "user" else "response"
        history.append(
            HistoryMessage(
                role=turn["role"],
                fragments=[HistoryMessageFragment(fragment_type=fragment_type, content=turn["content"])],
                file_ids=file_ids(list(turn.get("documents") or [])),
            )
        )
    history.append(
        HistoryMessage(
            role="user",
            fragments=[HistoryMessageFragment(fragment_type="request", content=case["query"])],
            file_ids=file_ids(list(case.get("documents") or [])),
        )
    )
    return history, name_by_id, "\n".join(source_parts)


async def prepare_state(app, case: dict[str, Any]) -> Any:
    """Пройти боевой путь подготовки: DocumentService._prepare + _build_state."""
    from src.app.services.document import DocumentService  # noqa: PLC0415

    history, name_by_id, source_text = build_history(case)
    case["_source_text"] = source_text

    service = DocumentService(
        app,
        history,
        str(uuid4()),
        uuid4(),
        _fake_s3_loader(name_by_id),
        intent=case.get("intent"),
    )
    result = await service.execute()
    if isinstance(result, str):  # заглушка вместо ответа
        return result
    case["_attached_count"] = len(result.get("attached_documents") or [])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Прогон
# ─────────────────────────────────────────────────────────────────────────────


def build_app(overrides: dict[str, Any]):
    from src.agents.document_process import build_document_workflow  # noqa: PLC0415
    from src.agents.llm import (  # noqa: PLC0415
        answer_llm,
        classifier_llm,
        get_llm_timeout,
        summarize_llm,
        summarize_map_llm,
        validation_llm,
    )
    from src.common.config.config import settings  # noqa: PLC0415

    config = dict(settings.document_process)
    config.update(overrides)
    graph = build_document_workflow(
        validation_llm=validation_llm,
        classifier_llm=classifier_llm,
        summarize_llm=summarize_llm,
        summarize_map_llm=summarize_map_llm,
        answer_llm=answer_llm,
        validation_llm_timeout_s=get_llm_timeout("validation"),
        classifier_llm_timeout_s=get_llm_timeout("classifier"),
        summarize_llm_timeout_s=get_llm_timeout("summarize"),
        summarize_map_llm_timeout_s=get_llm_timeout("summarize_map"),
        answer_llm_timeout_s=get_llm_timeout("answer"),
        config=config,
    )
    return graph.compile()


async def run_case(app, case: dict[str, Any]) -> dict[str, Any]:
    from src.agents.document_process.graph import astream_events  # noqa: PLC0415
    from src.common.schemas.models import Message  # noqa: PLC0415
    from src.infrastructure.document_store import document_local_store  # noqa: PLC0415

    t0 = time.perf_counter()
    answer = ""
    intent = ""
    error = ""
    state: Any = None
    try:
        state = await prepare_state(app, case)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if isinstance(state, str):  # сервис вернул заглушку, граф не запускался
        stub = state
        checks = run_checks("", case) if case.get("expect_stub") == stub else []
        ok = case.get("expect_stub") == stub
        return {
            "id": case["id"],
            "group": case["group"],
            "query": case["query"],
            "documents": list(case.get("documents") or []),
            "attached_count": 0,
            "intent": "",
            "ok": ok,
            "error": "" if ok else f"заглушка {stub}",
            "checks": checks,
            "failed": [] if ok else [f"expect_stub: получена заглушка {stub}"],
            "answer": "",
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    try:
        if error:
            raise RuntimeError(error)
        async for event in astream_events(app, state):
            content = getattr(event, "content", "") or ""
            if getattr(event, "fragment_type", "") == "think" and "Тип запроса" in content:
                match = INTENT_RE.search(content)
                if match:
                    intent = INTENT_BY_LABEL.get(match.group(1).strip().casefold(), match.group(1).strip())
            if isinstance(event, Message):
                for fragment in event.fragments:
                    if fragment.fragment_type == "response" and fragment.content:
                        answer = fragment.content
                # Узел глотает исключение LLM и отдаёт извинение как обычный ответ.
                # Для замера это не промах проверок, а несостоявшийся вызов.
                if event.status == "error":
                    error = f"навык вернул ошибку: {answer[:160] or '—'}"
            elif getattr(event, "fragment_type", "") == "response" and content:
                answer = content
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        docs = state.get("attached_documents") or []
        document_local_store.release_attached_local_docs([doc.local_doc_id for doc in docs])

    checks = [] if error else run_checks(answer, case)
    if case.get("expect_attached") is not None:
        attached = case.get("_attached_count", 0)
        checks.append(
            {
                "check": "expect_attached",
                "ok": attached == case["expect_attached"],
                "detail": f"до графа доехало {attached} документов, ожидалось {case['expect_attached']}",
            }
        )
    if not error and case.get("expect_intent"):
        checks.append(
            {
                "check": "expect_intent",
                "ok": intent == case["expect_intent"],
                "detail": f"ожидался {case['expect_intent']}, получен {intent or '—'}",
            }
        )
    if not error and not answer:
        checks.append({"check": "has_answer", "ok": False, "detail": "пустой ответ"})

    failed = [c for c in checks if not c["ok"]]
    return {
        "id": case["id"],
        "group": case["group"],
        "query": case["query"],
        "documents": list(case.get("documents") or []),
        "attached_count": case.get("_attached_count", 0),
        "intent": intent,
        "ok": not error and not failed,
        "error": error,
        "checks": checks,
        "failed": [f"{c['check']}: {c['detail']}" for c in failed],
        "answer": answer,
        "latency_s": round(time.perf_counter() - t0, 2),
    }


async def run_prepare_only(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Проверить только, сколько документов доезжает до графа.

    LLM не вызывается: подготовка состояния — чистый ввод-вывод. Поэтому этот режим
    работает без ключа и меряет обрезку документов детерминированно.
    """
    from src.infrastructure.document_store import document_local_store  # noqa: PLC0415

    results: list[dict[str, Any]] = []
    for case in cases:
        if case.get("expect_attached") is None:
            continue
        t0 = time.perf_counter()
        error = ""
        state: Any = None
        try:
            state = await prepare_state(None, case)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        attached = 0 if isinstance(state, str) or state is None else len(state.get("attached_documents") or [])
        if isinstance(state, dict):
            document_local_store.release_attached_local_docs(
                [doc.local_doc_id for doc in state.get("attached_documents") or []]
            )
        expected = case["expect_attached"]
        check = {
            "check": "expect_attached",
            "ok": not error and attached == expected,
            "detail": f"до графа доехало {attached} документов, ожидалось {expected}",
        }
        results.append(
            {
                "id": case["id"],
                "group": case["group"],
                "query": case["query"],
                "documents": list(case.get("documents") or []),
                "attached_count": attached,
                "intent": "",
                "ok": check["ok"],
                "error": error,
                "checks": [check],
                "failed": [] if check["ok"] else [f"{check['check']}: {check['detail']}"],
                "answer": "",
                "latency_s": round(time.perf_counter() - t0, 2),
            }
        )
    return results


async def run_all(cases: list[dict[str, Any]], *, workers: int, overrides: dict[str, Any]) -> list[dict[str, Any]]:
    app = build_app(overrides)
    semaphore = asyncio.Semaphore(workers)
    done = 0

    async def guarded(case: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        async with semaphore:
            result = await run_case(app, case)
        done += 1
        mark = "·" if result["ok"] else "×"
        print(f"{mark} {done}/{len(cases)} {result['id']}", file=sys.stderr, flush=True)
        return result

    return await asyncio.gather(*(guarded(case) for case in cases))


# ─────────────────────────────────────────────────────────────────────────────
# Отчёт
# ─────────────────────────────────────────────────────────────────────────────


def render_report(results: list[dict[str, Any]], *, label: str) -> str:
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    errors = sum(1 for r in results if r["error"])
    latencies = sorted(r["latency_s"] for r in results)

    def q(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(len(latencies) * p))] if latencies else 0.0

    lines = [
        f"# Качество ответов chat_w_doc: {label}",
        "",
        f"- кейсов: {total}",
        f"- **прошло: {passed}/{total} ({passed / total:.0%})**" if total else "- кейсов нет",
        f"- ошибок вызова: {errors}",
        f"- латентность: p50 {q(0.5)}с · p95 {q(0.95)}с",
        "",
    ]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_group.setdefault(result["group"], []).append(result)
    lines += ["## По группам", "", "| Группа | Прошло | Всего |", "|---|---|---|"]
    for group, items in sorted(by_group.items()):
        lines.append(f"| {group} | {sum(1 for i in items if i['ok'])} | {len(items)} |")
    lines.append("")

    by_check: dict[str, int] = {}
    for result in results:
        for check in result["checks"]:
            if not check["ok"]:
                by_check[check["check"]] = by_check.get(check["check"], 0) + 1
    if by_check:
        lines += ["## Какие проверки падают", "", "| Проверка | Кейсов |", "|---|---|"]
        for name, count in sorted(by_check.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")

    failures = [r for r in results if not r["ok"]]
    if failures:
        lines += ["## Промахи", "", "| Кейс | Запрос | Что не так |", "|---|---|---|"]
        for result in failures:
            reason = result["error"] or "; ".join(result["failed"])
            query = result["query"].replace("|", "/")[:60]
            lines.append(f"| {result['id']} | {query} | {reason.replace(chr(10), ' ')[:180]} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(QA_DIR / "cases"))
    parser.add_argument("--label", default="qa")
    parser.add_argument("--only", default="", help="подстрока id или группы")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--chunk-threshold",
        type=int,
        default=None,
        help="понизить порог чанкования, чтобы прогнать RAG-ветку на коротких документах",
    )
    parser.add_argument("--dry-run", action="store_true", help="только проверить набор, без вызовов LLM")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="дойти до подготовки состояния и проверить только expect_attached — без LLM и без ключа",
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.dataset))
    if args.only:
        cases = [c for c in cases if args.only in c["id"] or args.only in c["group"]]
    if not cases:
        print("кейсы не найдены", file=sys.stderr)
        return 2

    bad = unknown_checks(cases)
    if bad:
        print(f"неизвестные проверки в наборе: {', '.join(sorted(bad))}", file=sys.stderr)
        return 2
    missing = sorted(
        {
            name
            for case in cases
            for name in list(case.get("documents") or []) + [d for t in case.get("history") or [] for d in t.get("documents") or []]
            if not (QA_DIR / "docs" / name).exists()
        }
    )
    if missing:
        print(f"нет документов: {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"набор валиден: {len(cases)} кейсов, {len({c['group'] for c in cases})} групп")
        return 0

    if args.prepare_only:
        results = asyncio.run(run_prepare_only(cases))
        report = render_report(results, label=f"{args.label}-prepare")
        reports = QA_DIR / "reports"
        reports.mkdir(exist_ok=True)
        (reports / f"{args.label}-prepare.md").write_text(report, encoding="utf-8")
        (reports / f"{args.label}-prepare.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(report)
        return 0

    overrides: dict[str, Any] = {}
    if args.chunk_threshold is not None:
        overrides["chunk_threshold"] = args.chunk_threshold
        overrides["summarize_map_reduce_threshold"] = args.chunk_threshold

    results = asyncio.run(run_all(cases, workers=args.workers, overrides=overrides))

    reports = QA_DIR / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{args.label}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = render_report(results, label=args.label)
    (reports / f"{args.label}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
