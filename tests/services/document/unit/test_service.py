
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from src.agents.document_process.models import AttachedDocument
from src.agents.document_process.state import AgentState
from src.app.api.utils.document_text import DocumentFileTooLargeError
from src.app.api.utils.stub_response import (STUB_ANONYMIZATION_PENDING,
                                             STUB_FILE_LOAD_FAILED,
                                             STUB_SUMMARIZE, STUB_TOO_LONG,
                                             STUB_UPLOAD_AND_ASK,
                                             STUB_UPLOAD_REQUIRED)
from src.app.schemas.loaded_document import S3LoadedDocument
from src.app.services.document import DocumentService
from src.common.schemas.models import (HistoryMessage, HistoryMessageFragment,
                                       Message, MessageFragment)
from src.infrastructure.s3_services.async_client import AnonymizingFileIsNotReadyError

from tests.custom_parametrize import parametrize_from_dict


def make_history_message(
    content: str = "",
    file_ids: list[UUID] | None = None,
    role: Literal['user', 'assistant'] = "user",
) -> HistoryMessage:
    if file_ids is None:
        file_ids = []
    fragment = HistoryMessageFragment(fragment_type="request", content=content if content else None)
    return HistoryMessage(role=role, fragments=[fragment], file_ids=file_ids)


def make_s3_loader(result: S3LoadedDocument | Exception) -> AsyncMock:
    if isinstance(result, Exception):
        return AsyncMock(side_effect=result)
    return AsyncMock(return_value=result)


def loaded_doc(
    filename: str,
    text_len: int,
    local_doc_id: str = "local-doc-1",
    *,
    file_id: str = "source-file-1",
) -> S3LoadedDocument:
    """Создаёт результат загрузки с исходным идентификатором файла."""
    document = S3LoadedDocument(
        file_id=file_id,
        filename=filename,
        char_count=text_len,
        local_doc_id=local_doc_id,
    )
    return document


def message_eq(actual: Message, expected: Message) -> bool:
    """
    Сравнивает два Message, игнорируя динамические поля:
    - message_id
    - created_at
    - updated_at
    - thinking_duration_s
    - accumulated_token_usage
    """
    if type(actual) != type(expected):
        return False
    for field in ["parent_id", "role", "status", "fragments"]:
        if getattr(actual, field) != getattr(expected, field):
            return False
    if len(actual.fragments) != len(expected.fragments):
        return False
    for af, ef in zip(actual.fragments, expected.fragments):
        if not fragment_eq(af, ef):
            return False
    return True


def fragment_eq(actual: MessageFragment, expected: MessageFragment) -> bool:
    """Сравнивает MessageFragment, игнорируя динамические поля (например, нет динамических, но на всякий случай)"""
    return (
        actual.fragment_id == expected.fragment_id
        and actual.fragment_type == expected.fragment_type
        and actual.status == expected.status
        and actual.streaming == expected.streaming
        and actual.content == expected.content
        and actual.token_usage == expected.token_usage
        and actual.duration_s == expected.duration_s
    )


def assert_state_subset(state: AgentState | str, expected: dict) -> None:
    for key, value in expected.items():
        assert key in state, f"Missing key: {key}"
        actual_val = state[key]

        if key == "input_message" and isinstance(value, Message):
            assert message_eq(actual_val, value), f"Message mismatch: {actual_val} != {value}"
        elif key == "meta" and isinstance(value, dict):
            assert actual_val == value, f"Meta mismatch: {actual_val} != {value}"
        elif key == "income_history_messages":
            assert len(actual_val) == len(value), f"income_history_messages length mismatch"
            for a, e in zip(actual_val, value):
                assert a.role == e.role, f"role mismatch: {a.role} != {e.role}"
                assert a.fragments == e.fragments, f"fragments mismatch"
        else:
            assert actual_val == value, f"Key '{key}': {actual_val} != {value}"


document_cases = {
    "success_with_file_and_text": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="Analyze this", file_ids=[uuid4()])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(loaded_doc("doc.txt", 21)),
        "expected": {
            "input_message": Message(
                parent_id=None,
                role="user",
                status="success",
                fragments=[MessageFragment(fragment_id=1, fragment_type="request", status="success", content="Analyze this")],
            ),
            "meta": {"configurable": {"thread_id": "thread_123"}},
            "chat_id": "thread_123",
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="doc.txt",
                    text="",
                    file_index=0,
                    char_count=21,
                    local_doc_id="local-doc-1",
                )
            ],
            "income_history_messages": [],
        },
    },
    "success_with_file_only_triggers_summarize": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="", file_ids=[uuid4()])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(loaded_doc("doc.txt", 7)),
        "expected": {
            "input_message": Message(
                parent_id=None,
                role="user",
                status="success",
                fragments=[MessageFragment(fragment_id=1, fragment_type="request", status="success", content=STUB_SUMMARIZE)],
            ),
            "meta": {"configurable": {"thread_id": "thread_123"}},
            "chat_id": "thread_123",
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="doc.txt",
                    text="",
                    file_index=0,
                    char_count=7,
                    local_doc_id="local-doc-1",
                )
            ],
            "income_history_messages": [],
        },
    },
    "no_text_no_file_returns_stub_upload_and_ask": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="", file_ids=[])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(loaded_doc("", 0)),
        "expected": STUB_UPLOAD_AND_ASK,
    },
    "text_only_returns_stub_upload_required": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="Hello", file_ids=[])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(loaded_doc("", 0)),
        "expected": STUB_UPLOAD_REQUIRED,
    },
    "history_anonymization_pending_returns_pending_stub": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="Previous", file_ids=[uuid4()]),
            make_history_message(content="Current question", file_ids=[]),
        ],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(AnonymizingFileIsNotReadyError("still processing")),
        "expected": STUB_ANONYMIZATION_PENDING,
    },
    "history_s3_error_returns_file_load_failed": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="Previous", file_ids=[uuid4()]),
            make_history_message(content="Current question", file_ids=[]),
        ],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(Exception("Attempted to access streaming response content, without having called `read()`.")),
        "expected": STUB_FILE_LOAD_FAILED,
    },
    "s3_loader_fails_returns_stub_file_load_failed": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="", file_ids=[uuid4()])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(Exception("S3 connection error")),
        "expected": STUB_FILE_LOAD_FAILED,
    },
    "document_too_large_returns_stub_too_long": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="", file_ids=[uuid4()])],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(DocumentFileTooLargeError(60 * 1024 * 1024, 50 * 1024 * 1024)),
        "expected": STUB_TOO_LONG,
    },
    "file_from_history_if_current_has_no_file": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="Previous", file_ids=[uuid4()]),
            make_history_message(content="Current question", file_ids=[]),
        ],
        "thread_id": "thread_123",
        "message_id": uuid4(),
        "s3_loader": make_s3_loader(loaded_doc("history_doc.txt", 19)),
        "expected": {
            "input_message": Message(
                parent_id=None,
                role="user",
                status="success",
                fragments=[MessageFragment(fragment_id=1, fragment_type="request", status="success", content="Current question")],
            ),
            "meta": {"configurable": {"thread_id": "thread_123"}},
            "chat_id": "thread_123",
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="history_doc.txt",
                    text="",
                    file_index=0,
                    char_count=19,
                    local_doc_id="local-doc-1",
                    from_current_message=False,
                )
            ],
            "income_history_messages": [make_history_message(content="Previous", file_ids=[uuid4()])],
        },
    },
    "empty_history_returns_stub_upload_and_ask": {
        "graph": MagicMock(),
        "history_messages": [],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(),
        "expected": STUB_UPLOAD_AND_ASK,
    },
    "message_without_fragments": {
        "graph": MagicMock(),
        "history_messages": [HistoryMessage(role="user", fragments=[])],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(),
        "expected": STUB_UPLOAD_AND_ASK,
    },
    "history_s3_error_continues_to_next_file": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="ok", file_ids=[uuid4()]),
            make_history_message(content="fail", file_ids=[uuid4()]),
            make_history_message(content="current question", file_ids=[]),
        ],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(side_effect=[ConnectionError("S3 timeout"), loaded_doc("ok.txt", 7, "local-2")]),
        "expected": {
            "input_message": Message(
                parent_id=None,
                role="user",
                status="success",
                fragments=[MessageFragment(fragment_id=1, fragment_type="request", status="success", content="current question")],
            ),
            "meta": {"configurable": {"thread_id": "t1"}},
            "chat_id": "t1",
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="ok.txt",
                    text="",
                    file_index=0,
                    char_count=7,
                    local_doc_id="local-2",
                    from_current_message=False,
                )
            ],
            "income_history_messages": [
                make_history_message(content="ok", file_ids=[uuid4()]),
                make_history_message(content="fail", file_ids=[uuid4()]),
            ],
        },
    },
    "all_files_of_current_message_reach_the_graph": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="Проверь оба", file_ids=[uuid4(), uuid4()])],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(
            side_effect=[
                loaded_doc("dkp.pdf", 11, "local-1"),
                loaded_doc("egrn.pdf", 22, "local-2", file_id="source-file-2"),
            ]
        ),
        "expected": {
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="dkp.pdf",
                    text="",
                    file_index=0,
                    char_count=11,
                    local_doc_id="local-1",
                ),
                AttachedDocument(
                    file_id="source-file-2",
                    filename="egrn.pdf",
                    text="",
                    file_index=1,
                    char_count=22,
                    local_doc_id="local-2",
                ),
            ],
        },
    },
    "history_document_joins_the_current_one": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="Проверь договор", file_ids=[uuid4()]),
            make_history_message(content="А теперь доверенность", file_ids=[uuid4()]),
        ],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(
            side_effect=[
                loaded_doc("poa.pdf", 5, "local-1"),
                loaded_doc("dkp.pdf", 7, "local-2", file_id="source-file-2"),
            ]
        ),
        "expected": {
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="poa.pdf",
                    text="",
                    file_index=0,
                    char_count=5,
                    local_doc_id="local-1",
                ),
                AttachedDocument(
                    file_id="source-file-2",
                    filename="dkp.pdf",
                    text="",
                    file_index=1,
                    char_count=7,
                    local_doc_id="local-2",
                    from_current_message=False,
                ),
            ],
        },
    },
    "broken_file_does_not_hide_the_readable_one": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="Проверь оба", file_ids=[uuid4(), uuid4()])],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(side_effect=[Exception("S3 timeout"), loaded_doc("ok.pdf", 9, "local-2")]),
        "expected": {
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="ok.pdf",
                    text="",
                    file_index=0,
                    char_count=9,
                    local_doc_id="local-2",
                ),
            ],
        },
    },
    "pending_among_current_files_asks_to_retry": {
        "graph": MagicMock(),
        "history_messages": [make_history_message(content="Проверь оба", file_ids=[uuid4(), uuid4()])],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(
            side_effect=[loaded_doc("ok.pdf", 9, "local-1"), AnonymizingFileIsNotReadyError("still processing")]
        ),
        "expected": STUB_ANONYMIZATION_PENDING,
    },
    "history_too_large_continues_to_next": {
        "graph": MagicMock(),
        "history_messages": [
            make_history_message(content="ok", file_ids=[uuid4()]),
            make_history_message(content="too large", file_ids=[uuid4()]),
            make_history_message(content="question", file_ids=[]),
        ],
        "thread_id": "t1",
        "message_id": uuid4(),
        "s3_loader": AsyncMock(
            side_effect=[
                DocumentFileTooLargeError(60 * 1024 * 1024, 50 * 1024 * 1024),
                loaded_doc("ok.txt", 9, "local-3"),
            ]
        ),
        "expected": {
            "input_message": Message(
                parent_id=None,
                role="user",
                status="success",
                fragments=[MessageFragment(fragment_id=1, fragment_type="request", status="success", content="question")],
            ),
            "meta": {"configurable": {"thread_id": "t1"}},
            "chat_id": "t1",
            "attached_documents": [
                AttachedDocument(
                    file_id="source-file-1",
                    filename="ok.txt",
                    text="",
                    file_index=0,
                    char_count=9,
                    local_doc_id="local-3",
                    from_current_message=False,
                )
            ],
            "income_history_messages": [
                make_history_message(content="ok", file_ids=[uuid4()]),
                make_history_message(content="too large", file_ids=[uuid4()]),
            ],
        },
    },
}


@parametrize_from_dict(document_cases)
@pytest.mark.asyncio
async def test_document_service(
    graph: MagicMock,
    history_messages: list[HistoryMessage],
    thread_id: str,
    message_id: UUID,
    s3_loader: AsyncMock,
    expected: dict | str,
) -> None:

    service = DocumentService(
        graph=graph,
        history_messages=history_messages,
        thread_id=thread_id,
        message_id=message_id,
        s3_loader=s3_loader,
    )
    result = await service.execute()

    if isinstance(expected, dict):
        assert_state_subset(result, expected)
    else:
        assert result == expected


async def prepared_state(
    history_messages: list[HistoryMessage],
    s3_loader: AsyncMock,
    intent: str | None = None,
) -> AgentState | str:
    service = DocumentService(
        graph=MagicMock(),
        history_messages=history_messages,
        thread_id="t1",
        message_id=uuid4(),
        s3_loader=s3_loader,
        intent=intent,
    )
    return await service.execute()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "expected"),
    [("summarize", "summarize"), ("qa", "document_qa"), ("QA", "document_qa")],
)
async def test_intent_from_orchestrator_wins(intent: str, expected: str) -> None:
    """Оркестратор уже решил, саммари нужно или ответ, — навык не угадывает заново."""
    state = await prepared_state(
        [make_history_message(content="Проверь", file_ids=[uuid4()])],
        make_s3_loader(loaded_doc("dkp.pdf", 11)),
        intent=intent,
    )

    assert state["document_intent_hint"] == expected


@pytest.mark.asyncio
async def test_without_intent_behaviour_is_unchanged() -> None:
    """Параметр необязателен: без него решает внутренний классификатор."""
    state = await prepared_state(
        [make_history_message(content="Проверь", file_ids=[uuid4()])],
        make_s3_loader(loaded_doc("dkp.pdf", 11)),
    )

    assert "document_intent_hint" not in state


@pytest.mark.asyncio
async def test_unknown_intent_is_ignored_not_propagated() -> None:
    state = await prepared_state(
        [make_history_message(content="Проверь", file_ids=[uuid4()])],
        make_s3_loader(loaded_doc("dkp.pdf", 11)),
        intent="разбери",
    )

    assert "document_intent_hint" not in state


@pytest.mark.asyncio
async def test_empty_message_still_defaults_to_summary() -> None:
    """Файл без текста — краткое содержание, даже если намерение не передали."""
    state = await prepared_state(
        [make_history_message(content="", file_ids=[uuid4()])],
        make_s3_loader(loaded_doc("dkp.pdf", 11)),
    )

    assert state["document_intent_hint"] == "summarize"


@pytest.mark.asyncio
async def test_unreadable_file_is_named_in_the_answer() -> None:
    """Молчаливый пропуск файла неотличим от «прочитал и не нашёл»."""
    state = await prepared_state(
        [make_history_message(content="Проверь оба", file_ids=[uuid4(), uuid4()])],
        AsyncMock(side_effect=[Exception("S3 timeout"), loaded_doc("ok.pdf", 9, "local-2")]),
    )

    warnings = state["attachment_warnings"]
    assert len(warnings) == 1
    assert "1 файл не удалось прочитать" in warnings[0]


@pytest.mark.asyncio
async def test_all_files_readable_gives_no_warning() -> None:
    state = await prepared_state(
        [make_history_message(content="Проверь оба", file_ids=[uuid4(), uuid4()])],
        AsyncMock(
            side_effect=[
                loaded_doc("a.pdf", 5, "local-1"),
                loaded_doc("b.pdf", 5, "local-2", file_id="source-file-2"),
            ]
        ),
    )

    assert "attachment_warnings" not in state


@pytest.mark.asyncio
async def test_pending_file_does_not_fall_through_to_an_older_document() -> None:
    """Большой файл ещё обезличивается — отвечать по предыдущему нельзя.

    Первый ход вернул «файл обрабатывается». На втором ходу файлов в сообщении нет,
    и обход истории дошёл бы до готового документа постарше — пользователь получил бы
    ответ по нему, думая, что спрашивает про новый.
    """
    loader = AsyncMock(
        side_effect=[
            AnonymizingFileIsNotReadyError("still processing"),
            loaded_doc("старый.pdf", 9, "local-old"),
        ]
    )
    history = [
        make_history_message(content="Первый договор", file_ids=[uuid4()]),
        make_history_message(content="А теперь этот", file_ids=[uuid4()]),
        make_history_message(content="Какая цена?", file_ids=[]),
    ]

    assert await prepared_state(history, loader) == STUB_ANONYMIZATION_PENDING
    assert loader.await_count == 1  # до старого документа даже не дошли


@pytest.mark.asyncio
async def test_stale_pending_does_not_block_a_fresh_attachment() -> None:
    """Необработанный файл из прошлого хода не мешает ответить по только что приложенному."""
    loader = AsyncMock(
        side_effect=[
            loaded_doc("новый.pdf", 11, "local-new"),
            AnonymizingFileIsNotReadyError("still processing"),
        ]
    )
    history = [
        make_history_message(content="Старый файл", file_ids=[uuid4()]),
        make_history_message(content="Проверь этот", file_ids=[uuid4()]),
    ]

    state = await prepared_state(history, loader)

    assert [doc.filename for doc in state["attached_documents"]] == ["новый.pdf"]


@pytest.mark.asyncio
async def test_same_file_attached_twice_is_loaded_once() -> None:
    file_id = uuid4()
    source_file_id = str(file_id)
    document = loaded_doc("dkp.pdf", 7, "local-1", file_id=source_file_id)
    loader = AsyncMock(side_effect=[document])

    state = await prepared_state(
        [
            make_history_message(content="Проверь договор", file_ids=[file_id]),
            make_history_message(content="Ещё раз", file_ids=[file_id]),
        ],
        loader,
    )

    assert len(state["attached_documents"]) == 1
    assert state["attached_documents"][0].file_id == source_file_id
    assert state["attached_documents"][0].from_current_message is True
    assert loader.await_count == 1


@pytest.mark.asyncio
async def test_same_filename_keeps_distinct_file_ids_and_origins() -> None:
    """Сохраняет идентичность одноимённых файлов текущего сообщения и истории."""
    old_file_id = uuid4()
    new_file_id = uuid4()
    old_source_id = str(old_file_id)
    new_source_id = str(new_file_id)
    old_document = loaded_doc("договор.pdf", 30, "local-old", file_id=old_source_id)
    new_document = loaded_doc("договор.pdf", 5, "local-new", file_id=new_source_id)
    loader = AsyncMock(side_effect=[new_document, old_document])
    history = [
        make_history_message(content="Старый договор", file_ids=[old_file_id]),
        make_history_message(content="Проверь новый договор", file_ids=[new_file_id]),
    ]

    state = await prepared_state(history, loader)

    assert isinstance(state, dict)
    documents = state["attached_documents"]
    assert [doc.file_id for doc in documents] == [new_source_id, old_source_id]
    assert [doc.filename for doc in documents] == ["договор.pdf", "договор.pdf"]
    assert [doc.from_current_message for doc in documents] == [True, False]
    assert [doc.local_doc_id for doc in documents] == ["local-new", "local-old"]
    assert [call.args[0] for call in loader.await_args_list] == [new_file_id, old_file_id]


@pytest.mark.asyncio
async def test_history_documents_are_bounded() -> None:
    """Весь чат не должен уезжать в каждый ответ: добор из истории ограничен."""
    loader = AsyncMock(
        side_effect=[
            loaded_doc(f"doc{i}.pdf", 5, f"local-{i}", file_id=f"source-file-{i}")
            for i in range(6)
        ]
    )
    history = [make_history_message(content=f"ход {i}", file_ids=[uuid4()]) for i in range(5)]
    history.append(make_history_message(content="Вопрос", file_ids=[]))

    state = await prepared_state(history, loader)

    assert len(state["attached_documents"]) == 2  # history_documents_limit


@pytest.mark.asyncio
async def test_current_documents_come_before_history_ones() -> None:
    state = await prepared_state(
        [
            make_history_message(content="Договор", file_ids=[uuid4()]),
            make_history_message(content="Доверенность", file_ids=[uuid4()]),
        ],
        AsyncMock(
            side_effect=[
                loaded_doc("poa.pdf", 5, "local-1"),
                loaded_doc("dkp.pdf", 7, "local-2", file_id="source-file-2"),
            ]
        ),
    )

    assert [doc.filename for doc in state["attached_documents"]] == ["poa.pdf", "dkp.pdf"]
    assert [doc.file_index for doc in state["attached_documents"]] == [0, 1]


@pytest.mark.asyncio
async def test_current_document_is_marked_apart_from_historical_ones() -> None:
    """Происхождение должно пережить сборку: дальше по конвейеру его взять неоткуда.

    Пользователь приложил доверенность и спросил про риски. Договор подряда
    приезжает из прошлого хода — и без пометки ответ уходил по нему, потому что
    риски нашлись именно там.
    """
    state = await prepared_state(
        [
            make_history_message(content="Договор подряда", file_ids=[uuid4()]),
            make_history_message(content="Проверь риски", file_ids=[uuid4()]),
        ],
        AsyncMock(
            side_effect=[
                loaded_doc("poa.pdf", 5, "local-1"),
                loaded_doc("contract.pdf", 7, "local-2", file_id="source-file-2"),
            ]
        ),
    )

    assert [(doc.filename, doc.from_current_message) for doc in state["attached_documents"]] == [
        ("poa.pdf", True),
        ("contract.pdf", False),
    ]


@pytest.mark.asyncio
async def test_documents_of_a_question_without_attachments_stay_current() -> None:
    """Файлов в сообщении нет — все документы исторические, соседства не возникает."""
    state = await prepared_state(
        [
            make_history_message(content="Договор", file_ids=[uuid4()]),
            make_history_message(content="Какая цена?", file_ids=[]),
        ],
        make_s3_loader(loaded_doc("dkp.pdf", 11)),
    )

    assert [doc.from_current_message for doc in state["attached_documents"]] == [False]
