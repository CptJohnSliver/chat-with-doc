from collections.abc import AsyncIterator
from uuid import UUID

from src.app.api.utils.sse import sse_pack
from src.common.config.config import settings
from src.common.schemas.models import Message, MessageFragment


def _stub(key: str, default: str) -> str:
    value = settings.get(f"stub_messages.{key}", default)
    return default if value is None else str(value)


STUB_UPLOAD_AND_ASK = _stub(
    "upload_and_ask",
    "Загрузите документ и задайте вопрос, или просто прикрепите файл.",
)
STUB_UPLOAD_REQUIRED = _stub(
    "upload_required",
    "Пожалуйста, загрузите документ, чтобы я мог ответить на ваш вопрос.",
)
STUB_SUMMARIZE = _stub("summarize", "Суммаризируй документ")
STUB_TOO_LONG = _stub(
    "too_long",
    (
        "Файл слишком большой для обработки. Максимальный размер файла — 50 МБ. "
        "Пожалуйста, загрузите документ меньшего размера или разделите его на части."
    ),
)
STUB_FILE_LOAD_FAILED = _stub(
    "file_load_failed",
    "Не удалось загрузить ваш файл. Попробуйте приложить другой документ.",
)
STUB_ANONYMIZATION_PENDING = _stub(
    "anonymization_pending",
    ("Документ ещё обрабатывается (обезличивание). Подождите несколько секунд и отправьте запрос снова."),
)
VALIDATION_USER_MESSAGE = _stub("validation_error", "Ошибка валидации данных")
PROCESSING_THINK = _stub("processing_think", "### Обрабатываю запрос...")


def build_stub_message(message_id: UUID, text: str) -> Message:
    """Final stub: close the early think fragment, then the user-visible response."""
    return Message(
        message_id=message_id,
        parent_id=None,
        role="assistant",
        status="success",
        fragments=[
            MessageFragment(
                fragment_id=1,
                fragment_type="think",
                status="success",
                content=PROCESSING_THINK,
            ),
            MessageFragment(
                fragment_id=2,
                fragment_type="response",
                status="success",
                content=text,
            ),
        ],
    )


async def iter_stub_sse(message_id: UUID, text: str, *, connected: bool = True) -> AsyncIterator[str]:
    """Yield a single assistant message without making an LLM call."""
    if connected:
        yield ": connected\n\n"
    yield sse_pack({"data": build_stub_message(message_id, text).model_dump()})
