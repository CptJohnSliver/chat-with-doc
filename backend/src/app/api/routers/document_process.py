from uuid import UUID

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from src.agents.document_process import get_document_app
from src.app.api.utils.document_text import extract_document_from_s3
from src.app.schemas.document import DocumentRequest
from src.app.services.document import DocumentService
from src.infrastructure.request_identity import bind_conversation

router = APIRouter()


@router.post("/document_sse/{thread_id}")
async def document_chat_stream_sse(thread_id: UUID, payload: DocumentRequest) -> StreamingResponse:
    thread_id_str = str(thread_id)
    bind_conversation(chat_id=thread_id_str, message_id=str(payload.message_id))
    service = DocumentService(
        get_document_app(),
        payload.history_messages,
        thread_id_str,
        payload.message_id,
        extract_document_from_s3,
    )
    return StreamingResponse(
        service.stream_response(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
