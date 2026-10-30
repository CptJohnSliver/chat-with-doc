import asyncio
import codecs
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from src.app.schemas.loaded_document import S3LoadedDocument
from src.common.exceptions import (
    DocumentTooLargeError,
    InvalidDocumentTextError,
)
from src.infrastructure.document_store import document_local_store
from src.infrastructure.logger import get_logger
from src.infrastructure.s3_services.async_client import (
    S3ObjectTooLargeError,
    S3ServiceAsyncClient,
    get_s3_client,
)
from src.infrastructure.s3_services.models import FileMetadata

logger = get_logger(__name__)

_STREAM_DECODE_CHUNK = 1024 * 1024
_MAX_CONTROL_BYTE_RATIO = 0.02

# Back-compat alias used by DocumentService / tests.
DocumentFileTooLargeError = DocumentTooLargeError


def _validate_text_bytes(data: bytes) -> None:
    if not data:
        return
    if b"\x00" in data:
        raise InvalidDocumentTextError("Anonymized document contains NUL bytes")
    disallowed_controls = sum(byte < 32 and byte not in {9, 10, 12, 13} for byte in data)
    if disallowed_controls / len(data) > _MAX_CONTROL_BYTE_RATIO:
        raise InvalidDocumentTextError("Anonymized document appears to be binary data")


def _decode_text_bytes(data: bytes) -> str:
    _validate_text_bytes(data)
    for encoding in ("utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidDocumentTextError("Unsupported anonymized document text encoding")


def _stream_encoded_to_extracted(
    raw_path: Path,
    out_path: Path,
    *,
    encoding: str,
) -> int:
    """Strict incremental decode into UTF-8 without retaining the full file in RAM."""
    total = 0
    decoder = codecs.getincrementaldecoder(encoding)("strict")
    with raw_path.open("rb") as src, out_path.open("w", encoding="utf-8") as dst:
        while chunk := src.read(_STREAM_DECODE_CHUNK):
            _validate_text_bytes(chunk)
            piece = decoder.decode(chunk)
            if piece:
                dst.write(piece)
                total += len(piece)
        tail = decoder.decode(b"", final=True)
        if tail:
            dst.write(tail)
            total += len(tail)
    return total


def _stream_utf8_to_extracted(raw_path: Path, out_path: Path) -> int:
    """Strict incremental UTF-8 decode; raises UnicodeDecodeError on invalid input."""
    return _stream_encoded_to_extracted(raw_path, out_path, encoding="utf-8")


def _fallback_decode_to_extracted(raw_path: Path, out_path: Path) -> int:
    """Stream the supported legacy encoding without loading the file into RAM."""
    size = raw_path.stat().st_size
    if size > document_local_store.max_upload_file_bytes():
        raise InvalidDocumentTextError(
            detail="Failed to decode anonymized document text as UTF-8",
        )
    return _stream_encoded_to_extracted(raw_path, out_path, encoding="cp1251")


def _stream_raw_to_extracted(raw_path: Path, out_path: Path) -> int:
    """Decode anonymized text bytes from disk into UTF-8 extracted.txt without loading whole file."""
    try:
        total = _stream_utf8_to_extracted(raw_path, out_path)
    except UnicodeDecodeError:
        # Incremental UTF-8 raises before any "empty decode → fallback" path can run.
        return _fallback_decode_to_extracted(raw_path, out_path)

    if total == 0 and raw_path.stat().st_size > 0:
        return _fallback_decode_to_extracted(raw_path, out_path)
    return total


async def _fetch_metadata_optional(client: S3ServiceAsyncClient, file_id: str) -> FileMetadata | None:
    try:
        return await client.metadata(key=file_id)
    except Exception as exc:
        await logger.awarning("S3 metadata unavailable for file %s: %s", file_id, exc)
        return None


async def load_anonymized_document(file_id: str, client: S3ServiceAsyncClient) -> S3LoadedDocument:
    """Download obfuscated text from s3-service (202 while processing)."""
    metadata = await _fetch_metadata_optional(client, file_id)

    local_doc_id, _workspace = document_local_store.allocate_workspace()
    out_path = document_local_store.extracted_text_path(local_doc_id)
    raw_path = document_local_store.source_file_path(local_doc_id, ".anonymized")

    limit = document_local_store.max_upload_file_bytes()
    try:
        await client.download_anonymized_to_file(
            key=file_id,
            output_path=raw_path,
            max_bytes=limit,
        )
        size = raw_path.stat().st_size
        if size == 0:
            raise InvalidDocumentTextError(
                detail="Anonymized document does not contain text",
            )

        decode_task = asyncio.create_task(asyncio.to_thread(_stream_raw_to_extracted, raw_path, out_path))
        try:
            char_count = await asyncio.shield(decode_task)
        except asyncio.CancelledError:
            # A worker thread cannot be force-cancelled. Wait until it stops touching
            # the workspace before the outer cancellation cleanup removes it.
            with suppress(Exception):
                await decode_task
            raise
        raw_path.unlink(missing_ok=True)
        filename = metadata.original_filename if metadata else ""
        document = S3LoadedDocument(
            file_id=file_id,
            filename=filename,
            char_count=char_count,
            local_doc_id=local_doc_id,
        )
        return document
    except S3ObjectTooLargeError as exc:
        document_local_store.release_local_doc(local_doc_id)
        raise DocumentTooLargeError(exc.size_bytes, exc.limit_bytes) from exc
    except asyncio.CancelledError:
        document_local_store.release_local_doc(local_doc_id)
        raise
    except Exception:
        document_local_store.release_local_doc(local_doc_id)
        raise


async def extract_document_from_s3(file_id: UUID, thread_id: str | None = None) -> S3LoadedDocument:
    del thread_id
    return await load_anonymized_document(str(file_id), get_s3_client())
