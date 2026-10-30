from __future__ import annotations

from contextlib import suppress
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.common.config.config import settings
from src.common.exceptions import (
    DocumentNotReadyError,
    DocumentTooLargeError,
    ExternalServiceError,
    NotFound,
)
from src.infrastructure.request_identity import current_identity

from .models import FileMetadata


class AnonymizingFileIsNotReadyError(DocumentNotReadyError):
    pass


class S3ServiceClientError(ExternalServiceError):
    user_message = "Ошибка при обращении к хранилищу документов"


class S3FileNotFoundError(NotFound):
    user_message = "Файл не найден"


class S3ObjectTooLargeError(DocumentTooLargeError):
    pass


retry_on_not_ready = retry(
    stop=stop_after_attempt(int(settings.get("S3_SERVICE_RETRY_ON_NOT_READY_ATTEMPTS", 5))),
    wait=wait_exponential(
        multiplier=float(settings.get("S3_SERVICE_RETRY_ON_NOT_READY_WAIT_MULTIPLIER", 2)),
        min=float(settings.get("S3_SERVICE_RETRY_ON_NOT_READY_WAIT_MIN_SECONDS", 2)),
        max=float(settings.get("S3_SERVICE_RETRY_ON_NOT_READY_WAIT_MAX_SECONDS", 30)),
    ),
    retry=retry_if_exception_type(AnonymizingFileIsNotReadyError),
    reraise=True,
)


def anonymized_params() -> dict[str, str] | None:
    """chat_id из X-Chat-Id: обезличенная версия файла этого чата."""
    raw = current_identity().chat_id
    if not raw:
        return None
    try:
        return {"chat_id": str(UUID(raw.strip()))}
    except ValueError:
        return None


class S3ServiceAsyncClient:
    def __init__(
        self,
        *,
        base_url: str,
        service_token: str | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._service_token = service_token
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> S3ServiceAsyncClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    def _service_headers(self) -> dict[str, str]:
        if not self._service_token:
            raise S3ServiceClientError("Service token is required for this operation")
        return {"Authorization": f"Bearer {self._service_token}"}

    @staticmethod
    async def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == HTTPStatus.NOT_FOUND:
            raise S3FileNotFoundError("File not found")
        # Streaming responses do not buffer the body; read before .json()/.text.
        if not hasattr(response, "_content"):
            with suppress(Exception):
                await response.aread()
        try:
            payload: object = response.json()
        except Exception:
            payload = getattr(response, "text", "") or ""
        raise S3ServiceClientError(f"HTTP {response.status_code}: {payload}")

    @staticmethod
    def _encode_key(key: str) -> str:
        return quote(key, safe="/")

    def _anonymized_download_url(self) -> str:
        return settings.get(
            "S3_SERVICE_UPLOAD_ANONYMIZED_FILE_PATH",
            "/api/s3-service/v1/s3/download/{file_id}/anonymized",
        )

    async def metadata(self, *, key: str) -> FileMetadata:
        encoded_key = self._encode_key(key)
        response = await self._client.get(
            f"{settings.get('S3_SERVICE_META_PATH', '/s3/meta')}/{encoded_key}", headers=self._service_headers()
        )
        await self._raise_for_status(response)
        data = response.json()
        return FileMetadata(
            file_id=data["file_id"],
            bucket_name=data["bucket_name"],
            original_filename=data["original_filename"],
            storage_key=data["storage_key"],
            size_bytes=data["size_bytes"],
            content_type=data.get("content_type"),
            created_by=data.get("created_by"),
        )

    @retry_on_not_ready
    async def download_anonymized_to_file(
        self,
        *,
        key: str,
        output_path: str | Path,
        chunk_size: int = 64 * 1024,
        max_bytes: int | None = None,
    ) -> Path:
        """Download obfuscated text artifact from s3-service.

        202 Accepted (still anonymizing) is retried with backoff. Callers should
        not hold the graph concurrency semaphore across this wait.

        If max_bytes is set, abort as soon as Content-Length or streamed bytes exceed it
        (avoids writing/transferring an oversized payload).
        """
        target = Path(output_path)
        encoded_key = self._encode_key(key)
        url = self._anonymized_download_url().format(file_id=encoded_key)

        async with self._client.stream("GET", url, headers=self._service_headers(), params=anonymized_params()) as response:
            if response.status_code == HTTPStatus.ACCEPTED:
                # Drain the body so the keep-alive connection can be reused cleanly.
                with suppress(Exception):
                    await response.aread()
                raise AnonymizingFileIsNotReadyError(f"File {key} is still processing in s3-service (202 Accepted)")

            await self._raise_for_status(response)

            if max_bytes is not None:
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared = int(content_length)
                    except ValueError:
                        pass
                    else:
                        if declared > max_bytes:
                            raise S3ObjectTooLargeError(declared, max_bytes)

            written = 0
            try:
                with target.open("wb") as file_handle:
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if max_bytes is not None and written > max_bytes:
                            raise S3ObjectTooLargeError(written, max_bytes)
                        file_handle.write(chunk)
            except S3ObjectTooLargeError:
                with suppress(OSError):
                    target.unlink(missing_ok=True)
                raise
        return target


class _S3ClientState:
    instance: S3ServiceAsyncClient | None = None


def get_s3_client() -> S3ServiceAsyncClient:
    """Lazy singleton — import of this module must not require S3 secrets."""
    if _S3ClientState.instance is not None:
        return _S3ClientState.instance

    base_url = settings.get("S3_SERVICE_BASE_URL")
    if not base_url:
        raise RuntimeError("S3_SERVICE_BASE_URL is not set. Copy backend/env.local.example to .env and fill secrets.")
    _S3ClientState.instance = S3ServiceAsyncClient(
        base_url=base_url,
        service_token=settings.get("S3_SERVICE_ACCESS_TOKEN"),
        timeout=float(settings.get("S3_SERVICE_TIMEOUT_SECONDS", 300.0)),
    )
    return _S3ClientState.instance
