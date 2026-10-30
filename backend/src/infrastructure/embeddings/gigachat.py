from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from gigachat import GigaChatAsyncClient
from gigachat.exceptions import RequestEntityTooLargeError

from src.common.config.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 64
_MIN_TEXT_SYMBOLS = 256

# Russian Trusted Root CA — required to verify gigachat.devices.sberbank.ru
_DEFAULT_CA_BUNDLE = Path(__file__).resolve().parents[2] / "common" / "config" / "certs" / "russian_trusted_root_ca_pem.crt"


def _resolve_ca_bundle() -> str:
    configured = settings.get("GIGACHAT_CA_BUNDLE")
    path = Path(configured) if configured else _DEFAULT_CA_BUNDLE
    if not path.is_file():
        raise FileNotFoundError(f"GigaChat CA bundle not found: {path}")
    return str(path)


class GigaChatEmbeddings:
    """Async GigaChat embeddings — never blocks the event loop with sync I/O or sleep."""

    def __init__(
        self,
        model_name: str,
        *,
        credentials: str | None = None,
        scope: str | None = None,
        api_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 5,
        ca_bundle_file: str | None = None,
    ) -> None:
        self.model_name = model_name

        credentials = credentials or settings.get("GIGACHAT_API_KEY")
        if not credentials:
            raise ValueError("GIGACHAT_API_KEY is not set")

        scope = scope or settings.get("GIGACHAT_SCOPE") or "GIGACHAT_API_CORP"
        api_url = api_url or settings.get("GIGACHAT_API_URL") or "https://gigachat.devices.sberbank.ru/api/v1/"
        ca_bundle = ca_bundle_file or _resolve_ca_bundle()

        self._client = GigaChatAsyncClient(
            credentials=credentials,
            model=model_name,
            base_url=api_url,
            verify_ssl_certs=True,
            ca_bundle_file=ca_bundle,
            profanity_check=False,
            scope=scope,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GigaChatEmbeddings:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _aembed(
        self,
        documents: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        model = model or self.model_name
        # Transient failures (429/5xx/network errors) are retried by the SDK.
        # Keeping a single retry layer avoids multiplying SDK and wrapper retries.
        embeddings = await self._client.aembeddings(documents, model=model)
        normalized: list[list[float]] = []
        for embedding in embeddings.data:
            emb = np.array(list(embedding.embedding), dtype=np.float32)
            emb /= np.linalg.norm(emb) + 1e-10
            normalized.append(emb.tolist())
        return normalized

    async def _aembed_batch(
        self,
        texts: list[str],
        *,
        cut_text_symbols: int,
    ) -> list[list[float]]:
        try:
            return await self._aembed(
                [text[:cut_text_symbols] for text in texts],
                model=self.model_name,
            )
        except RequestEntityTooLargeError:
            if len(texts) > 1:
                middle = len(texts) // 2
                logger.warning(
                    "GigaChat embed payload too large, splitting batch %s into %s and %s",
                    len(texts),
                    middle,
                    len(texts) - middle,
                )
                left = await self._aembed_batch(
                    texts[:middle],
                    cut_text_symbols=cut_text_symbols,
                )
                right = await self._aembed_batch(
                    texts[middle:],
                    cut_text_symbols=cut_text_symbols,
                )
                return [*left, *right]

            reduced_cut = int(cut_text_symbols * 0.75)
            if reduced_cut < _MIN_TEXT_SYMBOLS:
                raise
            logger.warning(
                "GigaChat embed text too large, retrying with cut=%s",
                reduced_cut,
            )
            return await self._aembed_batch(
                texts,
                cut_text_symbols=reduced_cut,
            )

    async def aembed_text(self, text: str, cut_text_symbols: int = 12000) -> list[float]:
        result = await self._aembed_batch([text], cut_text_symbols=cut_text_symbols)
        return result[0]

    async def aembed_texts(
        self,
        texts: list[str],
        cut_text_symbols: int = 12000,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(
                await self._aembed_batch(
                    batch,
                    cut_text_symbols=cut_text_symbols,
                )
            )
        return results
