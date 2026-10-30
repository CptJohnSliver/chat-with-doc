from __future__ import annotations

import traceback
from http import HTTPStatus
from typing import Any

import pydantic
from pydantic import BaseModel, Field

from src.common.config.config import settings


class BaseError(BaseModel):
    """Pydantic модель ответа для доменных исключений (без FastAPI зависимостей)."""

    user_message: str
    type: str | None = Field(
        "DomainError",
        description="Exception class name",
    )
    detail: str | None = None
    service_name: str | None = settings.get("SERVICE_NAME")
    source: str | None = None
    traceback: list[str] | None = None


class DomainException(Exception):
    """Базовый класс доменных исключений без зависимостей от FastAPI."""

    http_code: int = 400
    detail: Any = "Error occurred"
    user_message: str = "Unexpected error"
    type: str = "DomainException"

    def __init__(
        self,
        detail: Any | None = None,
        user_message: str | None = None,
        status_code: int | None = None,
        exc_type: str | None = None,
        source: Exception | None = None,
    ) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail
        self.user_message = user_message or self.user_message
        self.type = exc_type or self.__class__.__name__
        self.http_code = status_code or self.http_code
        self.source = source
        self.service_name = settings.get("SERVICE_NAME")

    def to_base(self) -> BaseError:
        """Строит ``BaseError`` для ответа клиенту.

        ``detail`` дублирует ``user_message``: оркестратор при HTTP-ошибке навыка
        читает ``body["detail"]``. Технический ``DomainException.detail`` — только в логах.
        """
        return BaseError(
            user_message=self.user_message,
            type=self.type,
            detail=self.user_message,
            service_name=self.service_name,
            source=str(self.source) if self.source and settings.get("DEBUG_MODE") else None,
            traceback=self.trace if settings.get("DEBUG_MODE") else None,
        )

    @property
    def trace(self) -> list[str]:
        return traceback.format_exception(self.source) if self.source else traceback.format_exception(self)

    @property
    def log_data(self) -> dict[str, Any]:
        return {
            "error_type": self.type,
            "http_code": self.http_code,
            "detail": str(self.detail) if self.detail is not None else None,
        }

    @property
    def response_model(self) -> dict:
        model = pydantic.create_model(
            f"{self.__class__.__name__}",
            detail=(str, self.detail),
            user_message=(str, self.user_message),
            type=(str, self.__class__.__name__),
        )
        return {
            self.http_code: {
                "description": HTTPStatus(self.http_code).phrase,
                "model": BaseError,
                "content": {
                    "application/json": {
                        "examples": {
                            model.__name__: {
                                "summary": model.__name__,
                                "value": model().model_dump(),
                            }
                        }
                    }
                },
            }
        }


def get_exception_responses(*args: type[DomainException]) -> dict:
    responses: dict = {}
    for cls in args:
        response = cls().response_model
        http_code = next(iter(response))
        if http_code in responses:
            responses[http_code]["content"]["application/json"]["examples"].update(
                response[http_code]["content"]["application/json"]["examples"]
            )
        else:
            responses.update(response)
    return responses
