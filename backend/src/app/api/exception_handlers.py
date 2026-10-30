"""Global FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette import status

from src.app.api.utils.stub_response import VALIDATION_USER_MESSAGE
from src.common.exceptions import BaseError, DomainException
from src.common.exceptions.tools import metrics_exception_type, wrap_exception
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def _request_context(request: Request) -> dict[str, str]:
    return {
        "path": request.url.path,
        "method": request.method,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainException)
    async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
        await logger.aerror(
            "domain_exception",
            **_request_context(request),
            **exc.log_data,
            exc_info=True,
        )
        return JSONResponse(
            status_code=exc.http_code,
            content=exc.to_base().model_dump(exclude_none=True, mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        err_msg = [{k: v for k, v in msg.items() if k not in {"input", "ctx"}} for msg in exc.errors()]
        await logger.aerror(
            "request_validation_error",
            **_request_context(request),
            errors=err_msg,
            exc_info=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=BaseError(
                user_message=VALIDATION_USER_MESSAGE,
                type="RequestValidationError",
                detail=VALIDATION_USER_MESSAGE,
            ).model_dump(exclude_none=True, mode="json"),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        err_msg = [{k: v for k, v in msg.items() if k not in {"input", "ctx"}} for msg in exc.errors()]
        await logger.aerror(
            "pydantic_validation_error",
            **_request_context(request),
            errors=err_msg,
            exc_info=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=BaseError(
                user_message=VALIDATION_USER_MESSAGE,
                type="ValidationError",
                detail=VALIDATION_USER_MESSAGE,
            ).model_dump(exclude_none=True, mode="json"),
        )

    @app.exception_handler(Exception)
    async def unknown_error_handler(request: Request, exc: Exception) -> JSONResponse:
        wrapped = wrap_exception(exc)
        await logger.aerror(
            "unexpected_error",
            **_request_context(request),
            error_type=metrics_exception_type(wrapped, original=exc),
            detail=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=wrapped.to_base().model_dump(exclude_none=True, mode="json"),
        )
