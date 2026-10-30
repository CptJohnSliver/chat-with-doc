from .base import DomainException


class ServerException(DomainException):
    http_code = 500
    detail = "Internal server error"
    user_message = "Ошибка сервера"


class NotFound(DomainException):
    http_code = 404
    detail = "Not Found"
    user_message = "Объект не найден"


class Forbidden(DomainException):
    http_code = 403
    detail = "Forbidden"
    user_message = "У вас недостаточно прав для этих действий"


class UnprocessableEntity(DomainException):
    http_code = 422
    detail = "Unprocessable Entity"
    user_message = "Неправильно введены данные"


class BadRequest(DomainException):
    http_code = 400
    detail = "Bad Request"
    user_message = "Некорректный запрос"


class TooManyRequests(DomainException):
    http_code = 429
    detail = "Too Many Requests"
    user_message = "Сервер перегружен, повторите попытку позже"


class ExternalServiceError(DomainException):
    http_code = 502
    detail = "External service error"
    user_message = "Ошибка при обращении к внешнему сервису"


class DocumentTooLargeError(DomainException):
    http_code = 413
    detail = "Document too large"
    user_message = (
        "Файл слишком большой для обработки. Максимальный размер файла — 50 МБ. "
        "Пожалуйста, загрузите документ меньшего размера или разделите его на части."
    )

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            detail=f"File size {size_bytes} exceeds limit {limit_bytes}",
        )


class DocumentNotReadyError(DomainException):
    http_code = 409
    detail = "Document anonymization pending"
    user_message = "Документ ещё обрабатывается (обезличивание). Подождите несколько секунд и отправьте запрос снова."


class DocumentLoadFailedError(ExternalServiceError):
    detail = "Failed to load document"
    user_message = "Не удалось загрузить ваш файл. Попробуйте приложить другой документ."


class InvalidDocumentTextError(UnprocessableEntity):
    detail = "Invalid document text"
    user_message = "Не удалось прочитать текст документа. Проверьте формат файла."
