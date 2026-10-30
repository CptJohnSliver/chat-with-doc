from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def _parse_level(level_name: str | None, default: int = logging.INFO) -> int:
    name = (level_name or "").strip().upper()
    return getattr(logging, name, default)


def _get_log_level() -> int:
    return _parse_level(os.environ.get("API_LOG_LEVEL", "INFO"), default=logging.INFO)


def _get_renderer() -> str:
    return os.environ.get("LOG_RENDERER", "json").lower()


def _is_file_log_enabled() -> bool:
    return os.environ.get("ENABLE_FILE_LOG", "false").lower() == "true"


def _get_log_path() -> Path:
    return Path(os.environ.get("API_LOG_PATH", "./logs"))


def _get_max_bytes() -> int:
    return int(os.environ.get("LOG_MAX_BYTES", str(50 * 1024 * 1024)))


def _get_backup_count() -> int:
    return int(os.environ.get("LOG_BACKUP_COUNT", "5"))


def _get_log_name() -> str:
    return os.environ.get("API_LOG_NAME", "API")


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level


class MinLevelFilter(logging.Filter):
    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


def _reset_root_logger(root: logging.Logger, level: int) -> None:
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)


def _make_processor_formatter() -> logging.Formatter:
    renderer_name = _get_renderer()
    renderer = (
        structlog.processors.JSONRenderer(serializer=lambda obj, **kwargs: json.dumps(obj, ensure_ascii=False, **kwargs))
        if renderer_name == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    pre_chain: list[structlog.typing.Processor] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    return structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=pre_chain,
    )


def _configure_stream_handlers(root: logging.Logger, formatter: logging.Formatter) -> None:
    h = logging.StreamHandler(stream=sys.stdout)
    h.setLevel(_get_log_level())
    h.setFormatter(formatter)
    root.addHandler(h)


def _make_file_handler(path: Path) -> logging.Handler:
    return RotatingFileHandler(
        filename=str(path),
        maxBytes=_get_max_bytes(),
        backupCount=_get_backup_count(),
        encoding="utf-8",
    )


def _configure_file_handlers(root: logging.Logger, formatter: logging.Formatter) -> None:
    if not _is_file_log_enabled():
        return

    log_dir = _get_log_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = _get_log_name()
    app_path = log_dir / f"{log_name}.log"
    err_path = log_dir / f"{log_name}_error.log"

    h_app = _make_file_handler(app_path)
    h_app.setLevel(_get_log_level())
    h_app.addFilter(MaxLevelFilter(logging.ERROR))
    h_app.setFormatter(formatter)
    root.addHandler(h_app)

    h_err = _make_file_handler(err_path)
    h_err.setLevel(logging.ERROR)
    h_err.addFilter(MinLevelFilter(logging.ERROR))
    h_err.setFormatter(formatter)
    root.addHandler(h_err)


def _configure_structlog() -> None:
    processors: list[structlog.typing.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_third_party_loggers() -> None:
    # FastMCP attaches its own RichHandler to the "fastmcp" logger at import
    # time with propagate=False, bypassing structlog and emitting 80-column
    # wrapped lines. Strip it so its records flow through the root handlers.
    fastmcp_logger = logging.getLogger("fastmcp")
    for handler in list(fastmcp_logger.handlers):
        fastmcp_logger.removeHandler(handler)
    fastmcp_logger.propagate = True
    # ctx.info/ctx.log echo every client notification (full document payloads)
    # to this logger — keep them out of server logs.
    logging.getLogger("fastmcp.server.context.to_client").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def _configure_once() -> None:
    level = _get_log_level()
    root = logging.getLogger()
    _reset_root_logger(root, level)

    formatter = _make_processor_formatter()

    _configure_stream_handlers(root, formatter)
    _configure_file_handlers(root, formatter)
    _configure_structlog()
    _configure_third_party_loggers()


def configure_logging() -> None:
    _configure_once()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or _get_log_name())


def bind_context(**kwargs) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
