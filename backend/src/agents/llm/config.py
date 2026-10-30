"""Load per-role LLM settings from ``[default.document_process_llms.*]`` in api.toml."""

from __future__ import annotations

from typing import Any

from src.common.config.config import settings


def _as_plain(value: Any) -> Any:
    """Convert Dynaconf boxes / nested structures into plain Python values."""
    if type(value).__name__ in {"DataDict", "DynaBox"} or (
        hasattr(value, "to_dict") and not isinstance(value, (dict, list, tuple, str, bytes))
    ):
        # Dynaconf v3: DataDict.to_dict() is deprecated in favor of dict(...).
        try:
            value = dict(value)
        except Exception:
            value = value.to_dict()
    if isinstance(value, dict):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def llm_config(role: str) -> dict[str, Any]:
    llms = settings.get("document_process_llms")
    if llms is None:
        raise RuntimeError("document_process_llms is not configured. Add [default.document_process_llms.*] sections to api.toml.")
    raw = llms.get(role) if hasattr(llms, "get") else getattr(llms, role, None)
    if raw is None:
        raise RuntimeError(f"Missing document_process_llms.{role} in settings (api.toml).")
    cfg = _as_plain(raw)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"document_process_llms.{role} must be a table, got {type(cfg).__name__}")
    return cfg


def get_llm_timeout(role: str) -> float | None:
    """Return the configured wall-clock timeout for a document-process LLM role."""
    timeout = llm_config(role).get("timeout")
    if timeout is None:
        return None
    return float(timeout)
