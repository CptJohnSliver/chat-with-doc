"""Build ChatOpenAI clients from role config.

Transport / provider ladder is llm-gateway (``extra_body.models`` on the
primary). App-side ``LocalFallbackRunnable`` covers failures the gateway does
not see: structured-output parse errors and mid-stream disconnects. Fallback
clients omit ``models`` so the gateway does not walk the same list twice.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.llm.config import llm_config
from src.agents.llm.fallback import LocalFallbackRunnable
from src.agents.llm.http import router_kwargs

_CHAT_OPENAI_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "timeout",
    "request_timeout",
    "streaming",
    "max_retries",
    "stream_usage",
)
# reasoning — только extra_body: первоклассный ChatOpenAI(reasoning=...)
# в langchain-openai≥1.4 уходит на POST /v1/responses, которого нет у шлюза.
_EXTRA_BODY_KEYS = ("reasoning", "reasoning_effort", "provider", "plugins", "models", "transforms")
_RESPONSES_API_KEYS = ("reasoning", "reasoning_effort")


def _copy_extra_body_without_models(extra_body: dict) -> dict:
    return {key: value for key, value in extra_body.items() if key != "models"}


def build_llm(
    model: str,
    *,
    name: str,
    **kwargs: Any,
) -> ChatOpenAI | LocalFallbackRunnable:
    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    for key in _RESPONSES_API_KEYS:
        value = kwargs.pop(key, None)
        if value is not None:
            extra_body[key] = value
    chat_kwargs = dict(kwargs)
    if extra_body:
        chat_kwargs["extra_body"] = extra_body

    primary = ChatOpenAI(model=model, name=name, **chat_kwargs)

    fallback_kwargs = dict(chat_kwargs)
    fallback_body = _copy_extra_body_without_models(extra_body)
    if fallback_body:
        fallback_kwargs["extra_body"] = fallback_body
    else:
        fallback_kwargs.pop("extra_body", None)

    seen = {model}
    fallbacks = []
    for fallback_model in extra_body.get("models") or []:
        if not fallback_model or fallback_model in seen:
            continue
        seen.add(fallback_model)
        fallbacks.append(ChatOpenAI(model=fallback_model, name=name, **fallback_kwargs))

    return LocalFallbackRunnable([primary, *fallbacks], name=name) if fallbacks else primary


def build_llm_from_config(role: str, *, name: str) -> ChatOpenAI | LocalFallbackRunnable:
    cfg = llm_config(role)
    model = cfg.get("model")
    if not model:
        raise RuntimeError(f"document_process_llms.{role}.model is required")

    chat_kwargs = {key: cfg[key] for key in _CHAT_OPENAI_KEYS if key in cfg}
    if "timeout" in chat_kwargs and "request_timeout" not in chat_kwargs:
        chat_kwargs["request_timeout"] = chat_kwargs["timeout"]

    extra_body = {key: cfg[key] for key in _EXTRA_BODY_KEYS if key in cfg and cfg[key] is not None}

    return build_llm(
        model=str(model),
        name=name,
        **router_kwargs(operation=role),
        extra_body=extra_body,
        **chat_kwargs,
    )
