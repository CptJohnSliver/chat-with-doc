"""HTTP clients and Router AI connection kwargs."""

from __future__ import annotations

import ssl
from typing import Any

import httpx

from src.common.config.config import settings
from src.infrastructure.request_identity import (
    attach_identity_headers,
    attach_identity_headers_async,
)

_CACHE: dict[str, Any] = {}


def _make_tls12_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def _make_http_client_kwargs() -> dict[str, Any] | None:
    client_kwargs: dict[str, Any] = {}

    proxy_url = settings.get("PROXY_URL")
    if proxy_url and settings.get("USE_PROXY", True):
        client_kwargs["proxy"] = proxy_url

    if settings.get("ROUTER_AI_FORCE_TLS12", False):
        client_kwargs["verify"] = _make_tls12_ssl_context()

    if not client_kwargs:
        return None

    client_kwargs["timeout"] = 90
    return client_kwargs


def get_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    cached = _CACHE.get("_http_clients")
    if cached is not None:
        return cached
    kwargs = _make_http_client_kwargs() or {}
    kwargs.setdefault("timeout", 90)
    existing_hooks = list(kwargs.get("event_hooks", {}).get("request", []))
    # httpx 0.28 AsyncClient always awaits request hooks; a sync hook returns None
    # and OpenAI wraps that TypeError as APIConnectionError ("Connection error").
    sync_kwargs = {**kwargs, "event_hooks": {"request": [*existing_hooks, attach_identity_headers]}}
    async_kwargs = {**kwargs, "event_hooks": {"request": [*existing_hooks, attach_identity_headers_async]}}
    clients = (httpx.Client(**sync_kwargs), httpx.AsyncClient(**async_kwargs))
    _CACHE["_http_clients"] = clients
    return clients


def gateway_headers(*, operation: str | None = None) -> dict[str, str]:
    """Заголовки учёта шлюза: сервис обязателен, операция — роль LLM."""
    headers: dict[str, str] = {}
    service_name = settings.get("SERVICE_NAME")
    if service_name:
        headers["X-Service-Name"] = str(service_name)
    if operation:
        headers["X-LLM-Operation"] = operation
    return headers


def router_kwargs(*, operation: str | None = None) -> dict[str, Any]:
    api_key = settings.get("ROUTER_AI_API_KEY")
    if not api_key:
        raise RuntimeError("ROUTER_AI_API_KEY is not set. Copy backend/env.local.example to .env and fill secrets.")
    sync_client, async_client = get_http_clients()
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": settings.get("ROUTER_AI_BASE_URL", "https://routerai.ru/api/v1"),
        "http_client": sync_client,
        "http_async_client": async_client,
    }
    headers = gateway_headers(operation=operation)
    if headers:
        kwargs["default_headers"] = headers
    return kwargs
