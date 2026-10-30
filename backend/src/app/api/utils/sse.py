from __future__ import annotations

import json
from typing import Any


def sse_pack(data: dict[str, Any], event: str | None = None) -> str:
    """
    Собирает SSE-сообщение.
    Можно добавить "event: <name>" при желании.
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"
