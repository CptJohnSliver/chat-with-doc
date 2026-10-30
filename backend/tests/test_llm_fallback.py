import asyncio
from typing import Any

import httpx
import pytest
from openai import APIStatusError
from pydantic import BaseModel, ValidationError
from src.agents.llm import invoke as llm_utils
from src.agents.llm.fallback import LocalFallbackRunnable, should_fallback_to_next_model
from src.agents.llm.invoke import (
    _collect_token_usage,
    _stream_chunk_finish_reason,
    _stream_chunk_to_text,
    astream_with_retries,
    call_llm_structured_async_with_timeout,
)
from tenacity import stop_after_attempt, wait_none


@pytest.mark.parametrize(
    ("exc", "delivered_content", "expected"),
    [
        (ValidationError.from_exception_data("Result", []), False, True),
        (ValueError("No JSON object found in LLM output"), False, True),
        (TimeoutError("client budget"), False, False),
        (ConnectionError("closed"), False, False),
        (ConnectionError("closed"), True, True),
    ],
)
def test_should_fallback_to_next_model(exc: Exception, delivered_content: bool, expected: bool) -> None:
    assert should_fallback_to_next_model(exc, delivered_content=delivered_content) is expected


def test_should_fallback_http_status_error() -> None:
    request = httpx.Request("POST", "http://llm-gateway/v1/chat/completions")
    response = httpx.Response(502, request=request)
    exc = APIStatusError("upstream failed", response=response, body=None)
    assert should_fallback_to_next_model(exc) is False
    assert should_fallback_to_next_model(exc, delivered_content=True) is True


class Result(BaseModel):
    value: str


class FlakyThenWorkingAdapter:
    """Fails the first N calls, then returns a valid structured/stream result."""

    model_name = "primary"

    def __init__(self, *, fail_times: int = 1) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def with_structured_output(self, schema: type[BaseModel]) -> "FlakyThenWorkingAdapter":
        return self

    async def ainvoke(self, _messages: Any) -> Result:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValidationError.from_exception_data("Result", [])
        return Result(value="recovered response")

    async def astream(self, _messages: Any):
        self.calls += 1
        if self.calls <= self.fail_times:
            yield "discard this partial response"
            raise ConnectionError("adapter disconnected mid-stream")
        yield "recovered response"


def _two_attempts(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_utils,
        "_RETRY_KWARGS",
        {
            "stop": stop_after_attempt(2),
            "wait": wait_none(),
            "reraise": True,
        },
    )


def test_async_structured_call_retries_same_client_after_schema_failure(
    monkeypatch,
) -> None:
    _two_attempts(monkeypatch)
    adapter = FlakyThenWorkingAdapter()

    result = asyncio.run(
        call_llm_structured_async_with_timeout(
            adapter,
            "question",
            Result,
            timeout_s=1,
        )
    )

    assert result == Result(value="recovered response")
    assert adapter.calls == 2


def test_stream_retries_same_client_after_failure(monkeypatch) -> None:
    _two_attempts(monkeypatch)
    adapter = FlakyThenWorkingAdapter(fail_times=1)

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in astream_with_retries(
                adapter,
                "question",
                stream_mode="delta",
            )
        ]

    assert asyncio.run(collect()) == ["discard this partial response", "", "recovered response"]
    assert adapter.calls == 2


def test_stream_restart_discards_partial_response_on_retry(monkeypatch) -> None:
    _two_attempts(monkeypatch)
    adapter = FlakyThenWorkingAdapter()

    async def collect_final_response() -> str:
        response = ""
        async for chunk in astream_with_retries(
            adapter,
            "question",
            stream_mode="delta",
        ):
            if not chunk:
                response = ""
            response += chunk
        return response

    assert asyncio.run(collect_final_response()) == "recovered response"
    assert adapter.calls == 2


class BrokenSchemaAdapter:
    model_name = "primary"

    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, schema: type[BaseModel]) -> "BrokenSchemaAdapter":
        return self

    async def ainvoke(self, _messages: Any) -> Result:
        self.calls += 1
        raise ValidationError.from_exception_data("Result", [])

    async def astream(self, _messages: Any):
        self.calls += 1
        yield "discard this partial response"
        raise ConnectionError("adapter disconnected mid-stream")


class WorkingAdapter:
    model_name = "fallback"

    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, schema: type[BaseModel]) -> "WorkingAdapter":
        return self

    async def ainvoke(self, _messages: Any) -> Result:
        self.calls += 1
        return Result(value="fallback response")

    async def astream(self, _messages: Any):
        self.calls += 1
        yield "fallback response"


def _one_attempt(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_utils,
        "_RETRY_KWARGS",
        {
            "stop": stop_after_attempt(1),
            "wait": wait_none(),
            "reraise": True,
        },
    )


def test_structured_validation_error_switches_to_fallback_model(monkeypatch) -> None:
    _one_attempt(monkeypatch)
    primary = BrokenSchemaAdapter()
    fallback = WorkingAdapter()

    result = asyncio.run(
        call_llm_structured_async_with_timeout(
            LocalFallbackRunnable([primary, fallback]),
            "question",
            Result,
            timeout_s=1,
        )
    )

    assert result == Result(value="fallback response")
    assert primary.calls == 1
    assert fallback.calls == 1


def test_mid_stream_failure_switches_to_fallback_model(monkeypatch) -> None:
    _one_attempt(monkeypatch)
    primary = BrokenSchemaAdapter()
    fallback = WorkingAdapter()

    async def collect_final_response() -> str:
        response = ""
        async for chunk in astream_with_retries(
            LocalFallbackRunnable([primary, fallback]),
            "question",
            stream_mode="delta",
        ):
            if not chunk:
                response = ""
            response += chunk
        return response

    assert asyncio.run(collect_final_response()) == "fallback response"
    assert primary.calls == 1
    assert fallback.calls == 1


class GatewayExhaustedAdapter:
    model_name = "primary"

    def __init__(self, exc: Exception) -> None:
        self.calls = 0
        self._exc = exc

    def with_structured_output(self, schema: type[BaseModel]) -> "GatewayExhaustedAdapter":
        return self

    async def ainvoke(self, _messages: Any) -> Result:
        self.calls += 1
        raise self._exc

    async def astream(self, _messages: Any):
        self.calls += 1
        raise self._exc
        yield  # pragma: no cover


def _gateway_status_error(status_code: int = 502) -> APIStatusError:
    request = httpx.Request("POST", "http://llm-gateway/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("upstream failed", response=response, body=None)


def test_structured_http_error_does_not_switch_model(monkeypatch) -> None:
    _one_attempt(monkeypatch)
    primary = GatewayExhaustedAdapter(_gateway_status_error())
    fallback = WorkingAdapter()

    with pytest.raises(APIStatusError):
        asyncio.run(
            call_llm_structured_async_with_timeout(
                LocalFallbackRunnable([primary, fallback]),
                "question",
                Result,
                timeout_s=1,
            )
        )

    assert primary.calls == 1
    assert fallback.calls == 0


def test_structured_timeout_does_not_switch_model(monkeypatch) -> None:
    _one_attempt(monkeypatch)
    primary = GatewayExhaustedAdapter(TimeoutError("client budget"))
    fallback = WorkingAdapter()

    with pytest.raises(TimeoutError):
        asyncio.run(
            call_llm_structured_async_with_timeout(
                LocalFallbackRunnable([primary, fallback]),
                "question",
                Result,
                timeout_s=1,
            )
        )

    assert primary.calls == 1
    assert fallback.calls == 0


def test_pre_content_stream_error_does_not_switch_model(monkeypatch) -> None:
    _one_attempt(monkeypatch)
    primary = GatewayExhaustedAdapter(ConnectionError("gateway closed before tokens"))
    fallback = WorkingAdapter()

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in astream_with_retries(
                LocalFallbackRunnable([primary, fallback]),
                "question",
                stream_mode="delta",
            )
        ]

    with pytest.raises(ConnectionError):
        asyncio.run(collect())

    assert primary.calls == 1
    assert fallback.calls == 0


def test_router_kwargs_set_gateway_attribution_headers(monkeypatch) -> None:
    class FakeSettings:
        def get(self, key: str, default: Any = None) -> Any:
            return {
                "ROUTER_AI_API_KEY": "test-key",
                "ROUTER_AI_BASE_URL": "http://llm-gateway:8000/v1",
                "SERVICE_NAME": "chat-with-documents",
            }.get(key, default)

    monkeypatch.setattr("src.agents.llm.http.settings", FakeSettings())
    monkeypatch.setattr("src.agents.llm.http.get_http_clients", lambda: (None, None))

    from src.agents.llm.http import router_kwargs

    headers = router_kwargs(operation="validation")["default_headers"]
    assert headers["X-Service-Name"] == "chat-with-documents"
    assert headers["X-LLM-Operation"] == "validation"


def test_build_llm_from_config_passes_role_as_operation_header(monkeypatch) -> None:
    captured: list[dict] = []

    class CapturingChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("src.agents.llm.factory.ChatOpenAI", CapturingChatOpenAI)
    monkeypatch.setattr(
        "src.agents.llm.factory.llm_config",
        lambda _role: {"model": "openai/gpt-oss-120b", "temperature": 0.5},
    )
    monkeypatch.setattr(
        "src.agents.llm.factory.router_kwargs",
        lambda operation=None: {
            "api_key": "test-key",
            "base_url": "http://llm-gateway:8000/v1",
            "default_headers": {
                "X-Service-Name": "chat-with-documents",
                "X-LLM-Operation": operation,
            },
        },
    )

    from src.agents.llm.factory import build_llm_from_config

    build_llm_from_config("validation", name="document_validation")

    assert captured[0]["default_headers"] == {
        "X-Service-Name": "chat-with-documents",
        "X-LLM-Operation": "validation",
    }


def test_reasoning_goes_to_extra_body_not_chatopenai_kwarg(monkeypatch) -> None:
    """First-class reasoning= switches langchain-openai≥1.4 to POST /v1/responses."""
    captured: list[dict] = []

    class CapturingChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("src.agents.llm.factory.ChatOpenAI", CapturingChatOpenAI)

    from src.agents.llm.factory import build_llm

    build_llm(
        "openai/gpt-oss-120b",
        name="document_validation",
        reasoning={"effort": "medium", "exclude": False, "enabled": True},
        extra_body={
            "models": ["deepseek/deepseek-v4-flash"],
            "provider": {"sort": "throughput"},
        },
    )

    assert len(captured) == 2
    assert captured[0]["model"] == "openai/gpt-oss-120b"
    assert captured[1]["model"] == "deepseek/deepseek-v4-flash"
    for kwargs in captured:
        assert kwargs["name"] == "document_validation"
        assert "reasoning" not in kwargs
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"
        assert kwargs["extra_body"]["provider"] == {"sort": "throughput"}
    assert captured[0]["extra_body"]["models"] == ["deepseek/deepseek-v4-flash"]
    assert "models" not in captured[1]["extra_body"]


def test_build_llm_from_config_forwards_models_to_gateway(monkeypatch) -> None:
    captured: list[dict] = []

    class CapturingChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("src.agents.llm.factory.ChatOpenAI", CapturingChatOpenAI)
    monkeypatch.setattr(
        "src.agents.llm.factory.llm_config",
        lambda _role: {
            "model": "openai/gpt-oss-120b",
            "models": ["deepseek/deepseek-v4-flash"],
        },
    )
    monkeypatch.setattr(
        "src.agents.llm.factory.router_kwargs",
        lambda operation=None: {"api_key": "test-key", "base_url": "http://llm-gateway:8000/v1"},
    )

    from src.agents.llm.factory import build_llm_from_config

    build_llm_from_config("validation", name="document_validation")

    assert len(captured) == 2
    assert captured[0]["extra_body"]["models"] == ["deepseek/deepseek-v4-flash"]
    assert "extra_body" not in captured[1]


def test_stream_chunk_skips_reasoning_blocks() -> None:
    """GLM / Responses-style streams mix reasoning + text parts."""
    chunk = type(
        "Chunk",
        (),
        {
            "content": [
                {"type": "reasoning", "text": "think silently"},
                {"type": "text", "text": "Итог саммари"},
            ]
        },
    )()
    assert _stream_chunk_to_text(chunk) == "Итог саммари"

    assert (
        _stream_chunk_to_text(
            {"content": [{"type": "output_text", "text": "ok"}, {"type": "thinking"}]}
        )
        == "ok"
    )


def test_stream_reports_finish_reason_from_metadata() -> None:
    class FinishReasonAdapter:
        async def astream(self, _messages: Any):
            yield type(
                "Chunk",
                (),
                {"content": "answer", "response_metadata": {}},
            )()
            yield type(
                "Chunk",
                (),
                {
                    "content": "",
                    "response_metadata": {"finish_reason": "length"},
                },
            )()

    async def collect() -> tuple[list[str], list[str | None]]:
        reasons: list[str | None] = []
        chunks = [
            chunk
            async for chunk in astream_with_retries(
                FinishReasonAdapter(),
                "question",
                stream_mode="delta",
                on_finish_reason=reasons.append,
            )
        ]
        return chunks, reasons

    assert asyncio.run(collect()) == (["answer"], [None, "length"])


def test_blank_stream_finish_reason_is_missing() -> None:
    chunk = type(
        "Chunk",
        (),
        {"response_metadata": {"finish_reason": "   "}},
    )()

    assert _stream_chunk_finish_reason(chunk) is None


def test_collect_token_usage_sums_all_models() -> None:
    callback = type(
        "Callback",
        (),
        {
            "usage_metadata": {
                "primary": {"total_tokens": 120},
                "fallback": {"total_tokens": 35},
            }
        },
    )()

    assert _collect_token_usage(callback) == 155
