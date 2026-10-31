"""Tests for modular LLM provider adapters."""

import json
from typing import Any, Dict
import httpx
import pytest

from app.core.config import Settings
from app.models.registry import ModelDescriptor, ModelTier
from app.orchestrator.executor import ModelExecutor, ProviderExecutionError
from app.providers.anthropic import AnthropicProvider
from app.providers.base import BaseHTTPProvider
from app.providers.factory import (
    create_configured_providers,
    register_configured_providers,
)
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


def _dummy_model(provider: str = "openai", model_id: str = "gpt-4o-mini") -> ModelDescriptor:
    return ModelDescriptor(
        model_id=model_id,
        provider=provider,
        tier=ModelTier.LIGHTWEIGHT,
        context_window=128000,
        supported_capabilities=["chat"],
    )


# =====================================================================
# OpenAI Provider Tests
# =====================================================================

@pytest.mark.asyncio
async def test_openai_success():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello OpenAI"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello human!"}}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-test-openai-key", client=client)
    model = _dummy_model("openai", "gpt-4o-mini")

    res = await provider.call("Hello OpenAI", model)

    assert isinstance(res, dict)
    assert res["content"] == "Hello human!"
    assert res["prompt_tokens"] == 12
    assert res["completion_tokens"] == 8
    assert recorded_requests[0].headers["authorization"] == "Bearer sk-test-openai-key"


@pytest.mark.asyncio
async def test_openai_with_context_and_system():
    recorded_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal recorded_payload
        recorded_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "4"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-test", client=client)
    model = _dummy_model("openai", "gpt-4o")

    context = {
        "system_prompt": "You are a calculator.",
        "messages": [
            {"role": "user", "content": "2+1"},
            {"role": "assistant", "content": "3"},
        ],
    }

    res = await provider.call("2+2", model, context=context)

    assert res["content"] == "4"
    assert len(recorded_payload["messages"]) == 4
    assert recorded_payload["messages"][0] == {"role": "system", "content": "You are a calculator."}
    assert recorded_payload["messages"][1] == {"role": "user", "content": "2+1"}
    assert recorded_payload["messages"][2] == {"role": "assistant", "content": "3"}
    assert recorded_payload["messages"][3] == {"role": "user", "content": "2+2"}


@pytest.mark.asyncio
async def test_openai_missing_api_key():
    provider = OpenAIProvider(api_key=None)
    model = _dummy_model("openai", "gpt-4o-mini")
    with pytest.raises(ProviderExecutionError, match="OpenAI API key is not configured"):
        await provider.call("test", model)


@pytest.mark.asyncio
async def test_openai_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API key provided"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="bad-key", client=client)
    model = _dummy_model("openai", "gpt-4o-mini")

    with pytest.raises(ProviderExecutionError, match="OpenAI API returned error HTTP 401"):
        await provider.call("test", model)


@pytest.mark.asyncio
async def test_openai_empty_choices():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [], "usage": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-test", client=client)
    model = _dummy_model("openai", "gpt-4o-mini")

    with pytest.raises(ProviderExecutionError, match="OpenAI response did not contain any choices"):
        await provider.call("test", model)


# =====================================================================
# Anthropic Provider Tests
# =====================================================================

@pytest.mark.asyncio
async def test_anthropic_success_and_model_mapping():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        # Model mapping check: claude-3-5-sonnet -> claude-3-5-sonnet-20241022
        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["max_tokens"] == 4096
        assert body["system"] == "Be concise."
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Anthropic response"}],
                "usage": {"input_tokens": 15, "output_tokens": 25},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="anthropic-secret-key", client=client)
    model = _dummy_model("anthropic", "claude-3-5-sonnet")

    res = await provider.call("Hi Claude", model, context={"system_prompt": "Be concise."})

    assert res["content"] == "Anthropic response"
    assert res["prompt_tokens"] == 15
    assert res["completion_tokens"] == 25
    assert recorded_requests[0].headers["x-api-key"] == "anthropic-secret-key"
    assert recorded_requests[0].headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_anthropic_unmapped_model_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "custom-claude-model-id"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "OK"}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="key", client=client)
    model = _dummy_model("anthropic", "custom-claude-model-id")

    res = await provider.call("test", model)
    assert res["content"] == "OK"


@pytest.mark.asyncio
async def test_anthropic_missing_key():
    provider = AnthropicProvider(api_key="")
    model = _dummy_model("anthropic", "claude-3-haiku")
    with pytest.raises(ProviderExecutionError, match="Anthropic API key is not configured"):
        await provider.call("test", model)


@pytest.mark.asyncio
async def test_anthropic_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [], "usage": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    model = _dummy_model("anthropic", "claude-3-haiku")

    with pytest.raises(ProviderExecutionError, match="did not contain content blocks"):
        await provider.call("test", model)


# =====================================================================
# Google Gemini Provider Tests (Interactions API)
# =====================================================================

@pytest.mark.asyncio
async def test_gemini_success():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        url_str = str(request.url)
        assert url_str.endswith("/v1beta/interactions")
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gemini-3.8-flash"
        assert body["input"] == "Hello Gemini"
        assert body["store"] is False
        assert body["system_instruction"] == "System directive"
        assert body["generation_config"]["thinking_level"] == "low"
        return httpx.Response(
            200,
            json={
                "id": "int_123",
                "status": "completed",
                "steps": [
                    {
                        "type": "thought",
                        "signature": "thought-sig",
                    },
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "Gemini interaction response text"}],
                    },
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 16,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-secret-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call(
        "Hello Gemini",
        model,
        context={"system_prompt": "System directive"},
    )

    assert res["content"] == "Gemini interaction response text"
    assert res["prompt_tokens"] == 9
    assert res["completion_tokens"] == 16
    assert recorded_requests[0].headers["x-goog-api-key"] == "gemini-secret-key"


@pytest.mark.asyncio
async def test_gemini_interactions_output_text_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "int_456",
                "status": "completed",
                "output_text": "Direct output text response",
                "usage": {"input_tokens": 7, "output_tokens": 12},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call("ping", model)
    assert res["content"] == "Direct output text response"
    assert res["prompt_tokens"] == 7
    assert res["completion_tokens"] == 12


@pytest.mark.asyncio
async def test_gemini_conversation_history_mapping():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert "User: First question" in body["input"]
        assert "Model: First answer" in body["input"]
        assert "User: Follow-up question" in body["input"]
        return httpx.Response(
            200,
            json={
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "Multi-turn reply"}],
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    context = {
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"},
        ]
    }
    res = await provider.call("Follow-up question", model, context=context)
    assert res["content"] == "Multi-turn reply"


@pytest.mark.asyncio
async def test_gemini_generation_config_forwarded():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert "generation_config" in body
        assert body["generation_config"]["temperature"] == 0.5
        assert body["generation_config"]["thinking_level"] == "low"
        return httpx.Response(
            200,
            json={
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "pong"}],
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    context = {
        "generation_config": {
            "temperature": 0.5,
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    res = await provider.call("ping", model, context=context)
    assert res["content"] == "pong"
    assert len(recorded_requests) == 1


@pytest.mark.asyncio
async def test_gemini_missing_output_and_usage_fields():
    # 1. Missing output raises ProviderExecutionError
    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"steps": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(empty_handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    with pytest.raises(ProviderExecutionError, match="empty content"):
        await provider.call("test", model)

    # 2. Missing usage fields defaults to 0 safely
    def no_usage_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "Safe response"}],
                    }
                ]
            },
        )

    client2 = httpx.AsyncClient(transport=httpx.MockTransport(no_usage_handler))
    provider2 = GeminiProvider(api_key="gemini-key", client=client2)
    res2 = await provider2.call("test", model)
    assert res2["content"] == "Safe response"
    assert res2["prompt_tokens"] == 0
    assert res2["completion_tokens"] == 0


@pytest.mark.asyncio
async def test_gemini_http_errors_400_404_429():
    # 400 Bad Request
    def handler_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid argument"}})

    client_400 = httpx.AsyncClient(transport=httpx.MockTransport(handler_400))
    provider_400 = GeminiProvider(api_key="key", client=client_400)
    model = _dummy_model("google", "gemini-3.8-flash")
    with pytest.raises(ProviderExecutionError, match="HTTP 400"):
        await provider_400.call("test", model)

    # 404 Not Found
    def handler_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "Model not found"}})

    client_404 = httpx.AsyncClient(transport=httpx.MockTransport(handler_404))
    provider_404 = GeminiProvider(api_key="key", client=client_404)
    with pytest.raises(ProviderExecutionError, match="HTTP 404"):
        await provider_404.call("test", model)

    # 429 Rate Limit (retries exhausted)
    calls_429 = 0
    def handler_429(request: httpx.Request) -> httpx.Response:
        nonlocal calls_429
        calls_429 += 1
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    client_429 = httpx.AsyncClient(transport=httpx.MockTransport(handler_429))
    provider_429 = GeminiProvider(api_key="key", client=client_429, max_retries=1, retry_delay_seconds=0.0)
    with pytest.raises(ProviderExecutionError, match="HTTP 429"):
        await provider_429.call("test", model)
    assert calls_429 == 2


@pytest.mark.asyncio
async def test_smoke_test_gemini_supplies_low_thinking_level():
    """Verify scripts/smoke_test_gemini.py supplies low thinkingLevel in context."""
    import inspect
    import scripts.smoke_test_gemini as smoke_mod

    source = inspect.getsource(smoke_mod.run_smoke_test)
    assert '"thinkingLevel": "low"' in source or "'thinkingLevel': 'low'" in source
    assert "generation_config" in source


@pytest.mark.asyncio
async def test_gemini_interactions_none_output_text_with_valid_steps():
    """Verify response with output_text=None but valid steps is safely parsed."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "int_789",
                "status": "completed",
                "output_text": None,
                "steps": [
                    {
                        "type": "thought",
                        "signature": "sig123",
                    },
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": "GEMINI_OK"}
                        ],
                    },
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call("ping", model)
    assert res["content"] == "GEMINI_OK"
    assert res["prompt_tokens"] == 12
    assert res["completion_tokens"] == 4


@pytest.mark.asyncio
async def test_gemini_503_high_demand_recovers_on_retry():
    """Verify HTTP 503 high-demand error triggers retry and recovers on subsequent attempt."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                503,
                json={"error": {"message": "gemini-3.8-flash is currently experiencing high demand. Please try again later."}},
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "Recovered after 503"}]}
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(
        api_key="gemini-key",
        client=client,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call("ping", model)
    assert res["content"] == "Recovered after 503"
    assert call_count == 2


@pytest.mark.asyncio
async def test_gemini_retry_after_header_supported():
    """Verify Retry-After header is captured on transient 503/429 errors."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            503,
            headers={"Retry-After": "2"},
            json={"error": {"message": "Service unavailable"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(
        api_key="gemini-key",
        client=client,
        max_retries=0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    caught = None
    try:
        await provider.call("ping", model)
    except ProviderExecutionError as exc:
        caught = exc

    assert caught is not None
    assert caught.status_code == 503
    assert caught.retry_after == 2.0


@pytest.mark.asyncio
async def test_gemini_non_transient_errors_never_retried():
    """Verify client errors (400, 401, 403, 404) are never retried even with max_retries > 0."""
    call_counts = {}

    def make_handler(status_code: int):
        def handler(request: httpx.Request) -> httpx.Response:
            call_counts[status_code] = call_counts.get(status_code, 0) + 1
            return httpx.Response(status_code, json={"error": {"message": f"Client error {status_code}"}})
        return handler

    for code in (400, 401, 403, 404):
        client = httpx.AsyncClient(transport=httpx.MockTransport(make_handler(code)))
        provider = GeminiProvider(api_key="key", client=client, max_retries=3, retry_delay_seconds=0.0)
        model = _dummy_model("google", "gemini-3.8-flash")
        with pytest.raises(ProviderExecutionError, match=f"HTTP {code}"):
            await provider.call("test", model)
        assert call_counts[code] == 1, f"Expected exactly 1 call for HTTP {code}, got {call_counts[code]}"


@pytest.mark.asyncio
async def test_gemini_time_budget_exhaustion_aborts_retry():
    """Verify retry is aborted when remaining time budget is insufficient."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "Temporarily overloaded"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # Configure timeout_seconds=1.0 with retry_delay_seconds=2.0 so attempt 1 runs,
    # but the retry delay exceeds the remaining time budget
    provider = GeminiProvider(
        api_key="key",
        client=client,
        timeout_seconds=1.0,
        max_retries=2,
        retry_delay_seconds=2.0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    with pytest.raises(ProviderExecutionError):
        await provider.call("test", model)
    assert calls == 1


@pytest.mark.asyncio
async def test_gemini_first_attempt_timeout_recovers_on_retry():
    """Verify first-attempt timeout is retried and succeeds on second attempt within budget."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("Request timed out reading response")
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "Recovered after timeout"}]}
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(
        api_key="gemini-key",
        client=client,
        timeout_seconds=30.0,
        attempt_timeout_seconds=10.0,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call("ping", model)
    assert res["content"] == "Recovered after timeout"
    assert call_count == 2


@pytest.mark.asyncio
async def test_gemini_reporting_of_underlying_timeout_type():
    """Verify underlying timeout exception types (ReadTimeout, ConnectTimeout) are reported in cause_type."""
    def read_timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(read_timeout_handler))
    provider = GeminiProvider(api_key="key", client=client, max_retries=0)
    model = _dummy_model("google", "gemini-3.8-flash")

    caught_read = None
    try:
        await provider.call("ping", model)
    except ProviderExecutionError as exc:
        caught_read = exc

    assert caught_read is not None
    assert caught_read.cause_type == "ReadTimeout"

    def connect_timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connect timed out")

    client2 = httpx.AsyncClient(transport=httpx.MockTransport(connect_timeout_handler))
    provider2 = GeminiProvider(api_key="key", client=client2, max_retries=0)

    caught_connect = None
    try:
        await provider2.call("ping", model)
    except ProviderExecutionError as exc:
        caught_connect = exc

    assert caught_connect is not None
    assert caught_connect.cause_type == "ConnectTimeout"


@pytest.mark.asyncio
async def test_gemini_missing_key():
    provider = GeminiProvider(api_key=None)
    model = _dummy_model("google", "gemini-3.8-flash")
    with pytest.raises(ProviderExecutionError, match="Google Gemini API key is not configured"):
        await provider.call("test", model)


# =====================================================================
# Base Provider and Common Error Tests
# =====================================================================

@pytest.mark.asyncio
async def test_provider_timeout_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="key", client=client, timeout_seconds=5.0)
    model = _dummy_model("openai", "gpt-4o-mini")

    with pytest.raises(ProviderExecutionError, match="timed out after 5.0s"):
        await provider.call("prompt", model)


@pytest.mark.asyncio
async def test_provider_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="key", client=client)
    model = _dummy_model("openai", "gpt-4o-mini")

    with pytest.raises(ProviderExecutionError, match="Network communication failure"):
        await provider.call("prompt", model)


def test_provider_has_api_key_property():
    p1 = OpenAIProvider(api_key="sk-test")
    p2 = OpenAIProvider(api_key="")
    p3 = OpenAIProvider(api_key=None)

    assert p1.has_api_key is True
    assert p2.has_api_key is False
    assert p3.has_api_key is False


# =====================================================================
# Provider Factory Tests
# =====================================================================

def test_factory_unconfigured_settings():
    settings = Settings(
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        GEMINI_API_KEY=None,
    )
    providers = create_configured_providers(settings=settings)
    assert providers == {}

    executor = ModelExecutor()
    registered = register_configured_providers(executor, settings=settings)
    assert registered == []
    assert executor.list_providers() == []


def test_factory_partial_configuration():
    settings = Settings(
        OPENAI_API_KEY="sk-test",
        ANTHROPIC_API_KEY=None,
        GEMINI_API_KEY=None,
    )
    providers = create_configured_providers(settings=settings)
    assert list(providers.keys()) == ["openai"]
    assert isinstance(providers["openai"], OpenAIProvider)

    executor = ModelExecutor()
    registered = register_configured_providers(executor, settings=settings)
    assert registered == ["openai"]
    assert "openai" in executor.list_providers()
    assert "anthropic" not in executor.list_providers()


def test_factory_all_configured():
    settings = Settings(
        OPENAI_API_KEY="sk-openai",
        ANTHROPIC_API_KEY="sk-anthropic",
        GEMINI_API_KEY="sk-gemini",
    )
    providers = create_configured_providers(settings=settings)
    assert set(providers.keys()) == {"openai", "anthropic", "google"}

    executor = ModelExecutor()
    registered = register_configured_providers(executor, settings=settings)
    assert set(registered) == {"openai", "anthropic", "google"}
    assert set(executor.list_providers()) == {"openai", "anthropic", "google"}


def test_factory_gemini_custom_timeout_and_retries():
    settings = Settings(
        GEMINI_API_KEY="sk-gemini",
        GEMINI_TIMEOUT_SECONDS=45.0,
        GEMINI_MAX_RETRIES=2,
    )
    providers = create_configured_providers(settings=settings)
    assert "google" in providers
    gemini_p = providers["google"]
    assert isinstance(gemini_p, GeminiProvider)
    assert gemini_p.timeout_seconds == 45.0
    assert gemini_p.max_retries == 2


@pytest.mark.asyncio
async def test_gemini_transient_retry_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("Request timed out")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Recovered response"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(
        api_key="gemini-key",
        client=client,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    res = await provider.call("Test prompt", model)
    assert res["content"] == "Recovered response"
    assert call_count == 2


@pytest.mark.asyncio
async def test_gemini_transient_retry_exhausted_raises():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(
        api_key="gemini-key",
        client=client,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    model = _dummy_model("google", "gemini-3.8-flash")

    with pytest.raises(ProviderExecutionError, match="HTTP 503"):
        await provider.call("Test prompt", model)
    assert call_count == 2

