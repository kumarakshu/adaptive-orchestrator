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
# Google Gemini Provider Tests
# =====================================================================

@pytest.mark.asyncio
async def test_gemini_success():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        url_str = str(request.url)
        assert "gemini-3.8-flash:generateContent" in url_str
        body = json.loads(request.content.decode("utf-8"))
        assert len(body["contents"]) == 1
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "Hello Gemini"
        assert "system_instruction" in body
        assert "generationConfig" not in body
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini response text"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 16,
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

    assert res["content"] == "Gemini response text"
    assert res["prompt_tokens"] == 9
    assert res["completion_tokens"] == 16
    assert recorded_requests[0].headers["x-goog-api-key"] == "gemini-secret-key"


@pytest.mark.asyncio
async def test_gemini_generation_config_forwarded():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert "generationConfig" in body
        assert body["generationConfig"] == {
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "pong"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    context = {
        "generation_config": {
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    res = await provider.call("ping", model, context=context)
    assert res["content"] == "pong"
    assert len(recorded_requests) == 1


@pytest.mark.asyncio
async def test_gemini_generation_config_omitted_when_absent_or_not_dict():
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert "generationConfig" not in body
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="gemini-key", client=client)
    model = _dummy_model("google", "gemini-3.8-flash")

    # Absent
    await provider.call("ping", model, context=None)
    assert len(recorded_requests) == 1

    # Not a dict
    await provider.call("ping", model, context={"generation_config": "not-a-dict"})
    assert len(recorded_requests) == 2


@pytest.mark.asyncio
async def test_smoke_test_gemini_supplies_low_thinking_level():
    """Verify scripts/smoke_test_gemini.py supplies low thinkingLevel in context."""
    import inspect
    import scripts.smoke_test_gemini as smoke_mod

    source = inspect.getsource(smoke_mod.run_smoke_test)
    assert '"thinkingLevel": "low"' in source or "'thinkingLevel': 'low'" in source
    assert "generation_config" in source


@pytest.mark.asyncio
async def test_gemini_blocked_prompt():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="key", client=client)
    model = _dummy_model("google", "gemini-1.5-pro")

    with pytest.raises(ProviderExecutionError, match="blocked prompt with reason: SAFETY"):
        await provider.call("harmful prompt", model)


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

