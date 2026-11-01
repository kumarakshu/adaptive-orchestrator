"""Tests for Redis and in-memory conversation context persistence."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock
import pytest
import httpx

from app.api.dependencies import get_context_store, reset_dependencies
from app.core.config import Settings
from app.core.context import (
    InMemoryContextStore,
    RedisContextStore,
    create_context_store,
)
from app.main import create_app
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.executor import ModelExecutor


@pytest.mark.asyncio
async def test_in_memory_context_store_lifecycle():
    """Verify InMemoryContextStore save, get, append_turn, clear, and close."""
    store = InMemoryContextStore(default_ttl_seconds=3600)

    # 1. Non-existent context returns empty dict
    assert await store.get_context("unknown-session") == {}

    # 2. Save and retrieve context
    await store.save_context("sess-1", {"variables": {"user_id": "u123"}})
    ctx = await store.get_context("sess-1")
    assert ctx["variables"]["user_id"] == "u123"

    # 3. Append turns
    res1 = await store.append_turn("sess-1", role="user", content="Hello!")
    assert len(res1["turns"]) == 1
    assert res1["turns"][0]["role"] == "user"
    assert res1["turns"][0]["content"] == "Hello!"

    res2 = await store.append_turn(
        "sess-1",
        role="assistant",
        content="Hi there!",
        metadata={"model": "test-model"},
    )
    assert len(res2["turns"]) == 2
    assert res2["turns"][1]["role"] == "assistant"
    assert res2["turns"][1]["metadata"]["model"] == "test-model"

    # 4. Clear context
    await store.clear_context("sess-1")
    assert await store.get_context("sess-1") == {}


@pytest.mark.asyncio
async def test_in_memory_context_store_ttl_expiration():
    """Verify expired context is automatically pruned."""
    store = InMemoryContextStore(default_ttl_seconds=1)

    # Save with very short TTL
    await store.save_context("sess-exp", {"data": "ephemeral"}, ttl_seconds=1)
    assert (await store.get_context("sess-exp"))["data"] == "ephemeral"

    # Simulate expiration
    store._store["sess-exp"] = ({"data": "ephemeral"}, time.time() - 1.0)
    assert await store.get_context("sess-exp") == {}


def test_create_context_store_factory():
    """Verify create_context_store returns InMemoryContextStore when use_redis is False."""
    store = create_context_store(use_redis=False)
    assert isinstance(store, InMemoryContextStore)


@pytest.mark.asyncio
async def test_redis_context_store_fallback_on_unreachable():
    """Verify RedisContextStore falls back to in-memory gracefully when Redis fails."""
    # Use an invalid/unreachable port
    store = RedisContextStore(
        redis_url="redis://127.0.0.1:59999/0",
        default_ttl_seconds=60,
    )

    # Should not raise exception and should succeed via in-memory fallback
    await store.save_context("sess-fb", {"key": "val"})
    data = await store.get_context("sess-fb")
    assert data.get("key") == "val"

    turn_res = await store.append_turn("sess-fb", role="user", content="Ping")
    assert len(turn_res["turns"]) == 1

    await store.clear_context("sess-fb")
    assert await store.get_context("sess-fb") == {}
    await store.close()


@pytest.mark.asyncio
async def test_chat_endpoint_context_id_multi_turn_persistence():
    """Verify /api/v1/chat stores multi-turn conversation history when context_id is supplied."""
    reset_dependencies()

    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="echo-model",
            provider="test-p",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )

    def echo_provider(prompt, model, context=None):
        history_len = len(context.get("history", [])) if context else 0
        return f"Echo ({history_len} previous turns): {prompt}"

    executor = ModelExecutor()
    executor.register_provider("test-p", echo_provider)

    context_store = InMemoryContextStore()

    from app.api.dependencies import get_model_executor, get_model_registry
    from app.main import app

    async def override_registry():
        return registry

    async def override_executor():
        return executor

    async def override_context_store():
        return context_store

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor
    app.dependency_overrides[get_context_store] = override_context_store

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = "test-conv-123"

            # Turn 1: Initial prompt
            resp1 = await client.post(
                "/api/v1/chat",
                json={"prompt": "First message", "context_id": session_id},
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["success"] is True
            assert "0 previous turns" in data1["content"]

            # Verify context store now holds user + assistant turns (2 total)
            stored = await context_store.get_context(session_id)
            assert len(stored.get("turns", [])) == 2
            assert stored["turns"][0]["content"] == "First message"
            assert "Echo (0 previous turns)" in stored["turns"][1]["content"]

            # Turn 2: Subsequent prompt with same context_id
            resp2 = await client.post(
                "/api/v1/chat",
                json={"prompt": "Second message", "context_id": session_id},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["success"] is True
            # The second turn sees the 2 previous turns in context!
            assert "2 previous turns" in data2["content"]

            # Verify context store now has 4 turns (user1, asst1, user2, asst2)
            stored_after = await context_store.get_context(session_id)
            assert len(stored_after.get("turns", [])) == 4
    finally:
        app.dependency_overrides.clear()
        reset_dependencies()


@pytest.mark.asyncio
async def test_chat_endpoint_stateless_when_context_id_omitted():
    """Verify /api/v1/chat remains completely stateless when context_id is None."""
    reset_dependencies()

    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="echo-model",
            provider="test-p",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )

    call_count = [0]

    def count_provider(prompt, model, context=None):
        call_count[0] += 1
        history_len = len(context.get("history", [])) if context else 0
        return f"Turn {call_count[0]}, history: {history_len}"

    executor = ModelExecutor()
    executor.register_provider("test-p", count_provider)
    context_store = InMemoryContextStore()

    from app.api.dependencies import get_model_executor, get_model_registry
    from app.main import app

    async def override_registry():
        return registry

    async def override_executor():
        return executor

    async def override_context_store():
        return context_store

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor
    app.dependency_overrides[get_context_store] = override_context_store

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.post("/api/v1/chat", json={"prompt": "Ping 1"})
            assert resp1.status_code == 200
            assert "history: 0" in resp1.json()["content"]

            resp2 = await client.post("/api/v1/chat", json={"prompt": "Ping 2"})
            assert resp2.status_code == 200
            assert "history: 0" in resp2.json()["content"]

            # Store remains completely empty
            assert len(context_store._store) == 0
    finally:
        app.dependency_overrides.clear()
        reset_dependencies()


@pytest.mark.asyncio
async def test_in_memory_context_store_sanitizes_secrets():
    """Verify InMemoryContextStore redacts API keys and tokens before storing."""
    store = InMemoryContextStore()

    # 1. save_context with fake API keys in nested dictionary and lists
    fake_openai = "sk-fakeopenai12345678901234567890"
    fake_gemini = "AIzaSyFakeGeminiKey1234567890123456"
    fake_anthropic = "sk-ant-fakeanthropic123456789012"
    fake_bearer = "Bearer myFakeAuthToken123456789"

    raw_context = {
        "variables": {
            "api_key": fake_openai,
            "nested": {
                "gemini": fake_gemini,
                "token_list": [fake_bearer, "normal_value"],
            },
        },
        "safe_label": "production_cluster_1",
    }

    await store.save_context("sess-sec-1", raw_context)
    stored = await store.get_context("sess-sec-1")

    # Sensitive values must be scrubbed
    assert fake_openai not in json.dumps(stored)
    assert fake_gemini not in json.dumps(stored)
    assert "myFakeAuthToken" not in json.dumps(stored)
    assert "[REDACTED_API_KEY]" in stored["variables"]["api_key"]
    assert "[REDACTED_API_KEY]" in stored["variables"]["nested"]["gemini"]
    assert "Bearer [REDACTED_TOKEN]" in stored["variables"]["nested"]["token_list"][0]
    # Non-sensitive value must be unchanged
    assert stored["variables"]["nested"]["token_list"][1] == "normal_value"
    assert stored["safe_label"] == "production_cluster_1"

    # 2. append_turn with sensitive text in content and metadata
    await store.append_turn(
        "sess-sec-1",
        role="user",
        content=f"Please call endpoint using {fake_anthropic} as header",
        metadata={"auth_header": fake_bearer},
    )

    updated = await store.get_context("sess-sec-1")
    assert len(updated["turns"]) == 1
    turn = updated["turns"][0]
    assert fake_anthropic not in turn["content"]
    assert "[REDACTED_API_KEY]" in turn["content"]
    assert "Please call endpoint using" in turn["content"]
    assert fake_bearer not in json.dumps(turn["metadata"])
    assert "Bearer [REDACTED_TOKEN]" in turn["metadata"]["auth_header"]


@pytest.mark.asyncio
async def test_redis_context_store_sanitizes_before_serialization():
    """Verify RedisContextStore sanitizes data before serializing to Redis and fallback."""
    fake_openai = "sk-proj-testOpenAiSecretKey1234567890123456789"
    store = RedisContextStore(redis_url="redis://localhost:6379/0")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    store._client = mock_client

    # Test save_context sanitization
    await store.save_context(
        "sess-redis-sec",
        {"creds": {"key": fake_openai}, "description": "valid request"},
    )

    # Check mock client set call
    assert mock_client.set.called
    call_args = mock_client.set.call_args[0]
    key_arg, payload_arg = call_args[0], call_args[1]
    assert "sess-redis-sec" in key_arg
    assert fake_openai not in payload_arg
    assert "[REDACTED_API_KEY]" in payload_arg
    assert "valid request" in payload_arg

    # Test append_turn sanitization
    fake_token = "Bearer topSecretBearerToken987654321"
    mock_client.get = AsyncMock(
        return_value=json.dumps({"turns": [], "variables": {}})
    )

    res = await store.append_turn(
        "sess-redis-sec",
        role="assistant",
        content=f"Response containing {fake_token} safely",
        metadata={"trace_id": "tr-100", "auth": fake_token},
    )

    # Returned result and last mock set call payload must be redacted
    last_set_call = mock_client.set.call_args[0]
    last_payload = last_set_call[1]
    assert fake_token not in last_payload
    assert "Bearer [REDACTED_TOKEN]" in last_payload
    assert fake_token not in json.dumps(res)
    assert "Bearer [REDACTED_TOKEN]" in res["turns"][0]["content"]
    assert res["turns"][0]["metadata"]["trace_id"] == "tr-100"


@pytest.mark.asyncio
async def test_in_memory_checkpoint_lifecycle():
    """Verify save, get, and clear checkpoint on InMemoryContextStore."""
    store = InMemoryContextStore()

    # 1. Non-existent checkpoint returns None
    assert await store.get_checkpoint("task-unknown") is None

    # 2. Save checkpoint
    cp_data = {
        "task_id": "task-test-1",
        "status": "started",
        "prompt": "Hello world",
        "attempt_count": 0,
        "attempted_model_ids": [],
        "escalation_history": [],
        "checkpoint_version": 1,
        "updated_at": round(time.time(), 3),
    }
    await store.save_checkpoint("task-test-1", cp_data)

    retrieved = await store.get_checkpoint("task-test-1")
    assert retrieved is not None
    assert retrieved["task_id"] == "task-test-1"
    assert retrieved["status"] == "started"

    # 3. Clear checkpoint
    await store.clear_checkpoint("task-test-1")
    assert await store.get_checkpoint("task-test-1") is None


@pytest.mark.asyncio
async def test_in_memory_checkpoint_ttl_expiry():
    """Verify expired checkpoint is pruned upon retrieval."""
    store = InMemoryContextStore(default_ttl_seconds=1)

    await store.save_checkpoint(
        "task-exp",
        {"task_id": "task-exp", "status": "executing"},
        ttl_seconds=1,
    )
    assert await store.get_checkpoint("task-exp") is not None

    # Simulate expiration in store
    store._checkpoints["task-exp"] = ({"task_id": "task-exp"}, time.time() - 10)
    assert await store.get_checkpoint("task-exp") is None


@pytest.mark.asyncio
async def test_redis_checkpoint_store_mocked():
    """Verify RedisContextStore delegates save, get, and clear to Redis client."""
    store = RedisContextStore(redis_url="redis://localhost:6379/0")
    mock_client = AsyncMock()
    mock_client.set = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=1)
    mock_client.get = AsyncMock(
        return_value=json.dumps({"task_id": "t-redis-1", "status": "completed"})
    )
    store._client = mock_client

    # Test save_checkpoint
    await store.save_checkpoint("t-redis-1", {"task_id": "t-redis-1", "status": "executing"})
    assert mock_client.set.called
    call_key = mock_client.set.call_args[0][0]
    assert "orchestrator:checkpoint:t-redis-1" in call_key

    # Test get_checkpoint
    cp = await store.get_checkpoint("t-redis-1")
    assert cp is not None
    assert cp["task_id"] == "t-redis-1"
    assert cp["status"] == "completed"

    # Test clear_checkpoint
    await store.clear_checkpoint("t-redis-1")
    assert mock_client.delete.called


@pytest.mark.asyncio
async def test_checkpoint_sanitization_of_secrets():
    """Verify fake credentials in checkpoint payload are redacted."""
    store = InMemoryContextStore()
    fake_key = "sk-fakeKeyOpenAi1234567890123456789"
    fake_token = "Bearer fakeAuthToken123456789"

    raw_cp = {
        "task_id": "sec-task",
        "status": "executing",
        "prompt": f"Process prompt with secret: {fake_key}",
        "metadata": {"token": fake_token},
    }

    await store.save_checkpoint("sec-task", raw_cp)
    saved = await store.get_checkpoint("sec-task")

    assert fake_key not in json.dumps(saved)
    assert fake_token not in json.dumps(saved)
    assert "[REDACTED_API_KEY]" in saved["prompt"]
    assert "Bearer [REDACTED_TOKEN]" in saved["metadata"]["token"]


@pytest.mark.asyncio
async def test_in_memory_response_cache_hit_and_expiry():
    """Verify InMemoryContextStore caching, hit retrieval, and TTL expiry."""
    store = InMemoryContextStore(default_ttl_seconds=3600)
    assert store.backend_name == "in_memory"
    assert await store.get_active_backend() == "in_memory"

    cache_key = "test_key_123"
    # Miss before save
    assert await store.get_cached_response(cache_key) is None

    # Save and hit
    payload = {"content": "Cached output", "model_id": "test-model", "prompt_tokens": 10, "completion_tokens": 20}
    await store.save_cached_response(cache_key, payload, ttl_seconds=100)
    cached = await store.get_cached_response(cache_key)
    assert cached is not None
    assert cached["content"] == "Cached output"
    assert cached["model_id"] == "test-model"

    # Simulate expiration
    store._response_cache[cache_key] = (payload, time.time() - 1.0)
    assert await store.get_cached_response(cache_key) is None


@pytest.mark.asyncio
async def test_redis_response_cache_hit_and_fallback():
    """Verify RedisContextStore delegates response cache to Redis and falls back cleanly."""
    store = RedisContextStore(redis_url="redis://localhost:6379/0")
    assert store.backend_name == "redis"

    mock_client = AsyncMock()
    mock_client.set = AsyncMock(return_value=True)
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.get = AsyncMock(
        return_value=json.dumps({"content": "Redis cached content", "model_id": "gemini-3.8-flash"})
    )
    store._client = mock_client

    # Active backend probe
    assert await store.get_active_backend() == "redis"

    # Save to cache
    await store.save_cached_response("rk1", {"content": "Redis cached content", "model_id": "gemini-3.8-flash"})
    assert mock_client.set.called
    call_key = mock_client.set.call_args[0][0]
    assert "orchestrator:response_cache:rk1" in call_key

    # Get from cache
    cached = await store.get_cached_response("rk1")
    assert cached is not None
    assert cached["content"] == "Redis cached content"

    # When Redis ping fails, active backend reports in_memory fallback
    mock_client.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
    assert await store.get_active_backend() == "in_memory"


def test_compute_response_cache_key_isolation():
    """Verify compute_response_cache_key produces deterministic and isolated keys."""
    from app.core.context import compute_response_cache_key

    k1 = compute_response_cache_key("model-a", "Hello world")
    k2 = compute_response_cache_key("model-a", "Hello world")
    assert k1 == k2

    # Case insensitive model ID
    k3 = compute_response_cache_key("MODEL-A", "Hello world")
    assert k1 == k3

    # Different prompt produces different key
    k4 = compute_response_cache_key("model-a", "Different prompt")
    assert k1 != k4

    # Different context produces different key
    k5 = compute_response_cache_key("model-a", "Hello world", context={"history": [{"role": "user", "content": "prev"}]})
    assert k1 != k5

    # Different options produce different key
    k6 = compute_response_cache_key("model-a", "Hello world", options={"temperature": 0.7})
    assert k1 != k6


def test_cross_model_context_recovery_turn_extraction():
    """Verify _extract_messages extracts from history/turns and normalizes roles across models."""
    from app.providers.base import BaseHTTPProvider

    class DummyProvider(BaseHTTPProvider):
        async def call(self, prompt, model, context=None):
            return "ok"

    provider = DummyProvider(api_key="test-key")

    # Context with "history" and diverse roles
    context = {
        "system_prompt": "You are a helpful assistant.",
        "history": [
            {"role": "user", "content": "What is Python?"},
            {"role": "model", "content": "Python is a programming language."},
            {"role": "human", "content": "Give an example."},
            {"role": "bot", "content": "print('hello')"},
        ],
    }

    sys_prompt, messages = provider._extract_messages("Explain lists", context)
    assert sys_prompt == "You are a helpful assistant."
    assert len(messages) == 5
    assert messages[0] == {"role": "user", "content": "What is Python?"}
    assert messages[1] == {"role": "assistant", "content": "Python is a programming language."}
    assert messages[2] == {"role": "user", "content": "Give an example."}
    assert messages[3] == {"role": "assistant", "content": "print('hello')"}
    assert messages[4] == {"role": "user", "content": "Explain lists"}

    # Context with "turns"
    turns_context = {
        "turns": [
            {"role": "user", "content": "Turn 1"},
            {"role": "ai", "content": "Turn 2"},
        ]
    }
    _, turn_msgs = provider._extract_messages("Turn 3", turns_context)
    assert len(turn_msgs) == 3
    assert turn_msgs[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_engine_response_caching_end_to_end():
    """Verify OrchestrationEngine executes model on first request and serves from cache on second."""
    from app.orchestrator.engine import OrchestrationEngine
    from app.orchestrator.analyzer import PromptAnalyzer
    from app.orchestrator.router import IntelligentRouter
    from app.orchestrator.validator import ResponseValidator

    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="cacheable-model",
            provider="test-cache-p",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )

    execution_count = 0

    def counting_handler(prompt, model, context=None):
        nonlocal execution_count
        execution_count += 1
        return f"Response #{execution_count} for: {prompt}"

    executor = ModelExecutor()
    executor.register_provider("test-cache-p", counting_handler)

    store = InMemoryContextStore()
    engine = OrchestrationEngine(
        registry=registry,
        analyzer=PromptAnalyzer(),
        router=IntelligentRouter(registry=registry),
        executor=executor,
        validator=ResponseValidator(),
        context_store=store,
    )

    prompt = "Compute 42 * 42"

    # First request: Cache MISS -> executes model
    res1 = await engine.orchestrate(prompt=prompt)
    assert res1.success is True
    assert "Response #1" in res1.content
    assert res1.metadata.get("cached") is False
    assert res1.metadata.get("cache_status") == "MISS"
    assert execution_count == 1

    # Second request with identical prompt: Cache HIT -> does NOT call provider
    res2 = await engine.orchestrate(prompt=prompt)
    assert res2.success is True
    assert "Response #1" in res2.content
    assert res2.metadata.get("cached") is True
    assert res2.metadata.get("cache_status") == "HIT"
    assert res2.metadata.get("cache_saved_tokens") > 0
    assert execution_count == 1  # Executor was NOT called again!


@pytest.mark.asyncio
async def test_chat_endpoint_demonstration_fields():
    """Verify /api/v1/chat returns cached, cache_status, context_backend, context_turns_used."""
    reset_dependencies()

    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="demo-model",
            provider="demo-provider",
            tier=ModelTier.LIGHTWEIGHT,
        )
    )

    call_count = 0

    def demo_handler(prompt, model, context=None):
        nonlocal call_count
        call_count += 1
        return f"Echo {call_count}: {prompt}"

    executor = ModelExecutor()
    executor.register_provider("demo-provider", demo_handler)

    store = InMemoryContextStore()

    from app.api.dependencies import get_model_executor, get_model_registry
    from app.main import app

    async def override_registry():
        return registry

    async def override_executor():
        return executor

    async def override_context_store():
        return store

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor
    app.dependency_overrides[get_context_store] = override_context_store

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = "demo-session-456"

            # Turn 1: Cache miss
            r1 = await client.post("/api/v1/chat", json={"prompt": "Turn 1 prompt", "context_id": session_id})
            assert r1.status_code == 200
            data1 = r1.json()
            assert data1["cached"] is False
            assert data1["cache_status"] == "MISS"
            assert data1["context_backend"] == "in_memory"
            assert data1["context_turns_used"] == 0

            # Turn 2: Stored context has 2 turns (1 user, 1 assistant)
            r2 = await client.post("/api/v1/chat", json={"prompt": "Turn 2 prompt", "context_id": session_id})
            assert r2.status_code == 200
            data2 = r2.json()
            assert data2["context_turns_used"] == 2
            assert data2["context_backend"] == "in_memory"

            # Query identical prompt without session: verify cache hit
            r3 = await client.post("/api/v1/chat", json={"prompt": "Turn 1 prompt"})
            assert r3.status_code == 200
            data3 = r3.json()
            assert data3["cached"] is True
            assert data3["cache_status"] == "HIT"
            assert data3["cache_saved_tokens"] is not None
            assert data3["cache_saved_tokens"] > 0
    finally:
        app.dependency_overrides.clear()
        reset_dependencies()
