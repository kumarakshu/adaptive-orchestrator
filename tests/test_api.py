"""API integration tests for FastAPI endpoints."""

import pytest
import httpx

from app.api.dependencies import (
    get_context_store,
    get_model_executor,
    get_model_registry,
    reset_dependencies,
)
from app.core.context import InMemoryContextStore
from app.main import app
from app.models.registry import ModelDescriptor, ModelRegistry, ModelTier
from app.orchestrator.executor import ModelExecutor


@pytest.fixture
def fake_registry() -> ModelRegistry:
    """Fixture providing an explicit test model registry."""
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            model_id="api-test-fast",
            provider="test-provider",
            tier=ModelTier.LIGHTWEIGHT,
            context_window=4096,
            supported_capabilities=["chat"],
            enabled=True,
        )
    )
    registry.register(
        ModelDescriptor(
            model_id="api-test-advanced",
            provider="test-provider",
            tier=ModelTier.ADVANCED,
            context_window=16384,
            supported_capabilities=["chat", "code", "reasoning"],
            enabled=True,
        )
    )
    return registry


@pytest.fixture
def fake_executor() -> ModelExecutor:
    """Fixture providing a mock executor with a fake provider."""
    def fake_handler(prompt, model, context=None):
        return f"Echo from {model.model_id}: {prompt}"

    executor = ModelExecutor()
    executor.register_provider("test-provider", fake_handler)
    return executor


# 1. Health check endpoints
@pytest.mark.asyncio
async def test_root_health_endpoint():
    """Verify GET /health returns 200 OK and valid status."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "app" in data
        assert "version" in data


@pytest.mark.asyncio
async def test_api_v1_health_endpoint():
    """Verify GET /api/v1/health returns 200 OK."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# 2. Model listing endpoint with override
@pytest.mark.asyncio
async def test_list_models_endpoint(fake_registry: ModelRegistry):
    """Verify GET /api/v1/models returns registered models without secrets."""
    async def override_registry():
        return fake_registry

    app.dependency_overrides[get_model_registry] = override_registry
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2

            model_ids = [m["model_id"] for m in data]
            assert "api-test-fast" in model_ids
            assert "api-test-advanced" in model_ids

            # Ensure no credentials or keys exist in response
            for item in data:
                assert "api_key" not in item
                assert "secret" not in item
                assert "context_window" in item
                assert "enabled" in item
    finally:
        app.dependency_overrides.clear()


# 3. Successful chat execution with override
@pytest.mark.asyncio
async def test_chat_successful_execution(fake_registry: ModelRegistry, fake_executor: ModelExecutor):
    """Verify POST /api/v1/chat executes prompt and returns structured response."""
    async def override_registry():
        return fake_registry

    async def override_executor():
        return fake_executor

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "prompt": "What is the capital of Italy?",
            }
            response = await client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "api-test-fast" in data["selected_model"]
            assert "Echo from api-test-fast: What is the capital of Italy?" in data["content"]
            assert data["selected_tier"] == "lightweight"
            assert data["routing_reason"] is not None
            assert data["latency_ms"] >= 0.0
            assert data["error"] is None
    finally:
        app.dependency_overrides.clear()


# 4. Empty or whitespace prompt validation
@pytest.mark.asyncio
async def test_chat_empty_prompt_validation():
    """Verify empty or whitespace prompt triggers 422 Unprocessable Entity."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Empty string
        resp1 = await client.post("/api/v1/chat", json={"prompt": ""})
        assert resp1.status_code == 422

        # Whitespace-only string
        resp2 = await client.post("/api/v1/chat", json={"prompt": "    \t\n  "})
        assert resp2.status_code == 422

        # Missing prompt field
        resp3 = await client.post("/api/v1/chat", json={})
        assert resp3.status_code == 422


# 5. Missing provider execution failure behavior
@pytest.mark.asyncio
async def test_chat_missing_provider_returns_502(fake_registry: ModelRegistry):
    """Verify that unconfigured provider returns 502 with structured error message."""
    empty_executor = ModelExecutor()  # No providers registered

    async def override_registry():
        return fake_registry

    async def override_executor():
        return empty_executor

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat", json={"prompt": "Testing missing provider"})
            assert response.status_code == 502

            data = response.json()
            assert data["success"] is False
            assert "No provider configured for provider 'test-provider'" in data["error"]
            assert data["selected_model"] == "api-test-fast"
    finally:
        app.dependency_overrides.clear()


# 6. Provider exception behavior
@pytest.mark.asyncio
async def test_chat_provider_exception_returns_502(fake_registry: ModelRegistry):
    """Verify that provider exception returns 502 without unhandled traceback."""
    def crashing_handler(prompt, model):
        raise ConnectionError("Remote model provider refused connection")

    failing_executor = ModelExecutor()
    failing_executor.register_provider("test-provider", crashing_handler)

    async def override_registry():
        return fake_registry

    async def override_executor():
        return failing_executor

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat", json={"prompt": "Testing crashing provider"})
            assert response.status_code == 502

            data = response.json()
            assert data["success"] is False
            assert "Remote model provider refused connection" in data["error"]
    finally:
        app.dependency_overrides.clear()


# 7. Backward compatibility orchestrate alias
@pytest.mark.asyncio
async def test_orchestrate_alias_endpoint(fake_registry: ModelRegistry, fake_executor: ModelExecutor):
    """Verify legacy /api/v1/orchestrate functions identically."""
    async def override_registry():
        return fake_registry

    async def override_executor():
        return fake_executor

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/orchestrate", json={"prompt": "Legacy ping"})
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Echo from api-test-fast: Legacy ping" in data["content"]
    finally:
        app.dependency_overrides.clear()


# 8. Unconfigured defaults and dependency integration tests
@pytest.mark.asyncio
async def test_default_models_endpoint_without_overrides():
    """Verify GET /api/v1/models returns default registry models with no overrides."""
    reset_dependencies()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 6
            model_ids = [m["model_id"] for m in data]
            assert "gpt-4o-mini" in model_ids
            assert "claude-3-5-sonnet" in model_ids
            assert "gemini-3.8-flash" in model_ids
    finally:
        reset_dependencies()


@pytest.mark.asyncio
async def test_chat_without_overrides_unconfigured_keys_returns_502():
    """Verify POST /api/v1/chat returns 502 with structured error when no keys are configured."""
    reset_dependencies()
    import app.api.dependencies as deps_mod
    from app.core.config import Settings
    unconfigured_settings = Settings(
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        GEMINI_API_KEY=None,
    )
    orig_get_settings = deps_mod.get_settings
    deps_mod.get_settings = lambda: unconfigured_settings
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat", json={"prompt": "Hello unconfigured orchestrator"})
            assert response.status_code == 502
            data = response.json()
            assert data["success"] is False
            assert "No provider configured" in data["error"]
            assert data["selected_model"] is not None
    finally:
        deps_mod.get_settings = orig_get_settings
        reset_dependencies()


@pytest.mark.asyncio
async def test_get_model_executor_registers_configured_providers():
    """Verify get_model_executor respects configured settings via factory."""
    import app.api.dependencies as deps_mod
    from app.core.config import Settings

    reset_dependencies()
    orig_get_settings = deps_mod.get_settings
    try:
        # Part 1: Verify unconfigured environment registers zero providers
        unconfigured_settings = Settings(
            OPENAI_API_KEY=None,
            ANTHROPIC_API_KEY=None,
            GEMINI_API_KEY=None,
        )
        deps_mod.get_settings = lambda: unconfigured_settings
        executor = await deps_mod.get_model_executor()
        assert isinstance(executor, ModelExecutor)
        assert executor.list_providers() == []

        # Part 2: Verify configured test dummy keys register expected providers without network calls
        reset_dependencies()
        mock_settings = Settings(
            OPENAI_API_KEY="sk-mock-key",
            ANTHROPIC_API_KEY=None,
            GEMINI_API_KEY=None,
        )
        deps_mod.get_settings = lambda: mock_settings
        executor_configured = await deps_mod.get_model_executor()
        assert "openai" in executor_configured.list_providers()
        assert "anthropic" not in executor_configured.list_providers()
        assert "google" not in executor_configured.list_providers()
    finally:
        deps_mod.get_settings = orig_get_settings
        reset_dependencies()


# 9. Configuration and CORS Security Tests
def test_settings_debug_defaults_to_false():
    """Verify DEBUG defaults to False in settings and can be explicitly overridden."""
    from app.core.config import Settings

    default_settings = Settings(_env_file=None)
    assert default_settings.debug is False

    dev_settings = Settings(_env_file=None, DEBUG=True)
    assert dev_settings.debug is True


def test_settings_cors_origins_parsing():
    """Verify CORS origins parsing for strings, JSON arrays, and lists."""
    from app.core.config import Settings

    # Default is empty list (safe production default)
    empty_settings = Settings(_env_file=None)
    assert empty_settings.cors_allowed_origins == []
    assert empty_settings.cors_allow_credentials is False

    # Comma-separated string parsing
    comma_settings = Settings(
        _env_file=None,
        CORS_ALLOWED_ORIGINS="http://localhost:3000, http://localhost:5173",
    )
    assert comma_settings.cors_allowed_origins == [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # JSON array parsing via alternate alias ALLOWED_ORIGINS
    json_settings = Settings(
        _env_file=None,
        ALLOWED_ORIGINS='["http://localhost:3000"]',
    )
    assert json_settings.cors_allowed_origins == ["http://localhost:3000"]

    # Empty string parses to empty list
    blank_settings = Settings(_env_file=None, CORS_ALLOWED_ORIGINS="")
    assert blank_settings.cors_allowed_origins == []


@pytest.mark.asyncio
async def test_cors_wildcard_disallows_credentials():
    """Verify wildcard CORS ("*") forces allow_credentials=False for security."""
    from app.core.config import Settings
    from app.main import create_app

    # Even if credentials are requested, wildcard must disallow them
    wildcard_settings = Settings(
        _env_file=None,
        CORS_ALLOWED_ORIGINS="*",
        CORS_ALLOW_CREDENTIALS=True,
    )
    test_app = create_app(custom_settings=wildcard_settings)

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
        # Credentials must NOT be allowed when origin is wildcard
        assert response.headers.get("access-control-allow-credentials") is None


@pytest.mark.asyncio
async def test_cors_explicit_origin_allows_credentials():
    """Verify specific origin allows credentials when explicitly configured."""
    from app.core.config import Settings
    from app.main import create_app

    frontend_origin = "http://localhost:3000"
    explicit_settings = Settings(
        _env_file=None,
        CORS_ALLOWED_ORIGINS=frontend_origin,
        CORS_ALLOW_CREDENTIALS=True,
    )
    test_app = create_app(custom_settings=explicit_settings)

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Request from authorized origin
        response = await client.options(
            "/health",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == frontend_origin
        assert response.headers.get("access-control-allow-credentials") == "true"

        # Request from unauthorized origin receives no CORS headers
        unauth_response = await client.options(
            "/health",
            headers={
                "Origin": "http://malicious.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert unauth_response.headers.get("access-control-allow-origin") is None


@pytest.mark.asyncio
async def test_cors_default_empty_origins_blocks_cross_origin():
    """Verify default empty origin list prevents CORS headers while allowing local/direct API use."""
    from app.core.config import Settings
    from app.main import create_app

    safe_default_settings = Settings(_env_file=None)
    test_app = create_app(custom_settings=safe_default_settings)

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Cross-origin request receives no CORS approval header
        cors_req = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert cors_req.headers.get("access-control-allow-origin") is None

        # Direct API calls (same-origin / curl / Postman) succeed normally
        direct_req = await client.get("/health")
        assert direct_req.status_code == 200
        assert direct_req.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_and_task_status_endpoints(fake_registry: ModelRegistry, fake_executor: ModelExecutor):
    """Verify chat execution with task_id and querying/clearing task status."""
    store = InMemoryContextStore()

    async def override_registry():
        return fake_registry

    async def override_executor():
        return fake_executor

    async def override_store():
        return store

    app.dependency_overrides[get_model_registry] = override_registry
    app.dependency_overrides[get_model_executor] = override_executor
    app.dependency_overrides[get_context_store] = override_store

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            task_id = "test-task-api-1"

            # 1. Post chat with task_id
            resp = await client.post(
                "/api/v1/chat",
                json={"prompt": "Hello task", "task_id": task_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == task_id
            assert data["checkpoint_status"] == "completed"

            # 2. Query task status
            status_resp = await client.get(f"/api/v1/tasks/{task_id}")
            assert status_resp.status_code == 200
            task_data = status_resp.json()
            assert task_data["task_id"] == task_id
            assert task_data["status"] == "completed"
            assert task_data["attempt_count"] == 0

            # 3. Query non-existent task
            notFound = await client.get("/api/v1/tasks/task-missing-99")
            assert notFound.status_code == 404

            # 4. Invalid task_id returns 400
            badId = await client.get("/api/v1/tasks/bad*id!")
            assert badId.status_code == 400

            # 5. Clear task checkpoint
            del_resp = await client.delete(f"/api/v1/tasks/{task_id}")
            assert del_resp.status_code == 200

            # 6. Query after delete returns 404
            status_after_del = await client.get(f"/api/v1/tasks/{task_id}")
            assert status_after_del.status_code == 404
    finally:
        app.dependency_overrides.clear()
        reset_dependencies()

