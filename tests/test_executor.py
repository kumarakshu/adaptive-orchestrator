"""Unit tests for the model executor and provider abstraction."""

import pytest
from app.models.registry import ModelDescriptor, ModelTier
from app.orchestrator.executor import (
    ExecutionResult,
    ModelExecutor,
    ModelProvider,
    ProviderExecutionError,
    ProviderNotFoundError,
)
from app.orchestrator.router import RoutingDecision


@pytest.fixture
def sample_model() -> ModelDescriptor:
    """Fixture returning a mock model descriptor."""
    return ModelDescriptor(
        model_id="mock-fast-model",
        provider="mock-provider",
        tier=ModelTier.LIGHTWEIGHT,
        context_window=4096,
        supported_capabilities=["chat"],
    )


# 1. Successful execution using sync callable
@pytest.mark.asyncio
async def test_executor_sync_callable(sample_model: ModelDescriptor):
    """Verify execution with a synchronous callable provider."""
    def fake_provider(prompt, model, context=None):
        return f"Echo: {prompt} via {model.model_id}"

    executor = ModelExecutor()
    executor.register_provider("mock-provider", fake_provider)

    result = await executor.execute(prompt="Hello world", target=sample_model)

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.content == "Echo: Hello world via mock-fast-model"
    assert result.response_text == result.content
    assert result.model_id == "mock-fast-model"
    assert result.selected_model_id == "mock-fast-model"
    assert result.latency_ms >= 0.0
    assert result.error is None


# 2. Successful execution using async callable
@pytest.mark.asyncio
async def test_executor_async_callable(sample_model: ModelDescriptor):
    """Verify execution with an asynchronous callable provider."""
    async def async_provider(prompt, model, context=None):
        return {
            "content": "Async generated content",
            "prompt_tokens": 12,
            "completion_tokens": 25,
        }

    executor = ModelExecutor()
    executor.register_provider("mock-provider", async_provider)

    result = await executor.execute(prompt="Test prompt", target=sample_model)

    assert result.success is True
    assert result.content == "Async generated content"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 25
    assert result.raw_response is not None


# 3. Successful execution using ModelProvider class instance
@pytest.mark.asyncio
async def test_executor_model_provider_class(sample_model: ModelDescriptor):
    """Verify execution with a custom ModelProvider class implementation."""
    class CustomProvider(ModelProvider):
        async def call(self, prompt, model, context=None):
            return f"Processed: {prompt}"

    executor = ModelExecutor()
    executor.register_provider("mock-provider", CustomProvider())

    result = await executor.execute(prompt="Sample query", target=sample_model)

    assert result.success is True
    assert result.content == "Processed: Sample query"


# 4. Target can be RoutingDecision or ModelDescriptor
@pytest.mark.asyncio
async def test_executor_accepts_routing_decision(sample_model: ModelDescriptor):
    """Verify executor seamlessly extracts ModelDescriptor from RoutingDecision."""
    executor = ModelExecutor(default_provider=lambda p, m: "Default response")

    decision = RoutingDecision(
        model=sample_model,
        selected_tier=ModelTier.LIGHTWEIGHT,
        preferred_tier=ModelTier.LIGHTWEIGHT,
        routing_reason="Test reason",
        original_complexity="simple",
    )

    result = await executor.execute(prompt="Ping", target=decision)

    assert result.success is True
    assert result.model_id == "mock-fast-model"
    assert result.content == "Default response"


# 5. Missing provider returns actionable structured error
@pytest.mark.asyncio
async def test_executor_missing_provider_returns_failure(sample_model: ModelDescriptor):
    """Verify missing provider returns structured failure without crashing by default."""
    executor = ModelExecutor()  # No providers registered

    result = await executor.execute(prompt="Hello", target=sample_model)

    assert result.success is False
    assert result.content == ""
    assert result.model_id == "mock-fast-model"
    assert "No provider configured for provider 'mock-provider'" in result.error
    assert "Available providers" in result.error


# 6. Missing provider raises exception when raise_on_error=True
@pytest.mark.asyncio
async def test_executor_missing_provider_raises_exception(sample_model: ModelDescriptor):
    """Verify ProviderNotFoundError is raised when raise_on_error is enabled."""
    executor = ModelExecutor(raise_on_error=True)

    with pytest.raises(ProviderNotFoundError, match="No provider configured"):
        await executor.execute(prompt="Hello", target=sample_model)


# 7. Provider exception is captured as structured failure
@pytest.mark.asyncio
async def test_executor_provider_exception_captured(sample_model: ModelDescriptor):
    """Verify provider exceptions are caught and surfaced as structured errors."""
    def broken_provider(prompt, model):
        raise ConnectionResetError("Remote server closed connection unexpectedly")

    executor = ModelExecutor()
    executor.register_provider("mock-provider", broken_provider)

    result = await executor.execute(prompt="Hello", target=sample_model)

    assert result.success is False
    assert result.content == ""
    assert "Remote server closed connection unexpectedly" in result.error
    assert result.latency_ms >= 0.0


# 8. Provider exception raises ProviderExecutionError when raise_on_error=True
@pytest.mark.asyncio
async def test_executor_provider_exception_raises(sample_model: ModelDescriptor):
    """Verify ProviderExecutionError is raised when raise_on_error is enabled."""
    def broken_provider(prompt, model):
        raise TimeoutError("Request timed out after 30s")

    executor = ModelExecutor(raise_on_error=True)
    executor.register_provider("mock-provider", broken_provider)

    with pytest.raises(ProviderExecutionError, match="Request timed out"):
        await executor.execute(prompt="Hello", target=sample_model)


# 9. Provider registration and unregistration
def test_executor_provider_management():
    """Verify registering, listing, and unregistering providers."""
    executor = ModelExecutor()
    assert not executor.has_provider("test")

    executor.register_provider("test", lambda p: "ok")
    assert executor.has_provider("test")
    assert "test" in executor.list_providers()

    removed = executor.unregister_provider("test")
    assert removed is not None
    assert not executor.has_provider("test")

