"""Model execution abstraction module.

Decouples orchestration flow from specific model provider implementations
(e.g., OpenAI, Anthropic, Gemini, or mock/local providers).
"""

from abc import ABC, abstractmethod
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.models.registry import ModelDescriptor
from app.orchestrator.router import RoutingDecision


class ExecutionError(Exception):
    """Base exception for model execution failures."""
    pass


class ProviderNotFoundError(ExecutionError):
    """Raised when no provider is configured or registered for a target model."""
    pass


class ProviderExecutionError(ExecutionError):
    """Raised when an underlying provider fails during prompt execution."""
    pass


class ExecutionResult(BaseModel):
    """Normalized response returned from executing a model call."""

    content: str = Field(default="", description="Generated response content from the model")
    model_id: str = Field(..., description="ID of the model that executed the call")
    success: bool = Field(default=True, description="Whether the model execution succeeded")
    latency_ms: float = Field(default=0.0, description="Execution round-trip latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Detailed error description if execution failed")
    prompt_tokens: int = Field(default=0, description="Estimated or actual input token count")
    completion_tokens: int = Field(default=0, description="Output token count generated")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider payload if available")

    @property
    def response_text(self) -> str:
        """Alias for content providing consistent response text access."""
        return self.content

    @property
    def selected_model_id(self) -> str:
        """Alias for model_id."""
        return self.model_id


class ModelProvider(ABC):
    """Abstract interface for external model provider clients.

    Independent provider modules developed by teammates can inherit from this
    class or provide any callable matching the `call` signature.
    """

    @abstractmethod
    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], ExecutionResult]:
        """Execute prompt against external provider and return text or metadata payload."""
        pass


# Type alias for acceptable provider handlers: ModelProvider instance or callable
ProviderHandler = Union[
    ModelProvider,
    Callable[[str, ModelDescriptor, Optional[Dict[str, Any]]], Any],
    Callable[[str, ModelDescriptor], Any],
]


class BaseExecutor(ABC):
    """Abstract base executor interface for executing prompt calls."""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        target: Union[RoutingDecision, ModelDescriptor],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a prompt against the specified model descriptor or routing decision."""
        pass


class ModelExecutor(BaseExecutor):
    """Concrete executor managing provider dispatches and response normalization.

    Routes calls to registered provider handlers without hardcoding API keys or
    provider network calls.
    """

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderHandler]] = None,
        default_provider: Optional[ProviderHandler] = None,
        raise_on_error: bool = False,
    ) -> None:
        """Initialize executor with optional provider mapping."""
        self._providers: Dict[str, ProviderHandler] = dict(providers or {})
        self._default_provider = default_provider
        self.raise_on_error = raise_on_error

    def register_provider(self, provider_name: str, handler: ProviderHandler) -> None:
        """Register a provider handler by provider name (e.g., 'openai', 'anthropic', 'mock')."""
        self._providers[provider_name.lower()] = handler

    def unregister_provider(self, provider_name: str) -> Optional[ProviderHandler]:
        """Unregister a provider handler."""
        return self._providers.pop(provider_name.lower(), None)

    def has_provider(self, provider_name: str) -> bool:
        """Check whether a provider is registered."""
        return provider_name.lower() in self._providers or self._default_provider is not None

    def get_provider(self, provider_name: str) -> Optional[ProviderHandler]:
        """Retrieve a registered provider handler or the default provider."""
        return self._providers.get(provider_name.lower(), self._default_provider)

    def list_providers(self) -> List[str]:
        """List all explicitly registered provider names."""
        return list(self._providers.keys())

    def _resolve_model(self, target: Union[RoutingDecision, ModelDescriptor]) -> ModelDescriptor:
        """Extract ModelDescriptor from RoutingDecision or direct descriptor."""
        if isinstance(target, RoutingDecision):
            return target.model
        if isinstance(target, ModelDescriptor):
            return target
        raise ValueError("Target must be an instance of RoutingDecision or ModelDescriptor.")

    async def _invoke_handler(
        self,
        handler: ProviderHandler,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Invoke a provider handler, supporting both async and sync callables/methods."""
        if isinstance(handler, ModelProvider):
            return await handler.call(prompt=prompt, model=model, context=context)

        # Inspect callable signature to see if context argument is accepted
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())

        if len(params) >= 3:
            raw = handler(prompt, model, context)
        elif len(params) == 2:
            raw = handler(prompt, model)
        else:
            raw = handler(prompt)

        # Await if returned object is a coroutine
        if inspect.iscoroutine(raw):
            return await raw
        return raw

    def _normalize_result(
        self,
        raw_output: Any,
        model_id: str,
        latency_ms: float,
    ) -> ExecutionResult:
        """Normalize raw provider output into a standard ExecutionResult."""
        if isinstance(raw_output, ExecutionResult):
            if raw_output.latency_ms <= 0.0:
                raw_output.latency_ms = latency_ms
            return raw_output

        if isinstance(raw_output, str):
            return ExecutionResult(
                content=raw_output,
                model_id=model_id,
                success=True,
                latency_ms=latency_ms,
            )

        if isinstance(raw_output, dict):
            content = str(raw_output.get("content", raw_output.get("text", "")))
            prompt_tokens = int(raw_output.get("prompt_tokens", 0))
            completion_tokens = int(raw_output.get("completion_tokens", 0))
            return ExecutionResult(
                content=content,
                model_id=model_id,
                success=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw_response=raw_output,
            )

        # Fallback string representation
        return ExecutionResult(
            content=str(raw_output) if raw_output is not None else "",
            model_id=model_id,
            success=True,
            latency_ms=latency_ms,
        )

    async def execute(
        self,
        prompt: str,
        target: Union[RoutingDecision, ModelDescriptor],
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a prompt against the target model using the registered provider.

        Args:
            prompt: User input prompt to execute.
            target: RoutingDecision or ModelDescriptor specifying the target model.
            context: Optional execution context dictionary.

        Returns:
            ExecutionResult with response text, model ID, timing, and success status.

        Raises:
            ProviderNotFoundError: If no provider is registered and raise_on_error is True.
            ProviderExecutionError: If provider invocation fails and raise_on_error is True.
        """
        model = self._resolve_model(target)
        provider_name = model.provider.lower()
        handler = self.get_provider(provider_name)

        if handler is None:
            err_msg = (
                f"No provider configured for provider '{model.provider}' (model: '{model.model_id}'). "
                f"Available providers: {self.list_providers() or 'none registered'}."
            )
            if self.raise_on_error:
                raise ProviderNotFoundError(err_msg)
            return ExecutionResult(
                content="",
                model_id=model.model_id,
                success=False,
                error=err_msg,
                latency_ms=0.0,
            )

        start_time = time.perf_counter()
        try:
            raw_output = await self._invoke_handler(handler, prompt, model, context)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return self._normalize_result(raw_output, model.model_id, latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Provider '{model.provider}' failed during execution: {exc}"
            if self.raise_on_error:
                raise ProviderExecutionError(err_msg) from exc
            return ExecutionResult(
                content="",
                model_id=model.model_id,
                success=False,
                error=err_msg,
                latency_ms=latency_ms,
            )
