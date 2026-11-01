"""Base HTTP provider adapter abstractions for Adaptive LLM Orchestrator.

Provides shared HTTP client management, message transformation, error handling,
and credential protection across LLM provider implementations.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from app.models.registry import ModelDescriptor
from app.orchestrator.executor import (
    ExecutionResult,
    ModelProvider,
    ProviderExecutionError,
)


class BaseHTTPProvider(ModelProvider):
    """Abstract base class for HTTP-based model provider integrations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize provider with optional key, injected client, and timeout."""
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._client = client
        self._timeout_seconds = timeout_seconds

    @property
    def has_api_key(self) -> bool:
        """Check whether an API key has been configured."""
        return self._api_key is not None

    @property
    def timeout_seconds(self) -> float:
        """Configured request timeout in seconds."""
        return self._timeout_seconds

    def _ensure_api_key(self, provider_name: str) -> str:
        """Verify API key is configured without exposing any secret values."""
        if not self._api_key:
            raise ProviderExecutionError(
                f"{provider_name} API key is not configured. Please set the environment variable."
            )
        return self._api_key

    async def _send_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        provider_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send an async POST request with safe error wrapping and credential masking."""
        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(self._timeout_seconds)
        )
        timeout = httpx.Timeout(effective_timeout)

        try:
            if self._client is not None:
                response = await self._client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            cause_type = type(exc).__name__
            raise ProviderExecutionError(
                f"{provider_name} request timed out after {effective_timeout:.1f}s ({cause_type}).",
                cause_type=cause_type,
            ) from exc
        except httpx.RequestError as exc:
            cause_type = type(exc).__name__
            raise ProviderExecutionError(
                f"Network communication failure connecting to {provider_name} ({cause_type}).",
                cause_type=cause_type,
            ) from exc

        if response.status_code >= 400:
            status_code = response.status_code
            safe_detail = "Unknown upstream error"
            try:
                err_data = response.json()
                if isinstance(err_data, dict):
                    # Common error fields across OpenAI, Anthropic, Gemini
                    safe_detail = (
                        err_data.get("error", {}).get("message")
                        or err_data.get("message")
                        or str(err_data.get("error"))
                        or response.text[:200]
                    )
            except Exception:
                safe_detail = response.text[:200]

            retry_after: Optional[float] = None
            raw_retry_after = response.headers.get("retry-after")
            if raw_retry_after:
                try:
                    retry_after = float(raw_retry_after)
                except ValueError:
                    pass

            raise ProviderExecutionError(
                f"{provider_name} API returned error HTTP {status_code}: {safe_detail}",
                status_code=status_code,
                retry_after=retry_after,
            )

        try:
            return response.json()
        except Exception as exc:
            raise ProviderExecutionError(
                f"Malformed JSON response received from {provider_name} API."
            ) from exc

    def _extract_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Extract system instruction and normalized message sequence from prompt and context.

        Returns:
            Tuple of (system_prompt, messages_list)
        """
        messages: List[Dict[str, str]] = []
        system_prompt: Optional[str] = None

        if context and isinstance(context, dict):
            # Check for explicit system instruction
            if "system_prompt" in context and isinstance(context["system_prompt"], str):
                system_prompt = context["system_prompt"]

            raw_messages = (
                context.get("messages")
                or context.get("history")
                or context.get("turns")
            )
            if isinstance(raw_messages, list):
                for m in raw_messages:
                    if isinstance(m, dict) and "content" in m:
                        raw_role = str(m.get("role", "user")).lower()
                        content = str(m["content"])
                        if raw_role == "system" and not system_prompt:
                            system_prompt = content
                        elif raw_role in ("user", "human"):
                            messages.append({"role": "user", "content": content})
                        elif raw_role in ("assistant", "model", "bot", "ai"):
                            messages.append({"role": "assistant", "content": content})

        # Append current user prompt if not already present as the last message
        if not messages or messages[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        return system_prompt, messages

    @abstractmethod
    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], ExecutionResult]:
        """Execute prompt against external provider and return text or metadata payload."""
        pass

