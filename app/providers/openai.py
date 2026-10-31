"""OpenAI provider adapter for Adaptive LLM Orchestrator."""

from typing import Any, Dict, List, Optional, Union
import httpx

from app.models.registry import ModelDescriptor
from app.orchestrator.executor import ExecutionResult, ProviderExecutionError
from app.providers.base import BaseHTTPProvider


class OpenAIProvider(BaseHTTPProvider):
    """Provider adapter for OpenAI chat completion models."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            client=client,
            timeout_seconds=timeout_seconds,
        )
        self._base_url = base_url or self.DEFAULT_BASE_URL

    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], ExecutionResult]:
        """Execute chat completion request with OpenAI."""
        api_key = self._ensure_api_key("OpenAI")

        system_prompt, messages = self._extract_messages(prompt, context)

        request_messages: List[Dict[str, str]] = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = "user" if msg["role"] not in ("user", "assistant", "system") else msg["role"]
            request_messages.append({"role": role, "content": msg["content"]})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model.model_id,
            "messages": request_messages,
        }

        data = await self._send_request(
            url=self._base_url,
            headers=headers,
            payload=payload,
            provider_name="OpenAI",
        )

        try:
            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise ProviderExecutionError("OpenAI response did not contain any choices.")

            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                raise ProviderExecutionError("OpenAI response choice did not contain text content.")

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

            return {
                "content": str(content),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "raw_response": data,
            }
        except KeyError as exc:
            raise ProviderExecutionError(f"OpenAI response missing expected key: {exc}") from exc

