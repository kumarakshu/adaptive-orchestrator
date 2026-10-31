"""Anthropic provider adapter for Adaptive LLM Orchestrator."""

from typing import Any, Dict, List, Optional, Union
import httpx

from app.models.registry import ModelDescriptor
from app.orchestrator.executor import ExecutionResult, ProviderExecutionError
from app.providers.base import BaseHTTPProvider


class AnthropicProvider(BaseHTTPProvider):
    """Provider adapter for Anthropic Claude message models."""

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

    MODEL_MAPPINGS: Dict[str, str] = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-opus": "claude-3-opus-20240229",
    }

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
        """Execute messages request with Anthropic."""
        api_key = self._ensure_api_key("Anthropic")

        # Map shorthand model IDs to specific API versions if registered
        resolved_model_id = self.MODEL_MAPPINGS.get(model.model_id, model.model_id)

        system_prompt, extracted_messages = self._extract_messages(prompt, context)

        # Anthropic messages array: strictly 'user' and 'assistant'
        messages: List[Dict[str, str]] = []
        for msg in extracted_messages:
            role = "assistant" if msg["role"] in ("assistant", "model") else "user"
            messages.append({"role": role, "content": msg["content"]})

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": resolved_model_id,
            "max_tokens": 4096,
            "messages": messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        data = await self._send_request(
            url=self._base_url,
            headers=headers,
            payload=payload,
            provider_name="Anthropic",
        )

        try:
            content_blocks = data.get("content", [])
            if not isinstance(content_blocks, list) or not content_blocks:
                raise ProviderExecutionError("Anthropic response did not contain content blocks.")

            text_parts = [
                b.get("text", "")
                for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "".join(text_parts).strip()
            if not content:
                raise ProviderExecutionError("Anthropic response contained no text blocks.")

            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

            return {
                "content": content,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "raw_response": data,
            }
        except KeyError as exc:
            raise ProviderExecutionError(f"Anthropic response missing expected key: {exc}") from exc

