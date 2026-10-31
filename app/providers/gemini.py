"""Google Gemini provider adapter for Adaptive LLM Orchestrator."""

import asyncio
from typing import Any, Dict, List, Optional, Union
import httpx

from app.core.logging import get_logger
from app.models.registry import ModelDescriptor
from app.orchestrator.executor import ExecutionResult, ProviderExecutionError
from app.providers.base import BaseHTTPProvider

logger = get_logger("provider.gemini")


class GeminiProvider(BaseHTTPProvider):
    """Provider adapter for Google Gemini generative AI models."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        max_retries: int = 1,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            client=client,
            timeout_seconds=timeout_seconds,
        )
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._max_retries = max(0, min(int(max_retries), 3))
        self._retry_delay_seconds = max(0.0, min(float(retry_delay_seconds), 5.0))

    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], ExecutionResult]:
        """Execute generateContent request with Google Gemini API."""
        api_key = self._ensure_api_key("Google Gemini")

        system_prompt, extracted_messages = self._extract_messages(prompt, context)

        # Gemini expects contents with roles 'user' and 'model'
        contents: List[Dict[str, Any]] = []
        for msg in extracted_messages:
            role = "model" if msg["role"] in ("assistant", "model") else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        payload: Dict[str, Any] = {
            "contents": contents,
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}],
            }

        if context and isinstance(context, dict):
            generation_config = context.get("generation_config")
            if isinstance(generation_config, dict):
                payload["generationConfig"] = generation_config

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/{model.model_id}:generateContent"

        attempt = 0
        while True:
            try:
                data = await self._send_request(
                    url=url,
                    headers=headers,
                    payload=payload,
                    provider_name="Google Gemini",
                )
                break
            except ProviderExecutionError as exc:
                err_str = str(exc).lower()
                is_transient = (
                    "timed out" in err_str
                    or "network communication failure" in err_str
                    or "http 429" in err_str
                    or "http 503" in err_str
                    or "http 500" in err_str
                    or "http 502" in err_str
                    or "http 504" in err_str
                )
                if attempt < self._max_retries and is_transient:
                    attempt += 1
                    logger.warning(
                        "Google Gemini request failed with transient error; retrying attempt %d of %d...",
                        attempt,
                        self._max_retries,
                    )
                    await asyncio.sleep(self._retry_delay_seconds)
                    continue
                raise exc

        try:
            candidates = data.get("candidates", [])
            if not isinstance(candidates, list) or not candidates:
                # Check for block reason or finish reason
                prompt_feedback = data.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason")
                if block_reason:
                    raise ProviderExecutionError(
                        f"Google Gemini blocked prompt with reason: {block_reason}"
                    )
                raise ProviderExecutionError("Google Gemini response contained no candidates.")

            first_candidate = candidates[0]
            candidate_content = first_candidate.get("content", {})
            parts = candidate_content.get("parts", [])
            if not isinstance(parts, list) or not parts:
                raise ProviderExecutionError("Google Gemini candidate contained no content parts.")

            text_parts = [
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and "text" in p
            ]
            content = "".join(text_parts).strip()
            if not content:
                raise ProviderExecutionError("Google Gemini candidate parts contained empty text.")

            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("candidatesTokenCount", 0) if isinstance(usage, dict) else 0

            return {
                "content": content,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "raw_response": data,
            }
        except KeyError as exc:
            raise ProviderExecutionError(f"Google Gemini response missing expected key: {exc}") from exc

