"""Google Gemini provider adapter for Adaptive LLM Orchestrator using Interactions API."""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Union
import httpx

from app.core.logging import get_logger
from app.models.registry import ModelDescriptor
from app.orchestrator.executor import ExecutionResult, ProviderExecutionError
from app.providers.base import BaseHTTPProvider

logger = get_logger("provider.gemini")


class GeminiProvider(BaseHTTPProvider):
    """Provider adapter for Google Gemini generative AI models using the Interactions API."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        max_retries: int = 1,
        retry_delay_seconds: float = 1.0,
        attempt_timeout_seconds: float = 22.0,
        api_mode: str = "interactions",
    ) -> None:
        super().__init__(
            api_key=api_key,
            client=client,
            timeout_seconds=timeout_seconds,
        )
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        # If legacy base_url pointing to /models was passed, normalize to /interactions
        if self._base_url.endswith("/models"):
            self._base_url = self._base_url[:-7] + "/interactions"
        mode = (api_mode or "interactions").strip().lower().replace("-", "_")
        self._api_mode = mode if mode in ("interactions", "generate_content") else "interactions"
        self._max_retries = max(0, min(int(max_retries), 3))
        self._retry_delay_seconds = max(0.0, min(float(retry_delay_seconds), 5.0))
        self._attempt_timeout_seconds = max(1.0, min(float(attempt_timeout_seconds), 60.0))

    @property
    def api_mode(self) -> str:
        """Active API mode: 'interactions' (beta) or 'generate_content' (stable)."""
        return self._api_mode

    def _generate_content_url(self, model_id: str) -> str:
        """Build the stable generateContent endpoint URL for a model."""
        root = self._base_url
        for suffix in ("/interactions", "/models"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
        name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        return f"{root}/{name}:generateContent"

    @property
    def max_retries(self) -> int:
        """Configured maximum retry attempts for transient errors."""
        return self._max_retries

    @property
    def retry_delay_seconds(self) -> float:
        """Configured delay in seconds between retry attempts."""
        return self._retry_delay_seconds

    @property
    def attempt_timeout_seconds(self) -> float:
        """Configured per-attempt timeout in seconds."""
        return self._attempt_timeout_seconds

    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any], ExecutionResult]:
        """Execute interaction request with Google Gemini Interactions API."""
        api_key = self._ensure_api_key("Google Gemini")

        system_prompt, extracted_messages = self._extract_messages(prompt, context)

        # Map conversation history if multiple turns exist, otherwise use direct prompt
        if extracted_messages and len(extracted_messages) > 1:
            turns = []
            for msg in extracted_messages:
                speaker = "User" if msg["role"] == "user" else "Model"
                turns.append(f"{speaker}: {msg['content']}")
            input_text = "\n\n".join(turns)
        else:
            input_text = prompt

        # Generation config forwarding & normalization
        gen_config: Dict[str, Any] = {}
        if context and isinstance(context, dict):
            raw_gen_config = context.get("generation_config")
            if isinstance(raw_gen_config, dict):
                gen_config = dict(raw_gen_config)
                # Normalize legacy thinkingConfig -> thinking_level
                if "thinkingConfig" in gen_config and isinstance(gen_config["thinkingConfig"], dict):
                    tc = gen_config.pop("thinkingConfig")
                    lvl = tc.get("thinkingLevel") or tc.get("thinking_level")
                    if lvl and "thinking_level" not in gen_config:
                        gen_config["thinking_level"] = str(lvl).lower()

        # Apply thinking_level=low for reasoning models if not explicitly specified
        if "thinking_level" not in gen_config:
            if "gemini-3" in model.model_id.lower():
                gen_config["thinking_level"] = "low"

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        if self._api_mode == "generate_content":
            url = self._generate_content_url(model.model_id)
            contents = [
                {
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}],
                }
                for msg in extracted_messages
            ]
            payload: Dict[str, Any] = {"contents": contents}
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            gc: Dict[str, Any] = {}
            level = gen_config.pop("thinking_level", None)
            if level:
                gc["thinkingConfig"] = {"thinkingLevel": str(level).lower()}
            for key, value in gen_config.items():
                gc[key] = value
            if gc:
                payload["generationConfig"] = gc
        else:
            url = self._base_url
            payload = {
                "model": model.model_id,
                "input": input_text,
                "store": False,
            }
            if system_prompt:
                payload["system_instruction"] = system_prompt
            if gen_config:
                payload["generation_config"] = gen_config

        total_budget = float(self._timeout_seconds)
        start_time = time.monotonic()

        attempt = 0
        while True:
            elapsed = time.monotonic() - start_time
            remaining_budget = total_budget - elapsed
            if remaining_budget <= 0.5:
                raise ProviderExecutionError(
                    f"Google Gemini request timed out after total time budget of {total_budget:.1f}s."
                )

            retries_left = self._max_retries - attempt
            if retries_left > 0 and remaining_budget > self._attempt_timeout_seconds:
                # Leave budget for backoff and subsequent attempts
                call_timeout = min(self._attempt_timeout_seconds, remaining_budget - 1.0)
            else:
                call_timeout = min(self._attempt_timeout_seconds, remaining_budget)
            call_timeout = max(1.0, call_timeout)
            try:
                data = await self._send_request(
                    url=url,
                    headers=headers,
                    payload=payload,
                    provider_name="Google Gemini",
                    timeout_seconds=call_timeout,
                )
                break
            except ProviderExecutionError as exc:
                status_code = getattr(exc, "status_code", None)
                err_str = str(exc).lower()

                # Strictly avoid retrying non-transient client errors (400, 401, 403, 404, 422)
                if status_code in (400, 401, 403, 404, 422) or any(
                    f"http {code}" in err_str for code in (400, 401, 403, 404, 422)
                ):
                    raise exc

                # Transient errors eligible for retry
                is_transient = (
                    status_code in (429, 500, 502, 503, 504)
                    or "timed out" in err_str
                    or "network communication failure" in err_str
                    or "high demand" in err_str
                    or any(f"http {code}" in err_str for code in (429, 500, 502, 503, 504))
                )

                if attempt < self._max_retries and is_transient:
                    # Determine delay with Retry-After or exponential backoff + jitter
                    if getattr(exc, "retry_after", None) is not None and exc.retry_after > 0:
                        delay = min(float(exc.retry_after), 5.0)
                    else:
                        base_delay = self._retry_delay_seconds * (2 ** attempt)
                        jitter = random.uniform(0.1, 0.3) if self._retry_delay_seconds > 0 else 0.0
                        delay = min(base_delay + jitter, 5.0)

                    time_left = total_budget - (time.monotonic() - start_time)
                    min_retry_budget = 10.0
                    if (time_left - delay) < min_retry_budget:
                        logger.warning(
                            "Google Gemini: aborting retry; insufficient time budget remaining for viable attempt (%.2fs left, need %.2fs).",
                            time_left - delay,
                            min_retry_budget,
                        )
                        raise exc

                    # If this was a timeout and remaining time is less than needed for generation,
                    # retrying with insufficient time is guaranteed to fail.
                    if "timed out" in err_str and (time_left - delay) < (self._attempt_timeout_seconds * 0.75):
                        logger.warning(
                            "Google Gemini: aborting retry on timeout; remaining budget (%.2fs) is less than required for generation.",
                            time_left - delay,
                        )
                        raise exc

                    if delay >= time_left - 0.5:
                        logger.warning(
                            "Google Gemini: aborting retry; insufficient time budget remaining (%.2fs left, need %.2fs delay).",
                            time_left,
                            delay,
                        )
                        raise exc

                    attempt += 1
                    logger.warning(
                        "Google Gemini transient failure (status: %s); retrying in %.2fs (attempt %d of %d)...",
                        status_code or "transient",
                        delay,
                        attempt,
                        self._max_retries,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                raise exc

        # Parse response content safely
        content = ""

        # 1. Direct output_text convenience attribute (must be non-empty string)
        out_text = data.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            content = out_text.strip()

        # 2. Extract from steps timeline (model_output)
        if not content and isinstance(data.get("steps"), list):
            text_parts = []
            for step in data["steps"]:
                if not isinstance(step, dict):
                    continue
                if step.get("type") in ("model_output", "output"):
                    # Content block
                    step_content = step.get("content")
                    if isinstance(step_content, list):
                        for part in step_content:
                            if isinstance(part, dict):
                                txt = part.get("text") or part.get("value")
                                if isinstance(txt, str) and txt:
                                    text_parts.append(txt)
                            elif isinstance(part, str) and part:
                                text_parts.append(part)
                    elif isinstance(step_content, str) and step_content:
                        text_parts.append(step_content)

                    # Parts block
                    step_parts = step.get("parts")
                    if isinstance(step_parts, list):
                        for part in step_parts:
                            if isinstance(part, dict) and "text" in part and part["text"]:
                                text_parts.append(part["text"])
                            elif isinstance(part, str) and part:
                                text_parts.append(part)

                    # Direct text or output field
                    if isinstance(step.get("text"), str) and step["text"]:
                        text_parts.append(step["text"])
                    elif isinstance(step.get("output"), str) and step["output"]:
                        text_parts.append(step["output"])
            content = "".join(text_parts).strip()

        # 3. Direct outputs block
        if not content and isinstance(data.get("outputs"), list):
            out_parts = []
            for item in data["outputs"]:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("value")
                    if isinstance(txt, str) and txt:
                        out_parts.append(txt)
                elif isinstance(item, str) and item:
                    out_parts.append(item)
            content = "".join(out_parts).strip()

        # 4. Fallback for legacy candidates shape
        if not content and isinstance(data.get("candidates"), list) and data["candidates"]:
            cand = data["candidates"][0]
            cand_parts = cand.get("content", {}).get("parts", [])
            if isinstance(cand_parts, list):
                text_parts = [
                    p.get("text", "")
                    for p in cand_parts
                    if isinstance(p, dict) and "text" in p
                ]
                content = "".join(text_parts).strip()

        if not content:
            # Check for error or non-completed status
            status = data.get("status")
            if status == "failed":
                err_msg = data.get("error", {}).get("message") or "Interaction execution failed"
                raise ProviderExecutionError(f"Google Gemini interaction failed: {err_msg}")
            raise ProviderExecutionError("Google Gemini response contained empty content.")

        usage = data.get("usage") or data.get("usageMetadata") or {}
        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
        completion_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidatesTokenCount")
            or 0
        )

        return {
            "content": content,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "raw_response": data,
        }
