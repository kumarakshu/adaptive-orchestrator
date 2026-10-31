"""Minimal, controlled live smoke test for Google Gemini.

Makes exactly ONE request to verify live API key functionality and upstream
connectivity without retries, without calling other providers, and without
revealing credentials.
"""

import asyncio
import sys
import time

# Ensure Windows uses SelectorEventLoop for network compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.config import get_settings
from app.models.registry import ModelDescriptor, ModelTier
from app.orchestrator.executor import ProviderExecutionError
from app.providers.gemini import GeminiProvider


async def run_smoke_test() -> int:
    """Execute a single live verification call against Google Gemini."""
    settings = get_settings()

    if not settings.gemini_api_key or not settings.gemini_api_key.strip():
        print("=== Gemini Live Smoke Test ===")
        print("Status: FAILED")
        print("Error: GEMINI_API_KEY is not configured in environment or .env file.")
        return 1

    model = ModelDescriptor(
        model_id="gemini-3.8-flash",
        provider="google",
        tier=ModelTier.LIGHTWEIGHT,
        context_window=1048576,
        supported_capabilities=["chat"],
    )

    provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        timeout_seconds=float(settings.default_timeout_seconds),
    )

    prompt = "Respond with only the word 'pong'."
    context = {
        "generation_config": {
            "thinkingConfig": {
                "thinkingLevel": "low"
            }
        }
    }

    start_time = time.perf_counter()
    try:
        # Strictly single call - no retries
        result = await provider.call(prompt=prompt, model=model, context=context)
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = result.get("content", "") if isinstance(result, dict) else str(result)
        # Limit response text to prevent any accidental verbose exposure
        safe_preview = content.strip().replace("\n", " ")[:100]

        prompt_tokens = result.get("prompt_tokens") if isinstance(result, dict) else "N/A"
        completion_tokens = result.get("completion_tokens") if isinstance(result, dict) else "N/A"

        print("=== Gemini Live Smoke Test ===")
        print("Status: SUCCESS")
        print(f"Target Model: {model.model_id}")
        print(f"Response: {safe_preview}")
        print(f"Latency: {latency_ms:.1f} ms")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        return 0

    except ProviderExecutionError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print("=== Gemini Live Smoke Test ===")
        print("Status: FAILED")
        print(f"Target Model: {model.model_id}")
        print(f"Error: {exc}")
        print(f"Latency: {latency_ms:.1f} ms")
        return 1
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print("=== Gemini Live Smoke Test ===")
        print("Status: FAILED")
        print(f"Target Model: {model.model_id}")
        print(f"Error: Unexpected failure ({type(exc).__name__})")
        print(f"Latency: {latency_ms:.1f} ms")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_smoke_test())
    sys.exit(exit_code)

