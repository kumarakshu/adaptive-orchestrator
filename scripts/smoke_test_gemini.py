"""Minimal, controlled live smoke test for Google Gemini using Interactions API.

Runs a safe diagnostic experiment against Google Gemini Interactions API using
httpx.AsyncClient.stream to measure time-to-first-byte (headers) vs. body-read latency.
Prints rich operational diagnostics without exposing secrets, headers, or prompt contents.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional
import httpx

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure Windows uses SelectorEventLoop for network compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.config import get_settings


async def run_smoke_test() -> int:
    """Execute a single live diagnostic stream request against Google Gemini."""
    # Diagnostic experiment note: previously tested {"generation_config": {"thinkingLevel": "low"}}
    # Current experiment omits generation_config and system_instruction to measure minimal latency.
    settings = get_settings()

    if not settings.gemini_api_key or not settings.gemini_api_key.strip():
        print("=== Gemini Live Smoke Test ===")
        print("Status: FAILED")
        print("Error: GEMINI_API_KEY is not configured in environment or .env file.")
        return 1

    model_id = "gemini-3.8-flash"
    timeout_budget = 40.0
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    headers = {
        "x-goog-api-key": settings.gemini_api_key,
        "Content-Type": "application/json",
    }

    # Minimal payload: omit generation_config and system_instruction
    payload: Dict[str, Any] = {
        "model": model_id,
        "input": "Reply with exactly OK",
        "store": False,
    }

    start_time = time.perf_counter()
    header_latency_ms: Optional[float] = None
    body_latency_ms: Optional[float] = None
    status_code: Optional[int] = None

    timeout = httpx.Timeout(timeout_budget)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            t_req_start = time.perf_counter()
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                header_latency_ms = (time.perf_counter() - t_req_start) * 1000
                status_code = response.status_code

                t_body_start = time.perf_counter()
                body_bytes = await response.aread()
                body_latency_ms = (time.perf_counter() - t_body_start) * 1000

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Safe response parsing
        data: Dict[str, Any] = {}
        try:
            data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            pass

        content = ""
        out_text = data.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            content = out_text.strip()
        elif "steps" in data and isinstance(data["steps"], list):
            for step in data["steps"]:
                if isinstance(step, dict) and step.get("type") == "model_output":
                    step_content = step.get("content")
                    if isinstance(step_content, list):
                        parts = [
                            p.get("text", "")
                            for p in step_content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        combined = "".join(parts).strip()
                        if combined:
                            content = combined
                            break

        usable_output_detected = bool(content and content.strip())
        usage = data.get("usage", {})
        prompt_tokens = usage.get("total_input_tokens", usage.get("prompt_tokens", "N/A"))
        completion_tokens = usage.get("total_output_tokens", usage.get("completion_tokens", "N/A"))
        interaction_id = data.get("id", "N/A")
        interaction_status = data.get("status", "N/A")

        steps = data.get("steps", [])
        steps_count = len(steps) if isinstance(steps, list) else 0
        step_types = [s.get("type") for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []

        if status_code == 200 and usable_output_detected:
            print("=== Gemini Live Diagnostic Smoke Test ===")
            print("Status: SUCCESS")
            print(f"Target Model: {model_id}")
            print(f"HTTP Status: {status_code}")
            print(f"Header Latency (TTFB): {header_latency_ms:.1f} ms")
            print(f"Body Read Latency: {body_latency_ms:.1f} ms")
            print(f"Total Latency: {total_latency_ms:.1f} ms")
            print(f"Usable Output Detected: {usable_output_detected}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Interaction ID: {interaction_id}")
            print(f"Interaction Status: {interaction_status}")
            print(f"Steps Captured: {steps_count} {step_types}")
            print(f"Timeout Budget: {timeout_budget:.1f}s (single attempt, no retries)")
            return 0
        elif status_code == 200:
            print("=== Gemini Live Diagnostic Smoke Test ===")
            print("Status: FAILED")
            print(f"Target Model: {model_id}")
            print(f"HTTP Status: {status_code}")
            print(f"Header Latency (TTFB): {header_latency_ms:.1f} ms")
            print(f"Body Read Latency: {body_latency_ms:.1f} ms")
            print(f"Total Latency: {total_latency_ms:.1f} ms")
            print(f"Usable Output Detected: {usable_output_detected}")
            print(f"Error Detail: HTTP 200 received but no usable model output detected in response")
            print(f"Interaction ID: {interaction_id}")
            print(f"Interaction Status: {interaction_status}")
            print(f"Steps Captured: {steps_count} {step_types}")
            print(f"Timeout Budget: {timeout_budget:.1f}s (single attempt, no retries)")
            return 1
        else:
            err_msg = data.get("error", {}).get("message", "Unknown error from upstream")
            safe_err = str(err_msg)[:200].replace("\n", " ")
            print("=== Gemini Live Diagnostic Smoke Test ===")
            print("Status: FAILED")
            print(f"Target Model: {model_id}")
            print(f"HTTP Status: {status_code}")
            if header_latency_ms is not None:
                print(f"Header Latency (TTFB): {header_latency_ms:.1f} ms")
            if body_latency_ms is not None:
                print(f"Body Read Latency: {body_latency_ms:.1f} ms")
            print(f"Total Latency: {total_latency_ms:.1f} ms")
            print(f"Usable Output Detected: {usable_output_detected}")
            print(f"Error Detail: {safe_err}")
            print(f"Timeout Budget: {timeout_budget:.1f}s (single attempt, no retries)")
            return 1

    except httpx.TimeoutException as exc:
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        exc_type = type(exc).__name__
        print("=== Gemini Live Diagnostic Smoke Test ===")
        print("Status: FAILED")
        print(f"Target Model: {model_id}")
        print(f"HTTP Status: {status_code if status_code is not None else 'N/A'}")
        print(f"Exception Class: {exc_type}")
        if header_latency_ms is not None:
            print(f"Header Latency (TTFB): {header_latency_ms:.1f} ms (headers arrived before timeout)")
        else:
            print("Header Latency (TTFB): Not reached (timed out waiting for response headers)")
        if body_latency_ms is not None:
            print(f"Body Read Latency: {body_latency_ms:.1f} ms")
        else:
            print("Body Read Latency: Not reached")
        print(f"Total Latency: {total_latency_ms:.1f} ms")
        print(f"Error Detail: Request timed out after {timeout_budget:.1f}s ({exc_type})")
        print(f"Timeout Budget: {timeout_budget:.1f}s (single attempt, no retries)")
        return 1

    except Exception as exc:
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        exc_type = type(exc).__name__
        print("=== Gemini Live Diagnostic Smoke Test ===")
        print("Status: FAILED")
        print(f"Target Model: {model_id}")
        print(f"HTTP Status: {status_code if status_code is not None else 'N/A'}")
        print(f"Exception Class: {exc_type}")
        print(f"Total Latency: {total_latency_ms:.1f} ms")
        print(f"Error Detail: Unexpected failure ({exc_type})")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_smoke_test())
    sys.exit(exit_code)
