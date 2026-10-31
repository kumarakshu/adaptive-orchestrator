"""Tests for cumulative metrics, Prometheus exposition, and sensitive data redaction."""

import logging
import pytest
import httpx

from app.core.config import Settings
from app.core.logging import (
    RedactingFormatter,
    get_logger,
    redact_sensitive_text,
    safe_error_category,
)
from app.core.metrics import MetricsCollector, collector
from app.main import create_app


def test_metrics_collector_aggregation():
    """Verify in-memory metrics collector aggregates requests, models, escalations, and tokens."""
    c = MetricsCollector()

    # Record HTTP calls
    c.record_http_request("/health", 200, 0.005)
    c.record_http_request("/api/v1/chat", 200, 0.050)
    c.record_http_request("/api/v1/chat", 502, 0.020)

    # Record model executions
    c.record_model_execution("openai", "gpt-4o-mini", success=True)
    c.record_model_execution("openai", "gpt-4o-mini", success=False)

    # Record escalation
    c.record_escalation("lightweight", "balanced")

    # Record tokens
    c.record_tokens("gpt-4o-mini", prompt_tokens=15, completion_tokens=5)

    # Verify snapshot
    snapshot = c.get_snapshot()
    assert snapshot["http"]["total_requests"] == 3
    assert snapshot["models"]["total_executions"] == 2
    assert snapshot["escalations"]["total_escalations"] == 1
    assert snapshot["models"]["tokens_prompt"]["gpt-4o-mini"] == 15
    assert snapshot["models"]["tokens_completion"]["gpt-4o-mini"] == 5

    # Verify Prometheus exposition text
    prom_text = c.generate_prometheus_text()
    assert "http_requests_total" in prom_text
    assert 'endpoint="/api/v1/chat",status="200"' in prom_text
    assert 'endpoint="/api/v1/chat",status="502"' in prom_text
    assert "llm_model_executions_total" in prom_text
    assert 'provider="openai",model="gpt-4o-mini",status="success"' in prom_text
    assert "llm_escalations_total" in prom_text
    assert 'from_tier="lightweight",to_tier="balanced"' in prom_text
    assert "llm_tokens_prompt_total" in prom_text


@pytest.mark.asyncio
async def test_metrics_endpoints():
    """Verify GET /metrics and GET /api/v1/metrics expose cumulative data."""
    collector.reset()
    collector.record_http_request("/test-route", 200, 0.012)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Prometheus text endpoint
        res_prom = await client.get("/metrics")
        assert res_prom.status_code == 200
        assert "text/plain" in res_prom.headers.get("content-type", "")
        assert "http_requests_total" in res_prom.text

        # 2. JSON API endpoint
        res_json = await client.get("/api/v1/metrics")
        assert res_json.status_code == 200
        data = res_json.json()
        assert "http" in data
        assert "models" in data
        assert "escalations" in data


@pytest.mark.asyncio
async def test_metrics_disabled_returns_404():
    """Verify that when METRICS_ENABLED=False, metrics endpoints return 404."""
    disabled_settings = Settings(_env_file=None, METRICS_ENABLED=False)
    app = create_app(custom_settings=disabled_settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        prom_res = await client.get("/metrics")
        assert prom_res.status_code == 404

        json_res = await client.get("/api/v1/metrics")
        assert json_res.status_code == 404


def test_redact_sensitive_text_patterns():
    """Verify redaction strips API keys, tokens, and credentials from strings."""
    # Fake test strings only
    openai_raw = "Call failed with key sk-12345678901234567890abcdef and status 401"
    assert "sk-12345678901234567890abcdef" not in redact_sensitive_text(openai_raw)
    assert "[REDACTED_API_KEY]" in redact_sensitive_text(openai_raw)

    gemini_raw = "Gemini key AIzaSyDUMMYKEY1234567890123456789012 rejected"
    assert "AIzaSy" not in redact_sensitive_text(gemini_raw)
    assert "[REDACTED_API_KEY]" in redact_sensitive_text(gemini_raw)

    anthropic_raw = "Anthropic header sk-ant-12345678901234567890abc"
    assert "sk-ant-" not in redact_sensitive_text(anthropic_raw)
    assert "[REDACTED_API_KEY]" in redact_sensitive_text(anthropic_raw)

    bearer_raw = "Authorization: Bearer secret_access_token_12345"
    assert "secret_access_token" not in redact_sensitive_text(bearer_raw)
    assert "Bearer [REDACTED_TOKEN]" in redact_sensitive_text(bearer_raw)

    param_raw = "Connecting to https://api.service.com?api_key=mysecretparamkey123"
    assert "mysecretparamkey123" not in redact_sensitive_text(param_raw)
    assert "[REDACTED_CREDENTIAL]" in redact_sensitive_text(param_raw)


def test_redacting_formatter():
    """Verify RedactingFormatter scrubs log records."""
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Leaked key: sk-abcdefghijklmnopqrstuvwxyz123",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "sk-abcdef" not in formatted
    assert "[REDACTED_API_KEY]" in formatted


def test_safe_error_category():
    """Verify raw exceptions are mapped to safe categories without leaking raw strings."""
    auth_err = RuntimeError("Invalid API Key sk-1234567890 (HTTP 401)")
    category = safe_error_category(auth_err)
    assert "sk-123" not in category
    assert "Authentication/Authorization failure" in category

    timeout_err = TimeoutError("Request timed out after 30s")
    assert "Upstream connection timeout" in safe_error_category(timeout_err)

    rate_err = Exception("Rate limit 429 quota exceeded")
    assert "Rate limit exceeded" in safe_error_category(rate_err)

    conn_err = ConnectionResetError("Connection refused by host")
    assert "Network connection error" in safe_error_category(conn_err)
