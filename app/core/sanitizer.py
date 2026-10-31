"""Centralized data sanitization and secret redaction module.

Provides regex-based redaction of API keys, authentication tokens, and credentials
from strings and arbitrary nested data structures (dicts, lists, tuples, sets)
prior to logging or persistence in Redis/in-memory stores.
"""

import re
from typing import Any, List, Tuple

# Regex patterns identifying common API keys, tokens, and secrets
REDACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # OpenAI standard & project keys
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    # Google Gemini / Cloud API keys
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "[REDACTED_API_KEY]"),
    # Anthropic API keys (sk-ant-...)
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    # Authorization: Bearer tokens
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]+"), "Bearer [REDACTED_TOKEN]"),
    # Generic key-value secret fields (e.g., api_key="...", password="...")
    (
        re.compile(
            r"""(?i)\b(api[-_]?key|secret|token|password|auth(?:orization)?)\b(\s*[:=]\s*['"]?)[A-Za-z0-9_\-\.]{8,}""",
        ),
        r"\1\2[REDACTED_CREDENTIAL]",
    ),
]


def redact_sensitive_text(text: str) -> str:
    """Scrub sensitive credentials, tokens, and secrets from an arbitrary string."""
    if not text:
        return ""
    sanitized = str(text)
    for pattern, replacement in REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_data(data: Any) -> Any:
    """Recursively sanitize sensitive strings within nested data structures.

    Handles strings, dicts, lists, tuples, sets, and primitive values.
    Ensures no credentials or bearer tokens are persisted in storage or logs.
    """
    if data is None:
        return None
    if isinstance(data, str):
        return redact_sensitive_text(data)
    if isinstance(data, dict):
        return {sanitize_data(k): sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(sanitize_data(item) for item in data)
    if isinstance(data, set):
        return {sanitize_data(item) for item in data}
    return data

