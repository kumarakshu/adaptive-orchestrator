"""Structured logging and sensitive data redaction module.

Provides a hardened logging setup using the standard library, ensuring API keys,
tokens, and sensitive authorization headers are redacted before output.
"""

import logging
import re
from typing import Optional

from app.core.sanitizer import (
    REDACTION_PATTERNS,
    redact_sensitive_text,
    sanitize_data,
)


class RedactingFormatter(logging.Formatter):
    """Logging formatter that redacts credentials and sensitive data from all log records."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact_sensitive_text(original)


def safe_error_category(exc: Exception) -> str:
    """Map raw exceptions to sanitized high-level error categories.

    Prevents raw exception messages (which may inadvertently contain connection
    strings, query fragments, or credentials) from leaking into external logs.
    """
    exc_type = type(exc).__name__
    msg = str(exc).lower()

    if "auth" in msg or "401" in msg or "403" in msg or "permission" in msg or "key" in msg:
        return f"{exc_type}: Authentication/Authorization failure"
    if "timeout" in msg or "timed out" in msg:
        return f"{exc_type}: Upstream connection timeout"
    if "rate limit" in msg or "429" in msg or "quota" in msg:
        return f"{exc_type}: Rate limit exceeded"
    if "connection" in msg or "refused" in msg or "disconnect" in msg:
        return f"{exc_type}: Network connection error"
    return f"{exc_type}: Internal processing error"


_configured = False


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root and application logger with redaction formatting."""
    global _configured
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Replace existing handlers with a redacting stream handler
    formatter = RedactingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    _configured = True


def get_logger(name: str = "adaptive_orchestrator") -> logging.Logger:
    """Retrieve an application logger with redaction formatting."""
    if not _configured:
        setup_logging("INFO")
    return logging.getLogger(name)

