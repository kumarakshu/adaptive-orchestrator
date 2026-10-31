"""Cumulative metrics collection and Prometheus exposition module.

Built entirely using the Python standard library to track throughput, latency,
model executions, escalation events, and token consumption without external dependencies.
"""

from collections import defaultdict
import threading
import time
from typing import Any, Dict, Optional, Tuple


class MetricsCollector:
    """Thread-safe cumulative in-memory metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial empty state (primarily for test isolation)."""
        with self._lock:
            # (endpoint, status_code) -> count
            self._http_requests_total: Dict[Tuple[str, int], int] = defaultdict(int)
            self._http_duration_sum_seconds: float = 0.0
            self._http_duration_count: int = 0

            # (provider, model_id, status) -> count
            self._model_executions_total: Dict[Tuple[str, str, str], int] = defaultdict(int)

            # (from_tier, to_tier) -> count
            self._escalations_total: Dict[Tuple[str, str], int] = defaultdict(int)

            # model_id -> count
            self._tokens_prompt_total: Dict[str, int] = defaultdict(int)
            self._tokens_completion_total: Dict[str, int] = defaultdict(int)

            self._start_time: float = time.time()

    def record_http_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record an incoming HTTP request completion.

        Labels are strictly bounded (normalized endpoint route and integer status code).
        """
        # Normalize endpoint path to avoid cardinality explosion
        clean_endpoint = endpoint.split("?")[0].strip() or "/"
        with self._lock:
            self._http_requests_total[(clean_endpoint, status_code)] += 1
            if duration_seconds >= 0:
                self._http_duration_sum_seconds += duration_seconds
                self._http_duration_count += 1

    def record_model_execution(self, provider: str, model_id: str, success: bool) -> None:
        """Record an individual model execution call attempt."""
        status_label = "success" if success else "failure"
        clean_provider = (provider or "unknown").lower().strip()
        clean_model = (model_id or "unknown").strip()
        with self._lock:
            self._model_executions_total[(clean_provider, clean_model, status_label)] += 1

    def record_escalation(self, from_tier: str, to_tier: str) -> None:
        """Record an escalation tier transition event."""
        clean_from = (from_tier or "unknown").lower().strip()
        clean_to = (to_tier or "unknown").lower().strip()
        with self._lock:
            self._escalations_total[(clean_from, clean_to)] += 1

    def record_tokens(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record prompt and completion token counts when supplied by provider."""
        clean_model = (model_id or "unknown").strip()
        with self._lock:
            if prompt_tokens > 0:
                self._tokens_prompt_total[clean_model] += prompt_tokens
            if completion_tokens > 0:
                self._tokens_completion_total[clean_model] += completion_tokens

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a structured, JSON-serializable dictionary of current metrics."""
        with self._lock:
            avg_duration = (
                (self._http_duration_sum_seconds / self._http_duration_count)
                if self._http_duration_count > 0
                else 0.0
            )

            http_breakdown = [
                {"endpoint": ep, "status": code, "count": count}
                for (ep, code), count in sorted(self._http_requests_total.items())
            ]

            model_breakdown = [
                {"provider": prov, "model": m, "status": stat, "count": count}
                for (prov, m, stat), count in sorted(self._model_executions_total.items())
            ]

            escalation_breakdown = [
                {"from_tier": f, "to_tier": t, "count": count}
                for (f, t), count in sorted(self._escalations_total.items())
            ]

            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "http": {
                    "total_requests": sum(self._http_requests_total.values()),
                    "duration_sum_seconds": round(self._http_duration_sum_seconds, 4),
                    "duration_count": self._http_duration_count,
                    "avg_duration_seconds": round(avg_duration, 4),
                    "requests": http_breakdown,
                },
                "models": {
                    "total_executions": sum(self._model_executions_total.values()),
                    "executions": model_breakdown,
                    "tokens_prompt": dict(self._tokens_prompt_total),
                    "tokens_completion": dict(self._tokens_completion_total),
                },
                "escalations": {
                    "total_escalations": sum(self._escalations_total.values()),
                    "transitions": escalation_breakdown,
                },
            }

    def generate_prometheus_text(self) -> str:
        """Format metrics into standard Prometheus exposition format."""
        with self._lock:
            lines = [
                "# HELP http_requests_total Total HTTP requests by endpoint and status code.",
                "# TYPE http_requests_total counter",
            ]
            for (ep, code), count in sorted(self._http_requests_total.items()):
                lines.append(f'http_requests_total{{endpoint="{ep}",status="{code}"}} {count}')

            lines.extend([
                "# HELP http_request_duration_seconds_sum Cumulative HTTP request duration in seconds.",
                "# TYPE http_request_duration_seconds_sum counter",
                f"http_request_duration_seconds_sum {self._http_duration_sum_seconds:.6f}",
                "# HELP http_request_duration_seconds_count Total HTTP requests timed.",
                "# TYPE http_request_duration_seconds_count counter",
                f"http_request_duration_seconds_count {self._http_duration_count}",
                "# HELP llm_model_executions_total Total model invocation attempts by provider, model ID, and status.",
                "# TYPE llm_model_executions_total counter",
            ])
            for (prov, m, stat), count in sorted(self._model_executions_total.items()):
                lines.append(f'llm_model_executions_total{{provider="{prov}",model="{m}",status="{stat}"}} {count}')

            lines.extend([
                "# HELP llm_escalations_total Total model tier escalation transitions.",
                "# TYPE llm_escalations_total counter",
            ])
            for (f, t), count in sorted(self._escalations_total.items()):
                lines.append(f'llm_escalations_total{{from_tier="{f}",to_tier="{t}"}} {count}')

            lines.extend([
                "# HELP llm_tokens_prompt_total Total prompt tokens consumed by model ID.",
                "# TYPE llm_tokens_prompt_total counter",
            ])
            for m, count in sorted(self._tokens_prompt_total.items()):
                lines.append(f'llm_tokens_prompt_total{{model="{m}"}} {count}')

            lines.extend([
                "# HELP llm_tokens_completion_total Total completion tokens generated by model ID.",
                "# TYPE llm_tokens_completion_total counter",
            ])
            for m, count in sorted(self._tokens_completion_total.items()):
                lines.append(f'llm_tokens_completion_total{{model="{m}"}} {count}')

            # Conclude with trailing newline
            lines.append("")
            return "\n".join(lines)


# Shared global singleton collector
collector = MetricsCollector()

