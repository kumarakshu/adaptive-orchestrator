"""Main FastAPI application entrypoint."""

from contextlib import asynccontextmanager
import time
from typing import Any, AsyncGenerator, Optional
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api.routes import router as api_router
from app.api.schemas import HealthResponse
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.core.metrics import collector as metrics_collector

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup tasks
    yield
    # Shutdown tasks


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Factory creating configured FastAPI application instance.

    Configures:
    - CORS middleware securely (never pairs wildcard origins with credentials).
    - Structured logging with API key / token redaction.
    - HTTP request metrics tracking.
    - Prometheus /metrics and health check endpoints.
    """
    app_settings = custom_settings or get_settings()
    setup_logging(app_settings.log_level)

    application = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Adaptive multi-tier LLM orchestration service with intelligent routing and execution.",
        lifespan=lifespan,
        debug=app_settings.debug,
    )
    application.state.settings = app_settings

    # Determine allowed origins
    allowed_origins = list(app_settings.cors_allowed_origins)
    has_wildcard = "*" in allowed_origins

    # Security rule: Wildcard origins must never be combined with allow_credentials=True
    allow_credentials = app_settings.cors_allow_credentials and not has_wildcard

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class HTTPMetricsMiddleware:
        """Pure ASGI middleware tracking HTTP request status and latency without BaseHTTPMiddleware."""

        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return

            start_time = time.perf_counter()
            status_code = [500]

            async def send_wrapper(message: Any) -> None:
                if message.get("type") == "http.response.start":
                    status_code[0] = message.get("status", 500)
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                duration = time.perf_counter() - start_time
                endpoint = scope.get("path", "/")
                metrics_collector.record_http_request(
                    endpoint=endpoint,
                    status_code=status_code[0],
                    duration_seconds=duration,
                )

    application.add_middleware(HTTPMetricsMiddleware)

    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        summary="Root Health Check",
        tags=["System"],
    )
    async def root_health() -> HealthResponse:
        """Return health check indicating the API process is active."""
        return HealthResponse(
            status="ok",
            app=app_settings.app_name,
            version="0.1.0",
        )

    @application.get(
        "/metrics",
        summary="Prometheus Metrics",
        tags=["System"],
    )
    async def prometheus_metrics() -> Response:
        """Expose cumulative service metrics in standard Prometheus text exposition format."""
        if not app_settings.metrics_enabled:
            return PlainTextResponse(
                "Metrics collection is disabled.\n",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return PlainTextResponse(
            metrics_collector.generate_prometheus_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Include versioned API routes
    application.include_router(api_router, prefix="/api/v1")

    return application


app = create_app(settings)



