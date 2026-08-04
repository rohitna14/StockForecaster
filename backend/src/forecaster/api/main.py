"""FastAPI application factory.

The service is deliberately thin: it validates input, calls the domain layer,
and serialises the result. All forecasting logic lives in ``forecaster.*`` and
is exercised by the CLI and the test suite without any HTTP involved, which is
what lets the Streamlit and Next.js clients be genuinely interchangeable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from forecaster.api.deps import close_provider_router
from forecaster.api.errors import register_exception_handlers
from forecaster.config import Environment, get_settings
from forecaster.db.session import create_all, dispose_engine
from forecaster.logging import configure_logging, get_logger, log_context

log = get_logger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """
Leakage-free stock forecasting API.

**What this service will and will not tell you.**

Every forecast is returned alongside the model's measured skill against a naive
baseline. When a model historically *lost* to that baseline, the response says
so via `is_informative: false` rather than presenting the number as useful.

The honest summary of this project's own findings:

* **Directional price prediction does not work.** No model beat the naive
  baseline at any horizon tested.
* **Volatility forecasting does work**, with a statistically significant
  improvement over a random-walk baseline.

See `/api/v1/methodology` and `docs/RESULTS.md`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()

    if settings.is_sqlite:
        # Local/dev convenience only; production schema comes from Alembic.
        await create_all()

    log.info(
        "api_started",
        environment=settings.environment.value,
        database=settings.database_url.split("://")[0],
    )
    try:
        yield
    finally:
        await close_provider_router()
        await dispose_engine()
        log.info("api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="StockForecaster API",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Outside production, accept any localhost port. Dev servers shuffle ports
    # whenever one is busy (Next falls back 3000 -> 3001 -> 3002), and a fixed
    # allow-list turns that into a broken app with a misleading "cannot reach
    # the API" error. The regex is deliberately NOT applied in production,
    # where origins must be enumerated explicitly.
    local_origin_regex = (
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
        if settings.environment is not Environment.PRODUCTION
        else None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_origin_regex=local_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        with log_context(request_id=request_id, path=request.url.path):
            response = await call_next(request)

        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"

        # Health checks would otherwise dominate the logs.
        if not request.url.path.startswith(f"{API_PREFIX}/health"):
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round(elapsed * 1000, 1),
            )
        return response

    register_exception_handlers(app)

    from forecaster.api.routers import (
        backtests,
        health,
        instruments,
        models,
        runs,
        search,
    )

    # `search` is registered before `instruments` so /search and /resolve are
    # matched before the /instruments/{symbol} catch-all.
    for router in (health, search, instruments, models, runs, backtests):
        app.include_router(router.router, prefix=API_PREFIX)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "StockForecaster API",
            "version": "0.1.0",
            "docs": "/docs",
            "api": API_PREFIX,
        }

    return app


app = create_app()
