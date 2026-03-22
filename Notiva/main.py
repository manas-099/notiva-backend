"""
main.py — FastAPI application entry point
──────────────────────────────────────────
Project layout expected:
  your_project/
  ├── main.py              ← this file
  ├── logging_config.py
  ├── dependencies.py
  ├── meetingdb.py
  ├── meeting_llm.py
  ├── note_taker.py
  ├── stt.py               (standalone CLI runner — not imported here)
  └── routes/
      ├── __init__.py
      └── meetings.py

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Deploy on Railway:
  Set start command → uvicorn main:app --host 0.0.0.0 --port $PORT
  Env vars needed:
    DATABASE_URL
    SARVAM_API_KEY
    OPENROUTER_API_KEY
    SENDGRID_API_KEY
    FROM_EMAIL
    LOG_FORMAT=json          (Railway log aggregation)
    LOG_LEVEL=INFO
    ALLOWED_ORIGINS=https://your-electron-app.local
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

# ── Load .env FIRST — before any module reads os.environ ─────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
# ─────────────────────────────────────────────────────────────────────────────

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Configure logging FIRST — before any other import that uses structlog ─────
from .logging_config import configure_logging
configure_logging()

from .Dependences import get_app_state
from .Meetingdb import MeetingDBError, create_pool, run_migrations
from .routes.meetings import router as meetings_router

log = structlog.get_logger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup → yields (app serves requests) → runs on shutdown.
    Replaces the old @app.on_event("startup") / ("shutdown") pattern.
    """
    state = get_app_state()

    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("app.startup_begin", environment=os.environ.get("ENVIRONMENT", "development"))

    # Validate required env vars early
    missing = [v for v in ["DATABASE_URL", "SARVAM_API_KEY", "OPENROUTER_API_KEY"]
               if not os.environ.get(v)]
    if missing:
        log.error("app.missing_env_vars", missing=missing)
        raise RuntimeError(f"Missing required env vars: {missing}")

    # Create DB pool + run migrations
    try:
        state.pool = await create_pool()
        await run_migrations(state.pool)
        log.info("app.db_ready")
    except MeetingDBError as exc:
        log.error("app.db_startup_failed", error=str(exc))
        raise RuntimeError(f"DB startup failed: {exc}") from exc

    log.info("app.startup_complete")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("app.shutdown_begin")

    active = state.registry.active_meetings()
    if active:
        log.warning("app.shutdown_with_live_meetings", meetings=active)

    if state.pool:
        try:
            await state.pool.close()
            log.info("app.db_pool_closed")
        except Exception as exc:
            log.error("app.db_pool_close_failed", error=str(exc))

    log.info("app.shutdown_complete")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Meeting Notes AI",
        description = "Agentic real-time meeting note-taker API",
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Electron apps use custom protocols (app://) — allow all in dev,
    # restrict to your domain in production via ALLOWED_ORIGINS env var.
    allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
    allowed_origins = (
        ["*"] if allowed_origins_raw == "*"
        else [o.strip() for o in allowed_origins_raw.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = allowed_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )
    log.info("app.cors_configured", allowed_origins=allowed_origins)

    # ── Request ID + timing middleware ────────────────────────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start      = time.monotonic()

        # Bind request context so all logs within this request carry these fields
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id = request_id,
            method     = request.method,
            path       = request.url.path,
        )

        log.info("http.request_start")

        response = await call_next(request)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        log.info(
            "http.request_end",
            status_code = response.status_code,
            elapsed_ms  = elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    # ── Global exception handlers ─────────────────────────────────────────────

    @app.exception_handler(MeetingDBError)
    async def db_error_handler(request: Request, exc: MeetingDBError):
        log.error(
            "http.db_error",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            content     = {"error": "database_error", "detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def validation_error_handler(request: Request, exc: ValueError):
        log.warning("http.validation_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            content     = {"error": "validation_error", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        log.error(
            "http.unhandled_exception",
            path       = request.url.path,
            error      = str(exc),
            error_type = type(exc).__name__,
            exc_info   = True,
        )
        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content     = {"error": "internal_server_error", "detail": "An unexpected error occurred"},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(meetings_router)

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "meeting-notes-ai",
            "version": "1.0.0",
            "docs":    "/docs",
        }

    return app


# ── App instance (used by uvicorn) ────────────────────────────────────────────
app = create_app()