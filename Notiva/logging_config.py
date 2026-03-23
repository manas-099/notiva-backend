"""
logging_config.py — Centralized structlog configuration

"""

import logging
import os
import sys

import structlog


def configure_logging(log_level: str | None = None) -> None:
    level_name = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    level      = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "console")

    # ── 1. Configure stdlib logging FIRST ────────────────────────────────────
    # Must happen before structlog.configure() because LoggerFactory calls
    # logging.getLogger() internally.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # ── 2. Build processor chain ──────────────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,   # works because LoggerFactory gives stdlib loggers
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if log_format == "json":
        final_processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    else:
        final_processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # ── 3. Configure structlog ────────────────────────────────────────────────
    structlog.configure(
        processors=final_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),  # ← was PrintLoggerFactory()
        cache_logger_on_first_use=True,
    )