"""Structured logging.

The old codebase reported every failure with ``print()`` inside a bare
``except``, which meant errors were invisible in deployment and unsearchable
locally. Everything here goes through structlog with bound context, renders as
human-readable colour locally and as JSON in production.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

from forecaster.config import LogFormat, get_settings

_configured = False


def configure_logging(*, force: bool = False) -> None:
    """Idempotently configure stdlib logging + structlog."""
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format is LogFormat.JSON:
        renderer: Any = structlog.processors.JSONRenderer()
        shared_processors.append(structlog.processors.format_exc_info)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        shared_processors.append(structlog.processors.ExceptionPrettyPrinter())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # stdlib factory (not PrintLogger): gives every logger a .name for
        # add_logger_name, and routes through standard handlers so library
        # logs and ours land in the same stream.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    # Third-party libraries are chatty and mostly useless at INFO.
    for noisy in ("urllib3", "asyncio", "httpx", "httpcore", "peewee", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """Bind key/values to every log line emitted inside the block.

    >>> with log_context(symbol="AAPL", run_id=run.id):
    ...     ingest(...)   # every line carries symbol + run_id
    """
    bind_contextvars(**kwargs)
    try:
        yield
    finally:
        unbind_contextvars(*kwargs.keys())


__all__ = ["clear_contextvars", "configure_logging", "get_logger", "log_context"]
