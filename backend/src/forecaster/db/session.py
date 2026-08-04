"""Async engine and session management.

One engine per process, sessions per unit-of-work. The FastAPI layer injects
``get_session`` as a dependency; the CLI and jobs use the ``session_scope``
context manager.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from forecaster.config import get_settings
from forecaster.db.models import Base
from forecaster.logging import get_logger

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _engine_kwargs(url: str, echo: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        # SQLite + async: a shared pool causes cross-task locking grief.
        kwargs["poolclass"] = NullPool
        kwargs["connect_args"] = {"timeout": 30}
    else:
        kwargs |= {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True,  # Neon autosuspends; stale conns are expected
            "pool_recycle": 300,
        }
    return kwargs


def get_engine() -> AsyncEngine:
    """Lazily build the process-wide async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_directories()
        _engine = create_async_engine(
            settings.database_url, **_engine_kwargs(settings.database_url, settings.database_echo)
        )

        if settings.is_sqlite:

            @event.listens_for(_engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
                cur = dbapi_conn.cursor()
                # FKs are OFF by default in SQLite -- our ON DELETE CASCADE
                # rules are inert without this.
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA cache_size=-64000")
                cur.close()

        log.debug("engine_created", dialect=_engine.dialect.name)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commits on success, rolls back on any exception."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_scope() as session:
        yield session


async def create_all() -> None:
    """Create the schema directly from metadata.

    Used for local bootstrap and tests. Production schema changes go through
    Alembic migrations, not this.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("schema_created", tables=len(Base.metadata.tables))


async def drop_all() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def healthcheck() -> bool:
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 -- healthcheck must never raise
        log.warning("db_healthcheck_failed", error=str(exc))
        return False


async def dispose_engine() -> None:
    """Tear down the engine (app shutdown, test teardown)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
