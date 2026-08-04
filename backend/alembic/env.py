"""Alembic environment.

The database URL comes from application settings rather than alembic.ini, so
there is exactly one place a connection string is configured. Autogenerate sees
the same metadata the app uses, so a drift between ORM and migrations shows up
as a non-empty diff.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from forecaster.config import get_settings
from forecaster.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    # SQLite creates internal tables we never want in a migration.
    if type_ == "table" and name.startswith("sqlite_"):
        return False
    return True


def _render_item(type_, obj, autogen_context):  # noqa: ANN001
    """Render project-specific column types as references to their definitions.

    Left to itself, autogenerate expands a dialect-variant type into its
    constructor call -- e.g. ``JSON().with_variant(postgresql.JSONB(
    astext_type=String()), 'postgresql')`` -- and emits it without the imports
    those names need, so the migration fails at import time.

    Emitting ``forecaster.db.types.JSONColumn`` instead is both importable and
    correct by construction: the migration and the ORM then reference the *same*
    type object, so they cannot drift.
    """
    if type_ != "type":
        return False

    import sqlalchemy as sa_

    from forecaster.db import types as ft

    autogen_context.imports.add("import forecaster.db.types")

    if isinstance(obj, ft.TZDateTime):
        return "forecaster.db.types.TZDateTime()"

    has_variant = bool(getattr(obj, "_variant_mapping", None))
    if has_variant:
        if isinstance(obj, sa_.JSON):
            return "forecaster.db.types.JSONColumn"
        if isinstance(obj, sa_.BigInteger):
            return "forecaster.db.types.BigIntPK"
        if isinstance(obj, sa_.Numeric):
            return "forecaster.db.types.MoneyColumn"

    if isinstance(obj, sa_.Uuid):
        return "forecaster.db.types.UUIDColumn"

    return False


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        render_item=_render_item,
        # Needed for ALTER on SQLite, which has no native ALTER COLUMN.
        render_as_batch=settings.is_sqlite,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
        render_item=_render_item,
        render_as_batch=settings.is_sqlite,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
