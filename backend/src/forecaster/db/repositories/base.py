"""Repository foundations.

Repositories own all SQL. Services and the API talk to repositories, never to
``session.execute`` directly -- that keeps query knowledge in one layer and
makes the dialect differences below invisible to callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from forecaster.db.models import Base

ModelT = TypeVar("ModelT", bound=Base)

#: Rows per INSERT statement. SQLite caps host parameters at 32766; with ~10
#: columns per OHLCV row, 2000 rows stays comfortably under that ceiling while
#: keeping round-trips low.
DEFAULT_CHUNK_SIZE = 2000


class Repository:
    """Base class holding the session and dialect-aware helpers."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @property
    def dialect(self) -> str:
        return self.session.bind.dialect.name  # type: ignore[union-attr]

    async def upsert(
        self,
        table: type[ModelT] | Table,
        rows: Sequence[dict[str, Any]],
        *,
        conflict_cols: Sequence[str],
        update_cols: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> int:
        """Bulk INSERT ... ON CONFLICT DO UPDATE, portable across PG and SQLite.

        Both dialects support upsert but expose it through different
        constructors, so the statement is built per-dialect here rather than at
        every call site.

        Args:
            conflict_cols: columns forming the uniqueness constraint.
            update_cols: columns to overwrite on conflict. ``None`` updates
                every non-conflict column; an empty sequence makes it a
                DO NOTHING (insert-if-absent).

        Returns:
            Number of rows submitted (not necessarily the number changed).
        """
        if not rows:
            return 0

        tbl: Table = table if isinstance(table, Table) else table.__table__
        insert_fn = pg_insert if self.dialect == "postgresql" else sqlite_insert

        if update_cols is None:
            update_cols = [c.name for c in tbl.columns if c.name not in set(conflict_cols)]

        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            stmt = insert_fn(tbl).values(list(chunk))
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(conflict_cols),
                    set_={c: getattr(stmt.excluded, c) for c in update_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))
            await self.session.execute(stmt)
            total += len(chunk)
        return total
