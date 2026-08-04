"""Cross-dialect column types.

Local development runs on SQLite (no Docker required); production runs on
Postgres. Rather than maintaining two schemas, we declare each type once with a
Postgres variant so SQLAlchemy emits the right DDL per dialect:

===============  ==================  ==========================
Alias            Postgres            SQLite
===============  ==================  ==========================
``JSONColumn``   ``JSONB``           ``JSON`` (TEXT under the hood)
``UUIDColumn``   native ``UUID``     ``CHAR(32)``
``TZDateTime``   ``TIMESTAMPTZ``     ``DATETIME`` (UTC-normalised)
===============  ==================  ==========================
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import BigInteger, DateTime, Dialect, Float, Integer, Numeric, String, Uuid
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import JSON, TypeDecorator

# JSONB on Postgres (indexable, binary) -- plain JSON elsewhere.
JSONColumn = JSON().with_variant(postgresql.JSONB(astext_type=String()), "postgresql")

# Auto-incrementing primary key.
#
# SQLite only auto-assigns rowids for a column declared exactly INTEGER PRIMARY
# KEY -- BIGINT PRIMARY KEY silently loses that behaviour and every insert then
# fails with "NOT NULL constraint failed". Postgres keeps BIGINT (BIGSERIAL) so
# we do not cap the table at 2^31 rows.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")

# Native uuid on Postgres, CHAR(32) elsewhere.
UUIDColumn = Uuid(as_uuid=True, native_uuid=True)

# Money: exact on Postgres, float on SQLite (which has no DECIMAL affinity).
MoneyColumn = Numeric(20, 4).with_variant(Float(), "sqlite")


class TZDateTime(TypeDecorator[dt.datetime]):
    """Timezone-aware datetime that survives SQLite.

    SQLite discards tzinfo, which silently turns aware datetimes into naive
    ones on read and makes comparisons wrong. This normalises to UTC on write
    and re-attaches UTC on read so application code always sees aware values.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: dt.datetime | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"Naive datetime rejected: {value!r}. Attach a timezone.")
        return value.astimezone(dt.UTC)

    def process_result_value(self, value: dt.datetime | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value.astimezone(dt.UTC)


def utcnow() -> dt.datetime:
    """Timezone-aware 'now'. Never use ``datetime.utcnow()`` -- it returns naive."""
    return dt.datetime.now(dt.UTC)
