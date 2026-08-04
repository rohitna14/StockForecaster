"""Instrument (ticker) queries."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, or_, select, update

from forecaster.db.models import Instrument, Tier
from forecaster.db.repositories.base import Repository
from forecaster.exceptions import InstrumentNotFoundError


class InstrumentRepository(Repository):
    async def get_by_symbol(self, symbol: str) -> Instrument | None:
        stmt = select(Instrument).where(Instrument.symbol == symbol.upper())
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def require_by_symbol(self, symbol: str) -> Instrument:
        instrument = await self.get_by_symbol(symbol)
        if instrument is None:
            raise InstrumentNotFoundError(f"Unknown symbol {symbol!r}", symbol=symbol)
        return instrument

    async def get_by_symbols(self, symbols: Sequence[str]) -> dict[str, Instrument]:
        if not symbols:
            return {}
        upper = [s.upper() for s in symbols]
        stmt = select(Instrument).where(Instrument.symbol.in_(upper))
        rows = (await self.session.execute(stmt)).scalars().all()
        return {row.symbol: row for row in rows}

    async def search(
        self,
        query: str | None = None,
        *,
        sector: str | None = None,
        tier: Tier | None = None,
        active_only: bool = True,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Instrument]:
        """Typeahead search over symbol and name.

        Exact symbol matches rank first, then prefix matches, then substring --
        so typing "A" surfaces ``A`` before ``AGILENT`` before ``TESLA``.
        """
        stmt = select(Instrument)
        if active_only:
            stmt = stmt.where(Instrument.is_active.is_(True))
        if sector:
            stmt = stmt.where(Instrument.sector == sector)
        if tier:
            stmt = stmt.where(Instrument.tier == tier)

        if query:
            q = query.strip().upper()
            like = f"%{q}%"
            stmt = stmt.where(
                or_(Instrument.symbol.ilike(like), Instrument.name.ilike(f"%{query.strip()}%"))
            )
            stmt = stmt.order_by(
                (Instrument.symbol == q).desc(),
                Instrument.symbol.ilike(f"{q}%").desc(),
                func.length(Instrument.symbol),
                Instrument.symbol,
            )
        else:
            # No query: show the most liquid names first.
            stmt = stmt.order_by(Instrument.market_cap.desc().nullslast(), Instrument.symbol)

        stmt = stmt.limit(limit).offset(offset)
        return list((await self.session.execute(stmt)).scalars().all())

    async def list_symbols(
        self, *, tier: Tier | None = None, active_only: bool = True
    ) -> list[str]:
        stmt = select(Instrument.symbol)
        if tier:
            stmt = stmt.where(Instrument.tier == tier)
        if active_only:
            stmt = stmt.where(Instrument.is_active.is_(True))
        return list((await self.session.execute(stmt.order_by(Instrument.symbol))).scalars().all())

    async def upsert_many(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert or refresh instrument metadata.

        ``tier`` is deliberately excluded from the update set: re-running a
        metadata sync must not silently demote symbols already promoted to hot.
        """
        normalised = [{**r, "symbol": str(r["symbol"]).upper()} for r in rows]
        return await self.upsert(
            Instrument,
            normalised,
            conflict_cols=["symbol"],
            update_cols=[
                "name",
                "exchange",
                "asset_type",
                "sector",
                "industry",
                "country",
                "currency",
                "market_cap",
                "ipo_year",
                "is_active",
            ],
        )

    async def set_tier(self, symbol: str, tier: Tier) -> None:
        await self.session.execute(
            update(Instrument).where(Instrument.symbol == symbol.upper()).values(tier=tier)
        )

    async def update_coverage(
        self, instrument_id: int, first_date: dt.date, last_date: dt.date
    ) -> None:
        """Record the span of price history we actually hold."""
        await self.session.execute(
            update(Instrument)
            .where(Instrument.id == instrument_id)
            .values(first_date=first_date, last_date=last_date)
        )

    async def count(self, *, tier: Tier | None = None) -> int:
        stmt = select(func.count()).select_from(Instrument)
        if tier:
            stmt = stmt.where(Instrument.tier == tier)
        return int((await self.session.execute(stmt)).scalar_one())

    async def sectors(self) -> list[str]:
        stmt = (
            select(Instrument.sector)
            .where(Instrument.sector.is_not(None))
            .distinct()
            .order_by(Instrument.sector)
        )
        return list((await self.session.execute(stmt)).scalars().all())
