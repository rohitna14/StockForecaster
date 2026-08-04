"""Search service: loads the catalog into the index and answers queries."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from forecaster.db.models import Instrument, Tier
from forecaster.db.session import session_scope
from forecaster.logging import get_logger
from forecaster.search.index import Match, get_index

log = get_logger(__name__)


async def ensure_index(force: bool = False) -> int:
    """Load the instrument catalog into the in-memory index if stale."""
    index = get_index()
    if not force and not index.is_stale:
        return index.size

    async with session_scope() as session:
        stmt = select(
            Instrument.symbol,
            Instrument.name,
            Instrument.sector,
            Instrument.market_cap,
            Instrument.tier,
            Instrument.first_date,
        ).where(Instrument.is_active.is_(True))
        rows = (await session.execute(stmt)).all()

    index.build(
        [
            {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "market_cap": market_cap,
                "has_data": tier == Tier.HOT or first_date is not None,
            }
            for symbol, name, sector, market_cap, tier, first_date in rows
        ]
    )
    return index.size


async def search(query: str, limit: int = 10) -> list[Match]:
    await ensure_index()
    return get_index().search(query, limit=limit)


async def resolve(query: str) -> Match | None:
    """Best single match -- what Enter in the search bar navigates to."""
    await ensure_index()
    return get_index().best(query)


async def search_payload(query: str, limit: int = 10) -> dict[str, Any]:
    results = await search(query, limit=limit)
    return {
        "query": query,
        "results": [m.as_dict() for m in results],
        "best": results[0].as_dict() if results else None,
        "count": len(results),
    }
