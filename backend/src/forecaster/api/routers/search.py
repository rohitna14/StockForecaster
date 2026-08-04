"""Search, resolution, and company enrichment endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from forecaster.api.deps import OHLCVRepoDep, load_bars
from forecaster.exceptions import InstrumentNotFoundError
from forecaster.ingestion import profile as profile_provider
from forecaster.ingestion.on_demand import get_ingestor
from forecaster.logging import get_logger
from forecaster.search import service as search_service

log = get_logger(__name__)
router = APIRouter(tags=["search"])


@router.get("/search")
async def search(
    q: Annotated[str, Query(description="Ticker, company name, nickname, or typo.")],
    limit: Annotated[int, Query(ge=1, le=25)] = 8,
) -> dict[str, Any]:
    """Ranked matches for a free-text query.

    Handles tickers (`AAPL`), full names (`Apple Inc`), nicknames (`google`,
    `facebook`), partials (`mic`), and typos (`aple`, `teslla`). The ``best``
    field is what the Enter key should navigate to.
    """
    return await search_service.search_payload(q, limit=limit)


@router.get("/resolve")
async def resolve(
    q: Annotated[str, Query(description="Free-text query to resolve to one symbol.")],
) -> dict[str, Any]:
    """Resolve free text to a single best symbol.

    Powers pressing Enter without picking from the dropdown, and lets
    ``/s/google`` work as a URL.
    """
    match = await search_service.resolve(q)
    if match is None:
        raise InstrumentNotFoundError(
            f"Nothing in the catalog matches {q!r}.", query=q
        )
    return match.as_dict()


@router.get("/instruments/{symbol}/profile")
async def get_profile(symbol: str) -> dict[str, Any]:
    """Company fundamentals: valuation, size, sector, analyst view, calendar.

    Best effort -- an ETF has no P/E, many foreign listings have no employee
    count. Missing fields come back as null rather than failing the request.
    """
    return await profile_provider.get_profile(symbol.upper())


@router.get("/instruments/{symbol}/quote")
async def get_quote(symbol: str, ohlcv: OHLCVRepoDep) -> dict[str, Any]:
    """Freshest available price.

    **Delayed, not real-time** (~15 minutes on this data source). Real-time
    ticks require a paid feed; the response says so explicitly rather than
    implying otherwise. Falls back to the last stored close if the live quote
    endpoint is unavailable.
    """
    quote = await profile_provider.get_quote(symbol.upper())

    if quote.get("price") is None:
        try:
            frame, tier = await load_bars(symbol, ohlcv)
            if len(frame) >= 2:
                last = float(frame["close"].iloc[-1])
                previous = float(frame["close"].iloc[-2])
                quote.update(
                    {
                        "price": last,
                        "previous_close": previous,
                        "change": round(last - previous, 4),
                        "change_percent": round(last / previous - 1.0, 6),
                        "as_of": str(frame.index[-1].date()),
                        "source": f"stored close ({tier} tier)",
                    }
                )
        except InstrumentNotFoundError:
            pass

    return quote


@router.get("/instruments/{symbol}/news")
async def get_news(
    symbol: str,
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> dict[str, Any]:
    """Recent headlines. Coverage varies considerably by ticker."""
    items = await profile_provider.get_news(symbol.upper(), limit=limit)
    return {"symbol": symbol.upper(), "count": len(items), "items": items}


@router.post("/instruments/{symbol}/ingest")
async def ingest_symbol(
    symbol: str,
    years: Annotated[int, Query(ge=1, le=25)] = 5,
) -> dict[str, Any]:
    """Force ingestion of a symbol.

    Normally unnecessary -- reading any symbol endpoint triggers ingestion
    automatically. Exposed for warming a ticker before a demo, and for
    re-fetching after a provider outage.
    """
    ingestor = get_ingestor()
    ingestor.forget(symbol)
    outcome = await ingestor.ensure(symbol.upper(), years=years)
    return {
        "symbol": outcome.symbol,
        "ingested": outcome.ingested,
        "rows": outcome.rows,
        "reason": outcome.reason,
    }


@router.get("/search/index")
async def index_status() -> dict[str, Any]:
    """Search index diagnostics."""
    from forecaster.search.index import get_index

    size = await search_service.ensure_index()
    return {"entries": size, "stale": get_index().is_stale}
