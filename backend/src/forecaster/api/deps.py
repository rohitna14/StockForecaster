"""FastAPI dependencies and shared service accessors."""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from typing import Annotated

import pandas as pd
from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from forecaster.config import Settings, get_settings
from forecaster.db.repositories.instruments import InstrumentRepository
from forecaster.db.repositories.ohlcv import OHLCVRepository
from forecaster.db.repositories.runs import RunRepository
from forecaster.db.session import get_session
from forecaster.exceptions import InstrumentNotFoundError
from forecaster.ingestion.on_demand import get_ingestor
from forecaster.ingestion.router import ProviderRouter
from forecaster.lake import query as lake_query
from forecaster.logging import get_logger

log = get_logger(__name__)

_router: ProviderRouter | None = None


def get_provider_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router


async def close_provider_router() -> None:
    global _router
    if _router is not None:
        await _router.aclose()
    _router = None
    await get_ingestor().aclose()


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_instrument_repo(session: SessionDep) -> AsyncIterator[InstrumentRepository]:
    yield InstrumentRepository(session)


async def get_ohlcv_repo(session: SessionDep) -> AsyncIterator[OHLCVRepository]:
    yield OHLCVRepository(session)


async def get_run_repo(session: SessionDep) -> AsyncIterator[RunRepository]:
    yield RunRepository(session)


InstrumentRepoDep = Annotated[InstrumentRepository, Depends(get_instrument_repo)]
OHLCVRepoDep = Annotated[OHLCVRepository, Depends(get_ohlcv_repo)]
RunRepoDep = Annotated[RunRepository, Depends(get_run_repo)]


class PaginationParams:
    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> None:
        self.limit = limit
        self.offset = offset


PaginationDep = Annotated[PaginationParams, Depends()]


async def load_bars(
    symbol: str,
    ohlcv: OHLCVRepository,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    adjusted: bool = True,
    allow_fetch: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Fetch bars, ingesting the symbol on demand if we do not have it yet.

    Returns ``(frame, tier)``. Every symbol endpoint funnels through here, so
    this is the one place that resolves the hot/cold split *and* the one place
    that decides to go and get data we are missing. A user should never be told
    "not ingested" for a real listed company -- that is a detail of our storage,
    not a fact about the market.

    Resolution order: hot tier -> cold lake -> fetch from providers -> hot tier.
    """
    frame = await ohlcv.get_frame(symbol, start=start, end=end, adjusted=adjusted)
    if not frame.empty:
        return frame, "hot"

    frame = lake_query.read_bars(symbol, start=start, end=end, adjusted=adjusted)
    if not frame.empty:
        return frame, "cold"

    if allow_fetch:
        outcome = await get_ingestor().ensure(symbol)
        if outcome.succeeded:
            frame = await ohlcv.get_frame(symbol, start=start, end=end, adjusted=adjusted)
            if not frame.empty:
                return frame, "fetched"
            frame = lake_query.read_bars(symbol, start=start, end=end, adjusted=adjusted)
            if not frame.empty:
                return frame, "fetched"

    raise InstrumentNotFoundError(
        f"No market history is available for {symbol.upper()} from any data "
        f"provider. It may be delisted, an invalid symbol, or not covered by "
        f"the free data sources this project uses.",
        symbol=symbol.upper(),
    )
