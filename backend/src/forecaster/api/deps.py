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
) -> tuple[pd.DataFrame, str]:
    """Fetch bars from the hot tier, falling back to the cold lake.

    Returns ``(frame, tier)``. This is the single place the two-tier split is
    resolved -- routers never need to know which store answered.
    """
    frame = await ohlcv.get_frame(symbol, start=start, end=end, adjusted=adjusted)
    if not frame.empty:
        return frame, "hot"

    frame = lake_query.read_bars(symbol, start=start, end=end, adjusted=adjusted)
    if not frame.empty:
        return frame, "cold"

    raise InstrumentNotFoundError(
        f"No price history for {symbol.upper()}. Ingest it first.", symbol=symbol.upper()
    )
