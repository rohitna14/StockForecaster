"""Instrument search, price history, indicators, risk."""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from forecaster.api.deps import (
    InstrumentRepoDep,
    OHLCVRepoDep,
    PaginationDep,
    load_bars,
)
from forecaster.api.schemas.common import (
    Bar,
    IndicatorResponse,
    InstrumentSummary,
    OHLCVResponse,
    Page,
    RiskResponse,
)
from forecaster.db.models import Tier
from forecaster.exceptions import InsufficientDataError, UnknownFeatureError
from forecaster.lake import query as lake_query
from forecaster.lake import writer as lake_writer

router = APIRouter(prefix="/instruments", tags=["instruments"])

TRADING_DAYS = 252


@router.get("", response_model=Page[InstrumentSummary])
async def search_instruments(
    repo: InstrumentRepoDep,
    pagination: PaginationDep,
    q: Annotated[str | None, Query(description="Symbol or name substring.")] = None,
    sector: str | None = None,
    tier: Tier | None = None,
) -> Page[InstrumentSummary]:
    """Typeahead search. Exact symbol matches rank first."""
    rows = await repo.search(
        q, sector=sector, tier=tier, limit=pagination.limit + 1, offset=pagination.offset
    )
    has_more = len(rows) > pagination.limit
    return Page[InstrumentSummary](
        items=[InstrumentSummary.model_validate(r) for r in rows[: pagination.limit]],
        limit=pagination.limit,
        offset=pagination.offset,
        has_more=has_more,
    )


@router.get("/sectors", response_model=list[str])
async def list_sectors(repo: InstrumentRepoDep) -> list[str]:
    return await repo.sectors()


@router.get("/sparklines")
async def get_sparklines(
    ohlcv: OHLCVRepoDep,
    symbols: Annotated[str, Query(description="Comma-separated symbols.")],
    days: Annotated[int, Query(ge=5, le=750)] = 90,
) -> dict[str, Any]:
    """Compact recent-price series for many symbols in one round-trip.

    A grid of ticker cards each fetching its own history would fire a dozen
    requests on every page load. This returns normalised series (first value =
    100) so the frontend can draw a shape without knowing the price scale.
    """
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()][:40]
    if not wanted:
        return {"series": {}}

    cutoff = dt.date.today() - dt.timedelta(days=int(days * 1.6))
    panel = await ohlcv.get_panel(wanted, start=cutoff, field="adj_close")

    out: dict[str, Any] = {}
    for symbol in wanted:
        series = None
        if not panel.empty and symbol in panel.columns:
            series = panel[symbol].dropna().tail(days)
        if series is None or len(series) < 2:
            # Fall back to the cold lake for symbols not promoted to hot.
            frame = lake_query.read_bars(symbol, start=cutoff)
            if frame.empty:
                continue
            series = frame["close"].tail(days)

        values = series.to_numpy(dtype="float64")
        base = float(values[0])
        if base <= 0:
            continue

        out[symbol] = {
            "points": [round(float(v) / base * 100.0, 3) for v in values],
            "change_pct": round(float(values[-1] / base - 1.0), 6),
            "last": round(float(values[-1]), 4),
            "n": len(values),
        }

    return {"series": out, "days": days}


@router.get("/{symbol}", response_model=InstrumentSummary)
async def get_instrument(symbol: str, repo: InstrumentRepoDep) -> InstrumentSummary:
    instrument = await repo.require_by_symbol(symbol)
    summary = InstrumentSummary.model_validate(instrument)

    # Coverage may live in the lake even when the symbol is not promoted.
    if summary.first_date is None:
        coverage = lake_writer.coverage(symbol)
        if coverage:
            summary.first_date, summary.last_date, _ = coverage
    return summary


@router.get("/{symbol}/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str,
    ohlcv: OHLCVRepoDep,
    start: dt.date | None = None,
    end: dt.date | None = None,
    adjusted: Annotated[bool, Query(description="Split/dividend adjusted.")] = True,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 2000,
) -> OHLCVResponse:
    """Daily bars. Serves from the hot tier, falling back to the Parquet lake."""
    frame, tier = await load_bars(symbol, ohlcv, start=start, end=end, adjusted=adjusted)
    frame = frame.tail(limit)

    bars = [
        Bar(
            ts=ts.date() if hasattr(ts, "date") else ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            adj_close=float(row["adj_close"]),
            volume=int(row["volume"]),
        )
        for ts, row in frame.iterrows()
    ]
    return OHLCVResponse(
        symbol=symbol.upper(),
        adjusted=adjusted,
        source_tier=tier,
        bars=bars,
        count=len(bars),
    )


@router.get("/{symbol}/indicators", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str,
    ohlcv: OHLCVRepoDep,
    features: Annotated[str | None, Query(description="Comma-separated feature names.")] = None,
    feature_set: Annotated[str, Query(description="minimal|core|full|all")] = "core",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> IndicatorResponse:
    """Computed technical indicators, aligned to the price index."""
    from forecaster import features as F

    frame, _ = await load_bars(symbol, ohlcv)

    if features:
        names = [n.strip() for n in features.split(",") if n.strip()]
        unknown = [n for n in names if n not in {s.name for s in F.list_features()}]
        if unknown:
            raise UnknownFeatureError(f"Unknown features: {unknown}", unknown=unknown)
    else:
        names = F.feature_set(feature_set)

    matrix = F.build(frame, names).tail(limit)
    return IndicatorResponse(
        symbol=symbol.upper(),
        features=names,
        index=[ts.date() for ts in matrix.index],
        values={
            column: [None if not np.isfinite(v) else float(v) for v in matrix[column]]
            for column in matrix.columns
        },
    )


@router.get("/{symbol}/risk", response_model=RiskResponse)
async def get_risk(
    symbol: str,
    ohlcv: OHLCVRepoDep,
    lookback_days: Annotated[int, Query(ge=60, le=5000)] = 756,
) -> RiskResponse:
    """Realised risk statistics over a trailing window."""
    from forecaster.validation import metrics as M

    frame, _ = await load_bars(symbol, ohlcv)
    returns = frame["close"].pct_change().dropna().tail(lookback_days)

    if len(returns) < 30:
        raise InsufficientDataError(
            f"Need at least 30 return observations for {symbol.upper()}, got {len(returns)}",
            symbol=symbol.upper(),
        )

    values = returns.to_numpy()
    var_95 = float(np.percentile(values, 5))
    tail = values[values <= var_95]

    return RiskResponse(
        symbol=symbol.upper(),
        annual_volatility=float(values.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        sharpe=M.sharpe_ratio(values),
        sortino=M.sortino_ratio(values),
        max_drawdown=M.max_drawdown(values),
        var_95=var_95,
        cvar_95=float(tail.mean()) if len(tail) else None,
        n_observations=len(values),
    )


@router.get("/{symbol}/summary")
async def get_summary(symbol: str, ohlcv: OHLCVRepoDep) -> dict[str, object]:
    """Compact snapshot for the overview page: last price, change, range."""
    frame, tier = await load_bars(symbol, ohlcv)
    if len(frame) < 2:
        raise InsufficientDataError(f"Not enough bars for {symbol.upper()}")

    close = frame["close"]
    last, previous = float(close.iloc[-1]), float(close.iloc[-2])
    window = close.tail(TRADING_DAYS)

    def _change(bars: int) -> float | None:
        if len(close) <= bars:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - bars] - 1)

    return {
        "symbol": symbol.upper(),
        "source_tier": tier,
        "as_of": str(pd.Timestamp(frame.index[-1]).date()),
        "last_close": last,
        "change_1d": last / previous - 1,
        "change_1w": _change(5),
        "change_1m": _change(21),
        "change_1y": _change(TRADING_DAYS),
        "volume": int(frame["volume"].iloc[-1]),
        "avg_volume_20d": float(frame["volume"].tail(20).mean()),
        "range_52w_high": float(window.max()),
        "range_52w_low": float(window.min()),
        "n_bars": len(frame),
        "first_date": str(pd.Timestamp(frame.index[0]).date()),
    }
