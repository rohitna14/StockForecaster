"""Daily price-bar queries.

Returns pandas DataFrames rather than ORM objects: every downstream consumer
(features, models, backtest) is vectorised, and materialising 1,260 ORM
instances per symbol just to immediately discard them is pure waste.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select

from forecaster.db.models import Instrument, OHLCVDaily
from forecaster.db.repositories.base import Repository

#: Canonical column order for every price frame in the codebase.
OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class OHLCVRepository(Repository):
    async def upsert_bars(self, instrument_id: int, frame: pd.DataFrame, source: str) -> int:
        """Persist a frame of bars.

        ``frame`` must be indexed by date and carry the columns in
        :data:`OHLCV_COLUMNS`. Re-ingesting an overlapping window is safe and
        idempotent -- later sources overwrite earlier ones for the same bar.
        """
        if frame.empty:
            return 0

        missing = set(OHLCV_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"frame missing required columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for ts, row in frame.iterrows():
            ts_date = ts.date() if isinstance(ts, pd.Timestamp) else ts
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "ts": ts_date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "adj_close": float(row["adj_close"]),
                    "volume": int(row["volume"]),
                    "source": source,
                }
            )

        return await self.upsert(
            OHLCVDaily,
            rows,
            conflict_cols=["instrument_id", "ts"],
            update_cols=["open", "high", "low", "close", "adj_close", "volume", "source"],
        )

    async def get_frame(
        self,
        symbol: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Load bars for one symbol as a date-indexed DataFrame.

        Args:
            adjusted: when True (the default, and what models must use), the
                whole OHLC bar is back-adjusted by ``adj_close / close`` and
                volume is scaled by its reciprocal. Returning a raw ``open``
                next to an adjusted ``close`` would make every overnight-gap
                and intraday-range feature wrong across any split.
        """
        stmt = (
            select(
                OHLCVDaily.ts,
                OHLCVDaily.open,
                OHLCVDaily.high,
                OHLCVDaily.low,
                OHLCVDaily.close,
                OHLCVDaily.adj_close,
                OHLCVDaily.volume,
            )
            .join(Instrument, Instrument.id == OHLCVDaily.instrument_id)
            .where(Instrument.symbol == symbol.upper())
            .order_by(OHLCVDaily.ts)
        )
        if start:
            stmt = stmt.where(OHLCVDaily.ts >= start)
        if end:
            stmt = stmt.where(OHLCVDaily.ts <= end)

        rows = (await self.session.execute(stmt)).all()
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("ts")

        frame = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS])
        frame["ts"] = pd.to_datetime(frame["ts"])
        frame = frame.set_index("ts").sort_index()

        if adjusted:
            frame = apply_adjustment(frame)
        return frame

    async def get_panel(
        self,
        symbols: Sequence[str],
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Wide panel of one field across many symbols (dates x symbols).

        Used for cross-sectional features and portfolio backtests.
        """
        if not symbols:
            return pd.DataFrame()

        col = getattr(OHLCVDaily, field)
        stmt = (
            select(Instrument.symbol, OHLCVDaily.ts, col)
            .join(Instrument, Instrument.id == OHLCVDaily.instrument_id)
            .where(Instrument.symbol.in_([s.upper() for s in symbols]))
            .order_by(OHLCVDaily.ts)
        )
        if start:
            stmt = stmt.where(OHLCVDaily.ts >= start)
        if end:
            stmt = stmt.where(OHLCVDaily.ts <= end)

        rows = (await self.session.execute(stmt)).all()
        if not rows:
            return pd.DataFrame()

        long = pd.DataFrame(rows, columns=["symbol", "ts", field])
        long["ts"] = pd.to_datetime(long["ts"])
        return long.pivot(index="ts", columns="symbol", values=field).sort_index()

    async def latest_date(self, instrument_id: int) -> dt.date | None:
        stmt = select(func.max(OHLCVDaily.ts)).where(OHLCVDaily.instrument_id == instrument_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def coverage(self, instrument_id: int) -> tuple[dt.date | None, dt.date | None, int]:
        """(first_date, last_date, bar_count) in a single round-trip."""
        stmt = select(
            func.min(OHLCVDaily.ts), func.max(OHLCVDaily.ts), func.count()
        ).where(OHLCVDaily.instrument_id == instrument_id)
        first, last, count = (await self.session.execute(stmt)).one()
        return first, last, int(count or 0)

    async def delete_symbol(self, instrument_id: int) -> int:
        """Evict a symbol's bars (used when demoting hot -> cold)."""
        result = await self.session.execute(
            delete(OHLCVDaily).where(OHLCVDaily.instrument_id == instrument_id)
        )
        return int(result.rowcount or 0)

    async def total_bars(self) -> int:
        return int((await self.session.execute(select(func.count()).select_from(OHLCVDaily))).scalar_one())


def apply_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust a raw OHLCV frame for splits and dividends.

    factor = adj_close / close. Prices are multiplied by it, volume divided --
    a 2:1 split halves the price and doubles the share count, and the adjusted
    series must undo both consistently.
    """
    out = frame.copy()
    factor = out["adj_close"] / out["close"]
    for col in ("open", "high", "low"):
        out[col] = out[col] * factor
    out["close"] = out["adj_close"]
    out["volume"] = (out["volume"] / factor).round().astype("int64")
    return out
