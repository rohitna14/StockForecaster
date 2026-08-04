"""Provider abstraction.

Every price source implements :class:`PriceProvider` and returns the *same*
normalised frame, so the router can fail over between them without any caller
knowing which one answered.

The normalised contract (enforced by :func:`normalise_frame`):

* index: ``DatetimeIndex`` named ``ts``, tz-naive, sorted ascending, unique
* columns: exactly ``open, high, low, close, adj_close, volume``
* dtypes: float64 prices, int64 volume
* no NaNs, no non-positive closes
"""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass

import pandas as pd

from forecaster.db.repositories.ohlcv import OHLCV_COLUMNS
from forecaster.exceptions import DataValidationError
from forecaster.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What a provider can actually do, so the router can route intelligently."""

    daily: bool = True
    intraday: bool = False
    adjusted: bool = False
    corporate_actions: bool = False
    fundamentals: bool = False
    news: bool = False
    #: Approximate sustainable request rate. Used to size the token bucket.
    requests_per_minute: float = 60.0
    #: Hard daily cap, if the provider imposes one (Alpha Vantage: 25).
    requests_per_day: int | None = None
    #: How far back the free tier reaches.
    max_history_years: int | None = None


class PriceProvider(abc.ABC):
    """Base class for daily OHLCV sources."""

    #: Stable identifier persisted in ``ohlcv_daily.source``.
    name: str
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abc.abstractmethod
    async def fetch_daily(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        """Return normalised daily bars in ``[start, end]``.

        Implementations should raise :class:`~forecaster.exceptions.ProviderError`
        (or a subclass) on failure rather than returning an empty frame, so the
        router can distinguish "this provider is broken" from "this symbol has
        no data".
        """

    def is_configured(self) -> bool:
        """False when a required API key is absent; the router then skips it."""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def normalise_frame(frame: pd.DataFrame, *, symbol: str, provider: str) -> pd.DataFrame:
    """Coerce a provider's raw frame into the canonical contract.

    Raises:
        DataValidationError: if the frame cannot be made to satisfy the
            contract -- which means the provider returned something we should
            not silently persist.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("ts")

    out = frame.copy()

    # Flatten yfinance's MultiIndex columns when a single ticker is requested.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    aliases = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
        "vol": "volume",
    }
    out = out.rename(columns=aliases)

    if "adj_close" not in out.columns and "close" in out.columns:
        # Provider gives no adjusted series (Stooq). Its prices are already
        # split-adjusted, so adj_close == close is correct, not a fudge.
        out["adj_close"] = out["close"]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataValidationError(
            f"{provider} returned {symbol} without columns {missing}",
            symbol=symbol,
            provider=provider,
            got=sorted(out.columns),
        )

    out = out[OHLCV_COLUMNS]

    # Index -> tz-naive DatetimeIndex named 'ts'
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    out.index = idx.tz_convert(None) if idx.tz is not None else idx
    out.index.name = "ts"
    out = out[out.index.notna()]
    out = out.normalize() if hasattr(out, "normalize") else out
    out.index = out.index.normalize()

    for col in ("open", "high", "low", "close", "adj_close"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")

    before = len(out)
    out = out.dropna(subset=["open", "high", "low", "close", "adj_close"])
    out = out[(out[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1)]
    out = out[~out.index.duplicated(keep="last")].sort_index()

    dropped = before - len(out)
    if dropped:
        log.debug("normalise_dropped_rows", symbol=symbol, provider=provider, dropped=dropped)

    return out
