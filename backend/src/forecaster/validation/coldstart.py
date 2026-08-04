"""Forecasting for stocks with too little history to validate on their own.

Chooses the strongest method the available history supports, and labels the
result so a 36-bar IPO is never presented as if it carried the same evidence as
a 10-year-old mega cap.

===============  ==========  ==================================================
Method           Bars        What it means
===============  ==========  ==================================================
walk_forward     >= ~600     Per-stock, 5+ folds. Full confidence.
adaptive         >= ~150     Per-stock, shrunken windows, 2-3 folds.
transfer         >= 25       Pooled model trained on other stocks. The target's
                             own history is used only to compute features, never
                             to train -- so accuracy is quoted from
                             leave-one-symbol-out validation on stocks the model
                             had never seen.
none             < 25        Not enough bars to compute a single feature row.
===============  ==========  ==================================================

The pooled model is cached: fitting it takes seconds and it does not depend on
the symbol being predicted, so one fit serves every new listing.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from forecaster.exceptions import InsufficientDataError
from forecaster.logging import get_logger
from forecaster.models.pooled import (
    MIN_TARGET_BARS,
    PooledForecaster,
    PooledTrainingData,
    TransferResult,
    build_pool,
    validate_transfer,
)

log = get_logger(__name__)

#: Bars needed for full per-stock walk-forward with the core feature set.
WALK_FORWARD_BARS = 600
#: Bars needed for per-stock walk-forward with shrunken windows.
ADAPTIVE_BARS = 150

#: How long a fitted pooled model is reused before refitting.
POOL_TTL_SECONDS = 3600


class Method(StrEnum):
    WALK_FORWARD = "walk_forward"
    ADAPTIVE = "adaptive"
    TRANSFER = "transfer"
    NONE = "none"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


def choose_method(n_bars: int) -> Method:
    if n_bars >= WALK_FORWARD_BARS:
        return Method.WALK_FORWARD
    if n_bars >= ADAPTIVE_BARS:
        return Method.ADAPTIVE
    if n_bars >= MIN_TARGET_BARS:
        return Method.TRANSFER
    return Method.NONE


def confidence_for(method: Method, n_bars: int, skill_pct: float | None) -> Confidence:
    """Confidence is about *evidence*, not about how good the number looks.

    A high skill score from three folds on eight months of data is weaker
    evidence than a modest one from eight folds on five years, and the label
    reflects that rather than the headline figure.
    """
    if method is Method.NONE:
        return Confidence.VERY_LOW
    if skill_pct is not None and skill_pct <= 0:
        return Confidence.LOW
    if method is Method.WALK_FORWARD:
        return Confidence.HIGH if n_bars >= 1000 else Confidence.MEDIUM
    if method is Method.ADAPTIVE:
        return Confidence.MEDIUM if n_bars >= 300 else Confidence.LOW
    return Confidence.LOW if n_bars >= 60 else Confidence.VERY_LOW


@dataclass
class ColdStartForecast:
    symbol: str
    method: Method
    confidence: Confidence
    prediction: float | None
    lower: float | None
    upper: float | None
    as_of: dt.date | None
    horizon: int
    n_bars: int
    expected_skill_pct: float | None
    basis: str
    transfer: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "method": self.method.value,
            "confidence": self.confidence.value,
            "prediction": self.prediction,
            "lower": self.lower,
            "upper": self.upper,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "horizon": self.horizon,
            "n_bars": self.n_bars,
            "expected_skill_pct": self.expected_skill_pct,
            "basis": self.basis,
            "transfer_validation": self.transfer,
            "caveats": self.caveats,
        }


class _PoolCache:
    """One fitted pooled model, shared across every cold-start request."""

    def __init__(self) -> None:
        self._model: PooledForecaster | None = None
        self._pool: PooledTrainingData | None = None
        self._transfer: TransferResult | None = None
        self._fitted_at: float = 0.0

    @property
    def is_stale(self) -> bool:
        return self._model is None or (time.monotonic() - self._fitted_at) > POOL_TTL_SECONDS

    def set(
        self,
        model: PooledForecaster,
        pool: PooledTrainingData,
        transfer: TransferResult | None,
    ) -> None:
        self._model = model
        self._pool = pool
        self._transfer = transfer
        self._fitted_at = time.monotonic()

    def get(self) -> tuple[PooledForecaster | None, PooledTrainingData | None, TransferResult | None]:
        return self._model, self._pool, self._transfer


_cache = _PoolCache()


async def _load_pool_frames(exclude: str, limit: int = 24) -> dict[str, pd.DataFrame]:
    """Price history for the training pool, excluding the target symbol."""
    from forecaster.db.models import Tier
    from forecaster.db.repositories.instruments import InstrumentRepository
    from forecaster.db.repositories.ohlcv import OHLCVRepository
    from forecaster.db.session import session_scope
    from forecaster.lake import query as lake_query

    frames: dict[str, pd.DataFrame] = {}

    async with session_scope() as session:
        symbols = await InstrumentRepository(session).list_symbols(tier=Tier.HOT)
        ohlcv = OHLCVRepository(session)
        for symbol in symbols:
            if symbol == exclude or len(frames) >= limit:
                continue
            frame = await ohlcv.get_frame(symbol)
            if len(frame) >= 400:
                frames[symbol] = frame

    # Top up from the cold lake if the hot tier is thin.
    if len(frames) < limit:
        for symbol in lake_query.universe_summary().get("symbol", []):
            if symbol == exclude or symbol in frames or len(frames) >= limit:
                continue
            frame = lake_query.read_bars(symbol)
            if len(frame) >= 400:
                frames[symbol] = frame

    return frames


async def ensure_pool(
    exclude: str, *, horizon: int = 5, validate: bool = True
) -> tuple[PooledForecaster, PooledTrainingData, TransferResult | None]:
    """Fit (or reuse) the pooled model."""
    model, pool, transfer = _cache.get()
    if not _cache.is_stale and model is not None and pool is not None:
        return model, pool, transfer

    frames = await _load_pool_frames(exclude)
    pool = build_pool(frames, feature_set="cold_start", horizon=horizon)

    model = PooledForecaster()
    model.fit(pool.X, pool.y)

    transfer = None
    if validate:
        try:
            transfer = validate_transfer(pool, max_symbols=12)
        except Exception as exc:  # noqa: BLE001 -- a forecast is still usable
            log.warning("transfer_validation_failed", error=str(exc))

    _cache.set(model, pool, transfer)
    log.info(
        "pool_fitted",
        symbols=pool.n_symbols,
        rows=len(pool),
        transfer_skill=transfer.rmse_skill_pct if transfer else None,
    )
    return model, pool, transfer


async def forecast_cold_start(
    symbol: str, frame: pd.DataFrame, *, horizon: int = 5
) -> ColdStartForecast:
    """Forecast a symbol whose own history is too short to validate on.

    The target's history is used **only to compute the latest feature row**.
    Nothing about this stock trains the model, which is what makes the
    leave-one-symbol-out skill an honest estimate of its accuracy here.
    """
    from forecaster import features as F

    n_bars = len(frame)
    method = choose_method(n_bars)

    if method is Method.NONE:
        raise InsufficientDataError(
            f"{symbol} has {n_bars} bars. At least {MIN_TARGET_BARS} are needed "
            f"to compute a single feature row -- roughly five weeks of trading.",
            symbol=symbol,
            n_bars=n_bars,
        )

    model, pool, transfer = await ensure_pool(symbol, horizon=horizon)

    names = F.feature_set("cold_start")
    features = F.build(frame, names).replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise InsufficientDataError(
            f"{symbol} has {n_bars} bars but no complete feature row yet.",
            symbol=symbol,
        )

    # Align to the columns the pool was trained on.
    missing = [c for c in pool.feature_names if c not in features.columns]
    if missing:
        raise InsufficientDataError(
            f"Feature mismatch for {symbol}: missing {missing[:5]}", symbol=symbol
        )
    latest = features[pool.feature_names].iloc[[-1]].to_numpy(dtype="float64")

    prediction = float(model.predict(latest)[0])

    # Interval from the spread of pooled residuals -- the honest width for a
    # model that has never seen this company.
    residuals = pool.y - model.predict(pool.X)
    finite = residuals[np.isfinite(residuals)]
    half_width = float(np.quantile(np.abs(finite), 0.8)) if len(finite) else float("nan")

    skill = transfer.rmse_skill_pct if transfer else None
    confidence = confidence_for(method, n_bars, skill)

    caveats = [
        f"Only {n_bars} bars of history exist for {symbol}.",
        "The forecast comes from a model trained on other companies; nothing "
        "about this stock was used for training.",
    ]
    if transfer:
        caveats.append(
            f"Expected accuracy is quoted from leave-one-symbol-out validation "
            f"across {transfer.n_symbols} stocks the model had never seen "
            f"({transfer.rmse_skill_pct:+.1f}% RMSE skill)."
        )
    if n_bars < 60:
        caveats.append(
            "Under three months of history. Treat this as indicative only."
        )

    return ColdStartForecast(
        symbol=symbol,
        method=Method.TRANSFER,
        confidence=confidence,
        prediction=prediction,
        lower=prediction - half_width if np.isfinite(half_width) else None,
        upper=prediction + half_width if np.isfinite(half_width) else None,
        as_of=features.index[-1].date(),
        horizon=horizon,
        n_bars=n_bars,
        expected_skill_pct=skill,
        basis=(
            f"Pooled cross-sectional model fitted on {pool.n_symbols} symbols "
            f"({len(pool):,} rows)"
        ),
        transfer=transfer.as_dict() if transfer else None,
        caveats=caveats,
    )
