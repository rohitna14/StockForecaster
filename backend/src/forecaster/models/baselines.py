"""Naive baselines.

**These are the most important models in the repository.** A forecasting result
without a baseline is not a result -- "R-squared 0.62" means nothing until you
know what predicting *nothing* would have scored. The original prototype had no
baseline at all, which is why its "~15% better than naive last-price" claim had
no source in code.

All six are registered like any other model and appear on every leaderboard.
They cannot be filtered out of the UI.

A note on why ``NaiveLastPrice`` predicts exactly zero: for a *price* target the
naive forecast is ``price_{t+h} = price_t``. Expressed as a return -- which is
what we actually model -- that is ``0``. A model only beats it by correctly
identifying when returns are non-zero *and* in which direction, which is
genuinely hard. Random-walk theory says this baseline is close to optimal at
short horizons, and empirically it is very hard to beat on RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ModelContext


class NaiveLastPrice(Forecaster):
    """Random walk: tomorrow's price equals today's, so the predicted return is 0.

    The reference baseline for every skill score in the project.
    """

    name = "naive_last_price"
    display_name = "Naive (last price)"
    family = "baseline"

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        # Nothing to learn -- that is the point.
        self._n_features = X.shape[1] if X.ndim > 1 else 1

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")


class HistoricalMeanReturn(Forecaster):
    """Predicts the in-sample mean return -- the 'drift' of a random walk with drift.

    Slightly stronger than pure naive over long bull samples, and a useful
    check: if a model cannot beat a constant, it has learned nothing.
    """

    name = "historical_mean"
    display_name = "Historical mean return"
    family = "baseline"

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        finite = y[np.isfinite(y)]
        self._mean = float(finite.mean()) if len(finite) else 0.0

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.full(len(X), self._mean, dtype="float64")


class DriftBaseline(Forecaster):
    """Linear drift extrapolated from the recent trend.

    Fits ``price ~ a + b*t`` over the last ``window`` training bars and projects
    ``horizon`` bars forward, expressed as a return. Captures persistent trends
    that flat-zero misses, without any feature engineering.
    """

    name = "drift"
    display_name = "Linear drift"
    family = "baseline"

    def __init__(self, window: int = 60, **kwargs: object) -> None:
        super().__init__(window=window, **kwargs)
        self.window = window

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        self._drift = 0.0
        if context is None or context.close is None or len(context.close) < 3:
            # No price context: fall back to the mean training return.
            finite = y[np.isfinite(y)]
            self._drift = float(finite.mean()) if len(finite) else 0.0
            return

        prices = context.close.to_numpy(dtype="float64")
        tail = prices[-min(self.window, len(prices)) :]
        if len(tail) < 3 or tail[-1] <= 0:
            return

        t = np.arange(len(tail), dtype="float64")
        slope = float(np.polyfit(t, tail, 1)[0])
        # Slope is price per bar; convert to a return over the horizon.
        self._drift = slope * context.horizon / tail[-1]

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.full(len(X), self._drift, dtype="float64")


class EWMABaseline(Forecaster):
    """Exponentially weighted mean of recent returns.

    Weights recent observations more heavily than :class:`HistoricalMeanReturn`,
    so it tracks regime shifts instead of averaging across all of them.
    """

    name = "ewma"
    display_name = "EWMA of returns"
    family = "baseline"

    def __init__(self, span: int = 20, **kwargs: object) -> None:
        super().__init__(span=span, **kwargs)
        self.span = span

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        series = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            self._value = 0.0
            return
        self._value = float(series.ewm(span=self.span, adjust=False).mean().iloc[-1])

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.full(len(X), self._value, dtype="float64")


class SeasonalNaive(Forecaster):
    """Predicts the return observed one seasonal period ago.

    ``period=5`` tests a day-of-week effect; ``period=252`` an annual one. Rarely
    competitive on equities, which is itself informative -- it is the control
    that shows the calendar features are not doing the work.
    """

    name = "seasonal_naive"
    display_name = "Seasonal naive"
    family = "baseline"

    def __init__(self, period: int = 5, **kwargs: object) -> None:
        super().__init__(period=period, **kwargs)
        self.period = period

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        finite = y[np.isfinite(y)]
        self._fallback = float(finite.mean()) if len(finite) else 0.0
        # Keep the tail of training labels to seed the first test predictions.
        self._tail = y[-self.period :].copy() if len(y) >= self.period else np.array([])

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        n = len(X)
        out = np.full(n, self._fallback, dtype="float64")
        for i in range(min(n, len(self._tail))):
            value = self._tail[i]
            if np.isfinite(value):
                out[i] = value
        return out


class CoinFlip(Forecaster):
    """Random direction with a fixed seed -- the classification control.

    A directional accuracy of 55% sounds impressive until this scores 50% on the
    same folds and the difference sits inside the confidence interval.
    """

    name = "coin_flip"
    display_name = "Coin flip"
    family = "baseline"
    is_classifier = True

    def __init__(self, seed: int = 42, **kwargs: object) -> None:
        super().__init__(seed=seed, **kwargs)
        self.seed = seed

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        finite = y[np.isfinite(y)]
        # Match the training base rate rather than assuming 50/50: equities
        # drift up, so "always long" is a stronger control than a fair coin.
        self._p_up = float((finite > 0).mean()) if len(finite) else 0.5

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.choice([0.0, 1.0], size=len(X), p=[1 - self._p_up, self._p_up])

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame, context: ModelContext | None = None
    ) -> np.ndarray:
        return np.full(len(X), self._p_up, dtype="float64")


class AlwaysLong(Forecaster):
    """Always predicts 'up' -- buy-and-hold as a classifier.

    The honest benchmark for any directional model on equities: the market rises
    on roughly 53% of days, so a classifier scoring 53% has added nothing.
    """

    name = "always_long"
    display_name = "Always long"
    family = "baseline"
    is_classifier = True

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        finite = y[np.isfinite(y)]
        self._base_rate = float((finite > 0).mean()) if len(finite) else 0.5

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.ones(len(X), dtype="float64")

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame, context: ModelContext | None = None
    ) -> np.ndarray:
        return np.full(len(X), self._base_rate, dtype="float64")


#: The reference baseline used as the denominator of every skill score.
REFERENCE_BASELINE = NaiveLastPrice.name

BASELINES: dict[str, type[Forecaster]] = {
    NaiveLastPrice.name: NaiveLastPrice,
    HistoricalMeanReturn.name: HistoricalMeanReturn,
    DriftBaseline.name: DriftBaseline,
    EWMABaseline.name: EWMABaseline,
    SeasonalNaive.name: SeasonalNaive,
    CoinFlip.name: CoinFlip,
    AlwaysLong.name: AlwaysLong,
}
