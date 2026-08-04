"""Statistical / distributional features.

These describe the *character* of recent returns rather than price level, and
tend to be the features that survive walk-forward validation -- they are
stationary by construction and describe regime rather than direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_return(close: pd.Series, period: int = 1) -> pd.Series:
    return np.log(close / close.shift(period)).rename(f"log_return_{period}")


def rolling_skew(close: pd.Series, period: int = 20) -> pd.Series:
    """Skewness of returns -- crash risk vs melt-up asymmetry."""
    return close.pct_change().rolling(period, min_periods=period).skew().rename(f"skew_{period}")


def rolling_kurtosis(close: pd.Series, period: int = 20) -> pd.Series:
    """Excess kurtosis -- fat-tailedness of the recent return distribution."""
    return close.pct_change().rolling(period, min_periods=period).kurt().rename(f"kurtosis_{period}")


def autocorrelation(close: pd.Series, period: int = 20, lag: int = 1) -> pd.Series:
    """Rolling lag-k autocorrelation of returns.

    Positive => momentum regime, negative => mean-reverting regime. One of the
    few features with genuine theoretical justification for short-horizon
    prediction.
    """
    returns = close.pct_change()

    def _ac(window: np.ndarray) -> float:
        if len(window) <= lag + 1:
            return np.nan
        a, b = window[:-lag], window[lag:]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    return (
        returns.rolling(period, min_periods=period)
        .apply(_ac, raw=True)
        .rename(f"autocorr_{period}_lag{lag}")
    )


def hurst_exponent(close: pd.Series, period: int = 100, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range / variance scaling.

    H > 0.5 trending, H = 0.5 random walk, H < 0.5 mean-reverting. Expensive,
    so it is computed on a long window and excluded from the default fast
    feature set.
    """
    log_price = np.log(close)

    def _hurst(window: np.ndarray) -> float:
        lags = range(2, min(max_lag, len(window) // 2))
        tau = []
        for lag in lags:
            diff = window[lag:] - window[:-lag]
            std = np.std(diff)
            tau.append(std if std > 0 else np.nan)
        tau_arr = np.asarray(tau, dtype=float)
        if np.isnan(tau_arr).any() or len(tau_arr) < 3:
            return np.nan
        slope = np.polyfit(np.log(list(lags)), np.log(tau_arr), 1)[0]
        return float(slope)

    return (
        log_price.rolling(period, min_periods=period)
        .apply(_hurst, raw=True)
        .rename(f"hurst_{period}")
    )


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Standardise a series against its own trailing window.

    This is the workhorse for turning any price-scaled quantity into something
    a model can compare across tickers and across time.
    """
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1).replace(0.0, np.nan)
    return ((series - mean) / std).rename(f"{series.name or 'x'}_z{period}")


def drawdown_from_high(close: pd.Series, period: int = 252) -> pd.Series:
    """Current drawdown from the rolling high, as a negative fraction."""
    high = close.rolling(period, min_periods=1).max()
    return ((close - high) / high).rename(f"drawdown_{period}")


def days_since_high(close: pd.Series, period: int = 252) -> pd.Series:
    """Bars since the window maximum, normalised to [0, 1]."""
    return (
        close.rolling(period, min_periods=period)
        .apply(lambda x: (len(x) - 1 - int(np.argmax(x))) / len(x), raw=True)
        .rename(f"days_since_high_{period}")
    )


def downside_deviation(close: pd.Series, period: int = 20) -> pd.Series:
    """Std of negative returns only -- the denominator of the Sortino ratio."""
    returns = close.pct_change()
    negative = returns.where(returns < 0, np.nan)
    return (
        negative.rolling(period, min_periods=3).std(ddof=1) * np.sqrt(TRADING_DAYS)
    ).rename(f"downside_dev_{period}")


def gap_open(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap: today's open vs yesterday's close."""
    return (open_ / close.shift(1) - 1.0).rename("gap_open")


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """High-low range as a fraction of close."""
    return ((high - low) / close).rename("intraday_range")


def close_location(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Where the close sits within the bar, 0 (low) to 1 (high).

    A close pinned near the high after a wide range is the classic
    'buyers in control' bar; this encodes that without any charting mysticism.
    """
    span = (high - low).replace(0.0, np.nan)
    return ((close - low) / span).rename("close_location")
