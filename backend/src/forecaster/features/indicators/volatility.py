"""Volatility and range indicators. All causal."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Max of (H-L), |H-C_prev|, |L-C_prev| -- captures overnight gaps."""
    prev_close = close.shift(1)
    return (
        pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
        .max(axis=1)
        .rename("true_range")
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename(f"atr_{period}")


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR as a fraction of price -- the scale-free version, use this as a feature."""
    return (atr(high, low, close, period) / close).rename(f"atr_pct_{period}")


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands plus %B and bandwidth.

    ``bb_pct`` (where price sits in the band, 0-1) and ``bb_width`` (band width
    relative to the middle) are the model-usable outputs; the raw band levels
    are price-scaled and belong on a chart, not in a feature matrix.
    """
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct": (close - lower) / span,
            "bb_width": span / middle.replace(0.0, np.nan),
        }
    )


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    """Keltner Channels -- EMA centre with ATR-scaled bands."""
    middle = close.ewm(span=period, adjust=False, min_periods=period).mean()
    band = mult * atr(high, low, close, period)
    return pd.DataFrame(
        {
            "keltner_upper": middle + band,
            "keltner_lower": middle - band,
            "keltner_pct": (close - (middle - band)) / (2 * band).replace(0.0, np.nan),
        }
    )


def donchian(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.DataFrame:
    """Donchian channel position -- the classic breakout feature."""
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {f"donchian_pct_{period}": (close - lower) / span, f"donchian_width_{period}": span / close}
    )


def realized_volatility(close: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of log returns."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(period, min_periods=period).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol.rename(f"realized_vol_{period}")


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Parkinson high-low volatility estimator.

    Uses the intraday range rather than close-to-close, making it roughly 5x
    more efficient per observation than the standard deviation of returns --
    the same accuracy from a shorter window, which matters when regimes shift.
    """
    factor = 1.0 / (4.0 * np.log(2.0))
    hl = np.log(high / low) ** 2
    var = factor * hl.rolling(period, min_periods=period).mean()
    return (np.sqrt(var * TRADING_DAYS)).rename(f"parkinson_vol_{period}")


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Garman-Klass estimator -- uses the full OHLC bar."""
    hl = 0.5 * np.log(high / low) ** 2
    co = (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    var = (hl - co).rolling(period, min_periods=period).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS).rename(f"garman_klass_vol_{period}")


def volatility_ratio(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Short-window vol over long-window vol -- a volatility-regime signal."""
    log_ret = np.log(close / close.shift(1))
    short_vol = log_ret.rolling(short, min_periods=short).std(ddof=1)
    long_vol = log_ret.rolling(long, min_periods=long).std(ddof=1)
    return (short_vol / long_vol.replace(0.0, np.nan)).rename(f"vol_ratio_{short}_{long}")


def ulcer_index(close: pd.Series, period: int = 14) -> pd.Series:
    """Ulcer Index -- RMS drawdown over the window; penalises depth and duration."""
    rolling_max = close.rolling(period, min_periods=period).max()
    drawdown = 100.0 * (close - rolling_max) / rolling_max.replace(0.0, np.nan)
    return np.sqrt((drawdown**2).rolling(period, min_periods=period).mean()).rename(
        f"ulcer_{period}"
    )
