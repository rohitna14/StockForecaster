"""Trend-following indicators. All causal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean().rename(f"sma_{period}")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential MA in recursive form.

    ``adjust=False`` matters: the default ``adjust=True`` computes a
    re-weighted expanding average whose early values differ from what every
    charting platform shows, and whose value at bar *t* depends on the length
    of history supplied. That second property makes the feature non-stationary
    across folds -- the same date yields different numbers depending on where
    the training window started.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean().rename(f"ema_{period}")


def wma(series: pd.Series, period: int) -> pd.Series:
    """Linearly weighted MA."""
    weights = np.arange(1, period + 1, dtype=float)
    return (
        series.rolling(period, min_periods=period)
        .apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)
        .rename(f"wma_{period}")
    )


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI.

    ADX measures trend *strength* irrespective of direction: above ~25 is a
    trending regime, below ~20 is chop. Useful as a regime feature -- momentum
    signals behave very differently across that boundary.
    """
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1 / period
    atr_ = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame({f"adx_{period}": adx_, "plus_di": plus_di, "minus_di": minus_di})


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon Up/Down: how recently the window's extreme occurred."""
    up = high.rolling(period + 1, min_periods=period + 1).apply(
        lambda x: 100.0 * (period - (len(x) - 1 - int(np.argmax(x)))) / period, raw=True
    )
    down = low.rolling(period + 1, min_periods=period + 1).apply(
        lambda x: 100.0 * (period - (len(x) - 1 - int(np.argmin(x)))) / period, raw=True
    )
    return pd.DataFrame({f"aroon_up_{period}": up, f"aroon_down_{period}": down,
                         f"aroon_osc_{period}": up - down})


def price_to_ma(close: pd.Series, period: int) -> pd.Series:
    """Close relative to its own moving average.

    Scale-free by construction, so the feature is comparable across a $3 stock
    and a $3,000 one. Raw price levels are the single most common way to make a
    model memorise a ticker instead of learning a pattern.
    """
    ma = close.rolling(period, min_periods=period).mean()
    return (close / ma - 1.0).rename(f"price_to_sma_{period}")


def ma_crossover(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Normalised gap between two MAs, expressed in units of the slow MA."""
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    return ((fast_ma - slow_ma) / slow_ma.replace(0.0, np.nan)).rename(f"ma_cross_{fast}_{slow}")


def ichimoku(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Ichimoku conversion and base lines.

    The forward-shifted cloud spans (senkou A/B, displaced +26) are deliberately
    omitted: they are plotted into the future and would be lookahead if joined
    on the bar date.
    """
    conv = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
    base = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
    return pd.DataFrame({"ichimoku_conv": conv, "ichimoku_base": base,
                         "ichimoku_conv_base_gap": conv - base})
