"""Momentum and oscillator indicators.

Every function here is **causal**: the value at bar *t* depends only on bars
``<= t``. That is enforced by construction (``rolling``, ``ewm``, ``shift`` with
positive periods) and verified by ``tests/unit/test_leakage.py``, which corrupts
all data after a cut point and asserts nothing before it moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using **Wilder's smoothing**.

    The original prototype averaged gains and losses with a simple rolling
    mean. That is not RSI -- it reacts far too quickly and disagrees with every
    charting platform. Wilder's method is an EMA with ``alpha = 1/period``,
    which is what TradingView, ThinkOrSwim and TA-Lib all compute.

    Returns values in [0, 100]; NaN for the first ``period`` bars.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # adjust=False gives the recursive form: y_t = a*x_t + (1-a)*y_{t-1}
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains -> RSI is exactly 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out.rename(f"rsi_{period}")


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator (%K fast line, %D signal line)."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    k = 100.0 * (close - lowest) / span
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({f"stoch_k_{k_period}": k, f"stoch_d_{d_period}": d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R -- an inverted stochastic, ranging [-100, 0]."""
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return (-100.0 * (highest - close) / span).rename(f"williams_r_{period}")


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index.

    Uses mean absolute deviation (Lambert's original definition), not standard
    deviation -- the 0.015 constant is calibrated for MAD.
    """
    typical = (high + low + close) / 3.0
    sma = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return ((typical - sma) / (0.015 * mad.replace(0.0, np.nan))).rename(f"cci_{period}")


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change over ``period`` bars, as a fraction."""
    return close.pct_change(period).rename(f"roc_{period}")


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Price ratio momentum: ``close_t / close_{t-n} - 1``."""
    return (close / close.shift(period) - 1.0).rename(f"momentum_{period}")


def tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index -- double-smoothed momentum."""
    diff = close.diff()
    smooth = diff.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    abs_smooth = diff.abs().ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    return (100.0 * smooth / abs_smooth.replace(0.0, np.nan)).rename("tsi")


def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Awesome Oscillator: SMA5 - SMA34 of the median price."""
    median = (high + low) / 2.0
    ao = median.rolling(5, min_periods=5).mean() - median.rolling(34, min_periods=34).mean()
    return ao.rename("awesome_oscillator")
