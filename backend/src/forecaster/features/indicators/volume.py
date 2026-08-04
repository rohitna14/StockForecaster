"""Volume-based indicators. All causal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative signed volume.

    The raw level is a running total whose magnitude depends on how much
    history you happened to load, so it is useless as a model feature directly
    -- use :func:`obv_slope` or normalise it. Kept because it drives charts.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum().rename("obv")


def obv_slope(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV change over a window, scaled by window volume -- stationary."""
    raw = obv(close, volume)
    change = raw - raw.shift(period)
    scale = volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return (change / scale).rename(f"obv_slope_{period}")


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Today's volume over its own rolling average -- the volume-surge feature."""
    avg = volume.rolling(period, min_periods=period).mean().replace(0.0, np.nan)
    return (volume / avg).rename(f"volume_ratio_{period}")


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Standardised volume within a trailing window."""
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std(ddof=1).replace(0.0, np.nan)
    return ((volume - mean) / std).rename(f"volume_z_{period}")


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Rolling VWAP.

    Note this is a *rolling-window* VWAP, not the session-anchored VWAP a
    trading desk uses -- daily bars carry no intraday information to anchor to.
    """
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(period, min_periods=period).sum()
    vol = volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return (pv / vol).rename(f"vwap_{period}")


def price_to_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Close relative to rolling VWAP -- scale-free."""
    return (close / vwap(high, low, close, volume, period) - 1.0).rename(f"price_to_vwap_{period}")


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index -- RSI computed on price x volume."""
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    direction = np.sign(typical.diff()).fillna(0.0)

    positive = raw_flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()

    ratio = positive / negative.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    return out.where(negative != 0.0, 100.0).rename(f"mfi_{period}")


def accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution line."""
    span = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / span
    return (clv.fillna(0.0) * volume).cumsum().rename("ad_line")


def chaikin_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Chaikin Oscillator: EMA(3) - EMA(10) of the A/D line, volume-normalised."""
    ad = accumulation_distribution(high, low, close, volume)
    raw = ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()
    scale = volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    return (raw / scale).rename("chaikin_osc")


def dollar_volume(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Average daily dollar volume -- a liquidity filter, in log10 dollars.

    Used to exclude names too illiquid to trade at the sizes a backtest assumes.
    """
    dv = (close * volume).rolling(period, min_periods=period).mean()
    return np.log10(dv.replace(0.0, np.nan)).rename(f"log_dollar_volume_{period}")
