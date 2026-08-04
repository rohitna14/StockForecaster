"""Shared test fixtures.

Synthetic data is generated rather than downloaded so the whole unit suite runs
offline, deterministically, and in CI without an API key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 1500,
    *,
    seed: int = 42,
    start: str = "2018-01-01",
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Geometric Brownian motion OHLCV with realistic intraday structure.

    Deliberately contains **no predictable signal** beyond drift. Any model that
    shows meaningful skill on this data has found a bug, not a pattern -- which
    is exactly what several tests here assert.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.normal((annual_drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), size=n)
    close = start_price * np.exp(np.cumsum(shocks))

    # Intraday range scaled to daily volatility.
    daily_vol = annual_vol / np.sqrt(252)
    spread = np.abs(rng.normal(0, daily_vol, n)) * close
    open_ = close * (1 + rng.normal(0, daily_vol * 0.4, n))
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)
    volume = rng.lognormal(15.5, 0.4, n).astype("int64")

    index = pd.bdate_range(start=start, periods=n, name="ts")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


def make_predictable_ohlcv(n: int = 1500, *, seed: int = 7, strength: float = 0.6) -> pd.DataFrame:
    """OHLCV with a deliberately planted, learnable signal.

    Tomorrow's return is a strong linear function of a 5-day momentum term. Used
    as a positive control: a correctly wired pipeline MUST find skill here. If
    it does not, the failure is in the plumbing, not the market.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    noise = rng.normal(0, 0.01, n)
    for t in range(6, n):
        signal = np.mean(returns[t - 5 : t])
        returns[t] = strength * signal + noise[t]

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.lognormal(15.0, 0.3, n).astype("int64")

    index = pd.bdate_range(start="2018-01-01", periods=n, name="ts")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Random-walk OHLCV -- ~6 years of business days."""
    return make_ohlcv()


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    return make_ohlcv(n=400, seed=1)


@pytest.fixture
def predictable_ohlcv() -> pd.DataFrame:
    """OHLCV containing a real, learnable signal (positive control)."""
    return make_predictable_ohlcv()


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """Constant price -- exercises the division-by-zero paths in indicators."""
    n = 300
    index = pd.bdate_range("2020-01-01", periods=n, name="ts")
    return pd.DataFrame(
        {
            "open": 50.0,
            "high": 50.0,
            "low": 50.0,
            "close": 50.0,
            "adj_close": 50.0,
            "volume": 1_000_000,
        },
        index=index,
    )
