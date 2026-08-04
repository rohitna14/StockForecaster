"""Target (label) construction.

Every target here is **forward-looking by definition** -- that is the whole
point of a label. The danger is not that ``y_t`` depends on the future; it is
that the *training set* contains labels whose future overlaps the test set.

Concretely, with ``horizon=5``, the label at bar ``t`` is built from the close
at ``t+5``. If a fold trains through bar ``T`` and tests from ``T+1``, then the
labels at ``T-4 ... T`` were computed from bars inside the test window. The
model has therefore seen the test period's prices through its labels, and the
resulting metrics are inflated.

The fix is **purging**: drop the last ``horizon`` training labels. That is
implemented in :mod:`forecaster.validation.splitters` and asserted in
``tests/unit/test_leakage.py``. Nothing in this module tries to hide the
forward dependence -- it is declared, and the splitter compensates.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from forecaster.exceptions import InsufficientDataError


class TargetType(StrEnum):
    RETURN = "return"
    LOG_RETURN = "log_return"
    DIRECTION = "direction"
    VOL_SCALED_RETURN = "vol_scaled_return"
    TRIPLE_BARRIER = "triple_barrier"
    VOL_RATIO = "vol_ratio"


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Simple forward return: ``close_{t+h} / close_t - 1``.

    The last ``horizon`` values are NaN -- their future has not happened yet.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return (close.shift(-horizon) / close - 1.0).rename(f"fwd_return_{horizon}")


def forward_log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return -- additive across time, symmetric around zero."""
    return np.log(close.shift(-horizon) / close).rename(f"fwd_log_return_{horizon}")


def forward_direction(close: pd.Series, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """Binary up/down label.

    Args:
        threshold: dead-band. With ``threshold=0`` every bar is labelled, and
            near-flat days become coin-flips that dominate the sample. Setting
            it to e.g. one-quarter of daily volatility labels only moves worth
            trading, at the cost of discarding bars.
    """
    fwd = forward_return(close, horizon)
    if threshold <= 0:
        out = (fwd > 0).astype("float64")
        return out.where(fwd.notna()).rename(f"fwd_direction_{horizon}")

    out = pd.Series(np.nan, index=close.index, dtype="float64")
    out[fwd > threshold] = 1.0
    out[fwd < -threshold] = 0.0
    return out.rename(f"fwd_direction_{horizon}")


def forward_vol_scaled_return(
    close: pd.Series, horizon: int = 1, vol_window: int = 20
) -> pd.Series:
    """Forward return divided by *trailing* realised volatility.

    Two reasons this is usually the better regression target:

    1. It is roughly homoscedastic, so squared-error loss stops being dominated
       by whatever happened to be the most volatile month in the sample.
    2. It is comparable across tickers, which makes pooled cross-sectional
       training meaningful.

    The volatility divisor uses only data up to ``t`` (``shift(1)`` applied to a
    trailing window), so it introduces no additional lookahead beyond the label
    itself.
    """
    fwd = forward_return(close, horizon)
    daily_vol = close.pct_change().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    scale = (daily_vol * np.sqrt(horizon)).replace(0.0, np.nan)
    return (fwd / scale).rename(f"fwd_vol_scaled_{horizon}")


def forward_vol_ratio(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Log ratio of *future* realised volatility to *trailing* realised volatility.

    ``log( RV[t+1 .. t+h] / RV[t-h+1 .. t] )``

    Why this target is framed as a ratio rather than a level: it makes the
    project's reference baseline the correct one for free. A prediction of
    ``0`` means "volatility stays where it is" -- which is precisely the random
    walk in volatility, the standard naive forecast in the volatility
    literature. So ``naive_last_price`` (which predicts 0) becomes a genuine,
    well-posed baseline and the skill score answers a real question: *did the
    model beat assuming volatility persists?*

    Unlike direction, this is a target where a real edge is expected. Volatility
    clustering -- calm follows calm, turbulence follows turbulence -- is among
    the most robust empirical regularities in asset returns, and is the entire
    basis of the ARCH/GARCH literature. Volatility is also mean-reverting, so
    both persistence and reversion carry information a model can use.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    returns = np.log(close / close.shift(1))
    trailing = returns.rolling(horizon, min_periods=horizon).std(ddof=1)
    # Shift(-horizon) turns the trailing window into the forward window.
    forward = trailing.shift(-horizon)

    ratio = np.log(forward.replace(0.0, np.nan) / trailing.replace(0.0, np.nan))
    return ratio.rename(f"fwd_vol_ratio_{horizon}")


def triple_barrier(
    close: pd.Series,
    horizon: int = 5,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Lopez de Prado's triple-barrier labelling.

    From each bar, walk forward until one of three barriers is touched:
    a profit target (``+upper_mult`` x trailing vol), a stop
    (``-lower_mult`` x trailing vol), or the time limit ``horizon``. The label
    is the barrier that was hit first.

    This is closer to how a position is actually managed than a fixed-horizon
    return, and it yields far better-balanced classes than raw sign-of-return.

    Returns a frame with ``label`` (+1/-1/0), ``touch_index`` (bars until the
    barrier was hit) and ``realized_return``.
    """
    if len(close) < vol_window + horizon + 1:
        raise InsufficientDataError(
            f"triple_barrier needs > {vol_window + horizon + 1} bars, got {len(close)}"
        )

    vol = close.pct_change().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    values = close.to_numpy(dtype="float64")
    vol_values = vol.to_numpy(dtype="float64")
    n = len(values)

    labels = np.full(n, np.nan)
    touches = np.full(n, np.nan)
    realized = np.full(n, np.nan)

    for i in range(n):
        sigma = vol_values[i]
        if not np.isfinite(sigma) or sigma <= 0 or i + horizon >= n:
            continue

        entry = values[i]
        upper = entry * (1.0 + upper_mult * sigma)
        lower = entry * (1.0 - lower_mult * sigma)

        label, touch = 0.0, horizon
        for step in range(1, horizon + 1):
            price = values[i + step]
            if price >= upper:
                label, touch = 1.0, step
                break
            if price <= lower:
                label, touch = -1.0, step
                break

        labels[i] = label
        touches[i] = touch
        realized[i] = values[i + touch] / entry - 1.0

    return pd.DataFrame(
        {"label": labels, "touch_index": touches, "realized_return": realized},
        index=close.index,
    )


def build_target(
    frame: pd.DataFrame,
    target_type: TargetType | str = TargetType.RETURN,
    horizon: int = 1,
    **kwargs: float | int,
) -> pd.Series:
    """Dispatch to the requested target constructor."""
    target_type = TargetType(target_type)
    close = frame["close"]

    if target_type is TargetType.RETURN:
        return forward_return(close, horizon)
    if target_type is TargetType.LOG_RETURN:
        return forward_log_return(close, horizon)
    if target_type is TargetType.DIRECTION:
        return forward_direction(close, horizon, threshold=float(kwargs.get("threshold", 0.0)))
    if target_type is TargetType.VOL_SCALED_RETURN:
        return forward_vol_scaled_return(
            close, horizon, vol_window=int(kwargs.get("vol_window", 20))
        )
    if target_type is TargetType.VOL_RATIO:
        return forward_vol_ratio(close, horizon)
    if target_type is TargetType.TRIPLE_BARRIER:
        return triple_barrier(
            close,
            horizon=horizon,
            upper_mult=float(kwargs.get("upper_mult", 2.0)),
            lower_mult=float(kwargs.get("lower_mult", 2.0)),
        )["label"].rename(f"triple_barrier_{horizon}")

    raise ValueError(f"Unsupported target type: {target_type}")


def is_classification(target_type: TargetType | str) -> bool:
    return TargetType(target_type) in {TargetType.DIRECTION, TargetType.TRIPLE_BARRIER}
