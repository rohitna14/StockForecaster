"""Backtester tests.

The headline guard is :func:`test_same_bar_signal_earns_nothing`, which pins the
next-bar execution rule. Everything else in a backtest can be right and that one
mistake will still manufacture a spectacular fake edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.backtest.costs import COST_PRESETS, CostModel
from forecaster.backtest.engine import (
    BacktestConfig,
    Backtester,
    SizingMode,
    sweep_costs,
)


@pytest.fixture
def price_series() -> pd.Series:
    rng = np.random.default_rng(7)
    n = 600
    returns = rng.normal(0.0004, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.bdate_range("2021-01-01", periods=n), name="close")


def _zero_cost_config(**kwargs) -> BacktestConfig:
    return BacktestConfig(costs=COST_PRESETS["zero"], sizing=SizingMode.SIGN, **kwargs)


# ═══════════════════════ the lookahead guard ═══════════════════════════════
def test_same_bar_signal_earns_nothing(price_series: pd.Series) -> None:
    """A forecast of bar t's own return must NOT capture bar t's return.

    We hand the backtester a 'prediction' that is literally the realised return
    of the same bar -- perfect same-bar foresight. Under correct next-bar
    execution the position is established the following bar, so this oracle
    should earn roughly nothing beyond noise.

    If this test ever shows a large positive return, the engine is filling on
    the signal bar and every backtest it produces is worthless.
    """
    actual_returns = price_series.pct_change().fillna(0.0)

    result = Backtester(_zero_cost_config()).run(price_series, actual_returns)

    total = result.stats["total_return"]
    assert abs(total) < 0.75, (
        f"Same-bar oracle earned {total:+.2%}. The engine is executing on the "
        f"signal bar rather than the next one -- this is lookahead."
    )


def test_next_bar_oracle_is_extremely_profitable(price_series: pd.Series) -> None:
    """Positive control: a forecast of the NEXT bar's return should print money.

    Confirms the engine can capture a real edge when one exists, so the test
    above is measuring execution timing rather than a broken engine.
    """
    next_bar_returns = price_series.pct_change().shift(-1).fillna(0.0)

    result = Backtester(_zero_cost_config()).run(price_series, next_bar_returns)

    assert result.stats["total_return"] > 5.0, (
        f"Next-bar oracle only made {result.stats['total_return']:+.2%}; "
        f"a perfect forecast should compound enormously."
    )
    assert result.stats["hit_rate"] > 0.95


def test_positions_are_shifted_relative_to_signal(price_series: pd.Series) -> None:
    """The first bar must hold no position -- there is no prior signal."""
    predictions = pd.Series(1.0, index=price_series.index)
    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    assert result.positions.iloc[0] == 0.0
    assert (result.positions.iloc[1:] > 0).all()


# ═══════════════════════════ costs ═════════════════════════════════════════
def test_costs_reduce_returns_monotonically(price_series: pd.Series) -> None:
    rng = np.random.default_rng(3)
    predictions = pd.Series(rng.normal(0, 0.01, len(price_series)), index=price_series.index)

    sweep = sweep_costs(price_series, predictions)
    assert len(sweep) == len(COST_PRESETS)

    ordered = sweep.sort_values("round_trip_bps")
    returns = ordered["total_return"].to_numpy()
    assert np.all(np.diff(returns) <= 1e-9), (
        f"Returns must fall as costs rise, got {returns}"
    )


def test_costs_charged_on_turnover_not_per_bar(price_series: pd.Series) -> None:
    """A model holding one view all year must pay far less than one flipping daily.

    Charging costs per bar rather than per trade would make these equal, which
    unfairly punishes exactly the stable models you want to reward.
    """
    index = price_series.index
    steady = pd.Series(1.0, index=index)
    flipping = pd.Series(np.tile([1.0, -1.0], len(index) // 2 + 1)[: len(index)], index=index)

    config = BacktestConfig(costs=COST_PRESETS["realistic"], sizing=SizingMode.SIGN)
    steady_costs = Backtester(config).run(price_series, steady).stats["total_costs"]
    flip_costs = Backtester(config).run(price_series, flipping).stats["total_costs"]

    assert flip_costs > steady_costs * 50, (
        f"Daily flipping cost {flip_costs:.4f} vs steady {steady_costs:.4f}; "
        f"costs do not appear to scale with turnover"
    )


def test_short_positions_pay_borrow() -> None:
    model = CostModel(commission_bps=0, half_spread_bps=0, slippage_bps=0,
                      slippage_vol_coefficient=0, short_borrow_bps_annual=100.0)
    long_only = model.apply(np.ones(252))
    short_only = model.apply(-np.ones(252))

    assert short_only.sum() > long_only.sum()
    assert short_only[1:].sum() == pytest.approx(100.0 / 10_000, rel=0.05)


# ═══════════════════════════ benchmark ═════════════════════════════════════
def test_benchmark_is_buy_and_hold(price_series: pd.Series) -> None:
    predictions = pd.Series(0.0, index=price_series.index)
    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    expected = price_series.iloc[-1] / price_series.iloc[0] - 1.0
    assert result.stats["benchmark_return"] == pytest.approx(expected, rel=1e-6)


def test_flat_signal_produces_flat_equity(price_series: pd.Series) -> None:
    predictions = pd.Series(0.0, index=price_series.index)
    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    assert result.stats["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert result.stats["n_trades"] == 0
    assert result.stats["time_in_market"] == 0.0


# ═══════════════════════════ sizing ════════════════════════════════════════
def test_long_only_never_shorts(price_series: pd.Series) -> None:
    rng = np.random.default_rng(11)
    predictions = pd.Series(rng.normal(0, 0.01, len(price_series)), index=price_series.index)

    config = BacktestConfig(sizing=SizingMode.LONG_ONLY, costs=COST_PRESETS["zero"])
    result = Backtester(config).run(price_series, predictions)

    assert (result.positions >= 0).all()


def test_leverage_is_capped(price_series: pd.Series) -> None:
    predictions = pd.Series(1.0, index=price_series.index)
    config = BacktestConfig(
        sizing=SizingMode.VOL_TARGET, max_leverage=1.0,
        target_volatility=10.0,  # absurd target that would demand huge leverage
        costs=COST_PRESETS["zero"],
    )
    result = Backtester(config).run(price_series, predictions)

    assert result.positions.abs().max() <= 1.0 + 1e-9


def test_signal_threshold_suppresses_small_forecasts(price_series: pd.Series) -> None:
    rng = np.random.default_rng(5)
    predictions = pd.Series(rng.normal(0, 0.001, len(price_series)), index=price_series.index)

    unfiltered = Backtester(
        BacktestConfig(sizing=SizingMode.SIGN, signal_threshold=0.0, costs=COST_PRESETS["zero"])
    ).run(price_series, predictions)
    filtered = Backtester(
        BacktestConfig(sizing=SizingMode.SIGN, signal_threshold=0.01, costs=COST_PRESETS["zero"])
    ).run(price_series, predictions)

    assert filtered.stats["time_in_market"] < unfiltered.stats["time_in_market"]
    assert filtered.stats["time_in_market"] == 0.0


# ═══════════════════════════ bookkeeping ═══════════════════════════════════
def test_equity_curve_matches_compounded_returns(price_series: pd.Series) -> None:
    rng = np.random.default_rng(13)
    predictions = pd.Series(rng.normal(0, 0.01, len(price_series)), index=price_series.index)
    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    expected = result.config.initial_capital * (1 + result.returns).cumprod()
    pd.testing.assert_series_equal(result.equity, expected, check_names=False)


def test_drawdown_is_never_positive(price_series: pd.Series) -> None:
    rng = np.random.default_rng(17)
    predictions = pd.Series(rng.normal(0, 0.01, len(price_series)), index=price_series.index)
    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    assert (result.drawdown <= 1e-12).all()
    assert result.stats["max_drawdown"] <= 0.0


def test_misaligned_series_are_intersected(price_series: pd.Series) -> None:
    """Predictions covering only part of the price history must align, not shift."""
    subset = price_series.index[100:400]
    predictions = pd.Series(0.01, index=subset)

    result = Backtester(_zero_cost_config()).run(price_series, predictions)

    assert len(result.equity) == len(subset)
    assert result.equity.index[0] == subset[0]


def test_too_few_bars_raises(price_series: pd.Series) -> None:
    tiny = price_series.iloc[:5]
    with pytest.raises(ValueError, match="Not enough aligned bars"):
        Backtester(_zero_cost_config()).run(tiny, pd.Series(0.01, index=tiny.index))
