"""Backtesting endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter

from forecaster.api.deps import OHLCVRepoDep, load_bars
from forecaster.api.schemas.common import (
    BacktestRequest,
    BacktestResponse,
    EquityPoint,
)
from forecaster.backtest.costs import COST_PRESETS
from forecaster.backtest.engine import (
    BacktestConfig,
    Backtester,
    SizingMode,
    sweep_costs,
)
from forecaster.exceptions import ModelError
from forecaster.features.targets import TargetType
from forecaster.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["backtests"])


def _oos_predictions(report: Any, model: str) -> pd.Series:
    """Stitch a model's out-of-sample fold predictions into one series.

    Only out-of-sample predictions are used. Including in-sample fitted values
    would make any backtest look extraordinary and mean nothing.
    """
    result = report.results.get(model)
    if result is None or not result.succeeded:
        raise ModelError(
            f"Model {model} produced no usable predictions: "
            f"{result.error if result else 'not found'}",
            model=model,
        )

    dates = np.concatenate([fold.dates.to_numpy() for fold in result.folds])
    values = np.concatenate([fold.y_pred for fold in result.folds])
    series = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    return series[~series.index.duplicated(keep="first")]


@router.post("/backtests", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, ohlcv: OHLCVRepoDep) -> BacktestResponse:
    """Walk-forward evaluate, then backtest the out-of-sample predictions.

    The response always carries the buy-and-hold benchmark and a cost-
    sensitivity sweep, because a strategy return quoted without either is not
    interpretable.
    """
    from forecaster.validation.harness import EvaluationConfig, EvaluationHarness

    frame, _ = await load_bars(request.symbol, ohlcv)

    config = EvaluationConfig(
        symbol=request.symbol.upper(),
        models=[request.model],
        horizon=request.horizon,
        target_type=TargetType.RETURN,
    )
    report = await asyncio.to_thread(lambda: EvaluationHarness(config).run(frame))
    predictions = _oos_predictions(report, request.model)

    cost_model = COST_PRESETS.get(request.cost_preset, COST_PRESETS["realistic"])
    backtest_config = BacktestConfig(
        initial_capital=request.initial_capital,
        sizing=SizingMode(request.sizing),
        costs=cost_model,
    )

    result = await asyncio.to_thread(
        lambda: Backtester(backtest_config).run(
            frame["close"], predictions, opens=frame["open"]
        )
    )
    sweep = await asyncio.to_thread(
        lambda: sweep_costs(frame["close"], predictions, frame["open"])
    )

    equity_curve = [
        EquityPoint(
            ts=ts.date(),
            equity=float(result.equity.loc[ts]),
            benchmark_equity=float(result.benchmark_equity.loc[ts]),
            drawdown=float(result.drawdown.loc[ts]),
            position=float(result.positions.loc[ts]),
        )
        for ts in result.equity.index
    ]

    trades = result.trades.copy()
    if not trades.empty:
        for column in ("entry_date", "exit_date"):
            trades[column] = trades[column].astype(str)

    return BacktestResponse(
        symbol=request.symbol.upper(),
        model=request.model,
        stats=_json_safe(result.stats),
        equity_curve=equity_curve,
        trades=_json_safe(trades.to_dict("records")),
        cost_sensitivity=_json_safe(sweep.to_dict("records")),
    )


@router.get("/backtests/presets")
async def cost_presets() -> dict[str, Any]:
    """Available cost models, with the implied round-trip figure."""
    return {
        name: {**model.as_dict(), "description": _PRESET_NOTES.get(name, "")}
        for name, model in COST_PRESETS.items()
    }


_PRESET_NOTES = {
    "zero": "No costs. Useful only to isolate signal quality -- never a claim.",
    "optimistic": "Large-cap US equity, tight spreads, small size.",
    "realistic": "Default. Reasonable retail assumptions for liquid names.",
    "conservative": "Small caps or larger size; stress test for whether an edge survives.",
}


def _json_safe(obj: Any) -> Any:
    """NaN and inf are not valid JSON; convert to null rather than emitting garbage."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj
