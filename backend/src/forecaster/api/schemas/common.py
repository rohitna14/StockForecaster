"""Shared Pydantic response models."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class Page(Schema, Generic[T]):
    items: list[T]
    total: int | None = None
    limit: int
    offset: int
    has_more: bool = False


class HealthResponse(Schema):
    status: str
    version: str
    environment: str


class ReadinessResponse(Schema):
    status: str
    database: bool
    lake: bool
    providers: dict[str, Any]
    hot_symbols: int
    lake_symbols: int


# ── instruments ───────────────────────────────────────────────────────────
class InstrumentSummary(Schema):
    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    market_cap: float | None = None
    tier: str
    first_date: dt.date | None = None
    last_date: dt.date | None = None


class Bar(Schema):
    ts: dt.date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class OHLCVResponse(Schema):
    symbol: str
    adjusted: bool
    source_tier: str = Field(description="'hot' (Postgres) or 'cold' (Parquet/DuckDB)")
    bars: list[Bar]
    count: int


class IndicatorResponse(Schema):
    symbol: str
    features: list[str]
    index: list[dt.date]
    values: dict[str, list[float | None]]


class RiskResponse(Schema):
    symbol: str
    annual_volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    var_95: float | None
    cvar_95: float | None
    n_observations: int


# ── models & runs ─────────────────────────────────────────────────────────
class ModelInfo(Schema):
    name: str
    display_name: str
    family: str
    is_classifier: bool
    requires_scaling: bool
    is_sequence_model: bool
    is_baseline: bool
    requires_extra: str | None = None


class EvaluationRequest(Schema):
    symbol: str
    models: list[str] = Field(default_factory=lambda: ["ridge", "lightgbm"])
    horizon: int = Field(default=5, ge=1, le=60)
    target_type: str = "vol_ratio"
    feature_set: str = "core"
    train_size: int = Field(default=504, ge=100, le=3000)
    test_size: int = Field(default=63, ge=10, le=500)
    embargo: int = Field(default=2, ge=0, le=50)


class RunStatusResponse(Schema):
    run_id: uuid.UUID
    status: str
    progress: float
    model_name: str | None = None
    symbol: str | None = None
    horizon_days: int | None = None
    error_message: str | None = None
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_seconds: float | None = None


class LeaderboardRow(Schema):
    model: str
    display_name: str | None = None
    family: str
    is_baseline: bool
    rmse: float | None = None
    mase: float | None = None
    r2: float | None = None
    hit_rate: float | None = None
    base_rate: float | None = None
    rmse_skill_pct: float | None = None
    directional_skill_pct: float | None = None
    dm_pvalue: float | None = None
    interval_coverage: float | None = None
    sharpe: float | None = None


class FoldMetricRow(Schema):
    fold_index: int
    train_start: dt.date | None = None
    train_end: dt.date | None = None
    test_start: dt.date | None = None
    test_end: dt.date | None = None
    n_train: int | None = None
    n_test: int | None = None
    rmse: float | None = None
    mase: float | None = None
    r2: float | None = None
    directional_accuracy: float | None = None
    skill_vs_naive: float | None = None
    sharpe: float | None = None


class PredictionRow(Schema):
    as_of_date: dt.date
    target_date: dt.date
    y_pred: float
    y_pred_lower: float | None = None
    y_pred_upper: float | None = None
    y_true: float | None = None
    fold_index: int | None = None


class EvaluationResponse(Schema):
    symbol: str
    horizon: int
    target_type: str
    n_folds: int
    n_samples: int
    n_features: int
    leaderboard: list[LeaderboardRow]
    selected_model: str
    duration_seconds: float
    run_ids: dict[str, uuid.UUID] = Field(default_factory=dict)


class ExplanationResponse(Schema):
    run_id: uuid.UUID | None = None
    model: str
    method: str
    top_features: list[dict[str, Any]]
    narrative: str
    drivers: str


class ForecastResponse(Schema):
    symbol: str
    model: str
    horizon: int
    target_type: str
    as_of_date: dt.date
    prediction: float
    lower: float | None = None
    upper: float | None = None
    skill_pct: float | None = None
    dm_pvalue: float | None = None
    narrative: str
    is_informative: bool = Field(
        description="False when the model historically lost to the naive baseline."
    )


# ── backtests ─────────────────────────────────────────────────────────────
class BacktestRequest(Schema):
    symbol: str
    model: str = "lightgbm"
    horizon: int = Field(default=1, ge=1, le=60)
    cost_preset: str = "realistic"
    sizing: str = "vol_target"
    initial_capital: float = 100_000.0


class EquityPoint(Schema):
    ts: dt.date
    equity: float
    benchmark_equity: float | None = None
    drawdown: float | None = None
    position: float | None = None


class BacktestResponse(Schema):
    symbol: str
    model: str
    stats: dict[str, Any]
    equity_curve: list[EquityPoint]
    trades: list[dict[str, Any]]
    cost_sensitivity: list[dict[str, Any]] = Field(default_factory=list)


class GlossaryEntry(Schema):
    key: str
    label: str
    short: str
    plain: str
    good_direction: str
    caveat: str | None = None
