"""SQLAlchemy 2.0 ORM models.

Design notes worth defending in an interview:

* **Prices are keyed ``(instrument_id, ts)``** with a CHECK constraint enforcing
  OHLC sanity at the database level, so a bad provider response cannot land.
* **``adj_close`` is stored alongside ``close``.** Models train on
  split/dividend-adjusted series; the UI displays raw close. Conflating the two
  is a classic silent bug.
* **``fundamentals.report_date`` is the join key, never ``fiscal_period``.**
  Joining on the fiscal quarter leaks a company's results backwards into the
  quarter they describe -- the numbers were not public yet. See
  ``docs/METHODOLOGY.md``.
* **Every ``model_run`` records ``git_sha``, ``random_seed`` and ``split_config``**
  so any published metric can be reproduced exactly.
* **``predictions`` stores ``as_of_date`` and ``target_date`` separately.**
  ``as_of_date`` is the last bar the model was allowed to see. Keeping both
  makes lookahead auditable with a SQL query rather than trust.
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from forecaster.db.types import BigIntPK, JSONColumn, MoneyColumn, TZDateTime, UUIDColumn


class Base(DeclarativeBase):
    """Declarative base with a shared type map."""

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSONColumn,
        dt.datetime: TZDateTime,
    }

    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════
class Tier(StrEnum):
    HOT = "hot"
    COLD = "cold"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TargetType(StrEnum):
    RETURN = "return"
    LOG_RETURN = "log_return"
    DIRECTION = "direction"
    VOL_SCALED_RETURN = "vol_scaled_return"
    TRIPLE_BARRIER = "triple_barrier"


class ActionType(StrEnum):
    SPLIT = "split"
    DIVIDEND = "dividend"


# ═══════════════════════════════════════════════════════════════════════════
# Reference data
# ═══════════════════════════════════════════════════════════════════════════
class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(50))
    asset_type: Mapped[str] = mapped_column(String(20), default="equity")
    sector: Mapped[str | None] = mapped_column(String(100), index=True)
    industry: Mapped[str | None] = mapped_column(String(150))
    country: Mapped[str | None] = mapped_column(String(80))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    market_cap: Mapped[float | None] = mapped_column(Float)
    ipo_year: Mapped[int | None] = mapped_column(SmallInteger)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    #: 'hot' rows have their OHLCV in Postgres; 'cold' rows live only in the
    #: Parquet lake until first request promotes them.
    tier: Mapped[str] = mapped_column(String(10), default=Tier.COLD, index=True)

    first_date: Mapped[dt.date | None] = mapped_column(Date)
    last_date: Mapped[dt.date | None] = mapped_column(Date)

    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    bars: Mapped[list[OHLCVDaily]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("idx_instruments_tier_active", "tier", "is_active"),
        Index("idx_instruments_name", "name"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Prices
# ═══════════════════════════════════════════════════════════════════════════
class OHLCVDaily(Base):
    __tablename__ = "ohlcv_daily"

    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[dt.date] = mapped_column(Date, primary_key=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    #: Split- and dividend-adjusted close. This is what models train on.
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)

    source: Mapped[str] = mapped_column(String(30))
    ingested_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())

    instrument: Mapped[Instrument] = relationship(back_populates="bars")

    __table_args__ = (
        CheckConstraint(
            "high >= low AND high >= open AND high >= close "
            "AND low <= open AND low <= close AND volume >= 0 AND close > 0",
            name="ck_ohlcv_sane",
        ),
        Index("idx_ohlcv_ts", "ts"),
    )


class OHLCVIntraday(Base):
    """5-minute bars, retained for a rolling 60 days and pruned nightly."""

    __tablename__ = "ohlcv_intraday"

    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(30))


class CorporateAction(Base):
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), index=True
    )
    ex_date: Mapped[dt.date] = mapped_column(Date)
    action_type: Mapped[str] = mapped_column(String(20))
    ratio: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("instrument_id", "ex_date", "action_type", name="uq_corp_action"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fundamentals, macro, news
# ═══════════════════════════════════════════════════════════════════════════
class Fundamental(Base):
    __tablename__ = "fundamentals"

    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    fiscal_period: Mapped[str] = mapped_column(String(10), primary_key=True)

    #: The date the filing became public. ALWAYS as-of join on this column --
    #: joining on fiscal_period leaks results backwards into their own quarter.
    report_date: Mapped[dt.date] = mapped_column(Date, index=True)

    revenue: Mapped[float | None] = mapped_column(Float)
    net_income: Mapped[float | None] = mapped_column(Float)
    eps_diluted: Mapped[float | None] = mapped_column(Float)
    total_assets: Mapped[float | None] = mapped_column(Float)
    total_debt: Mapped[float | None] = mapped_column(Float)
    free_cash_flow: Mapped[float | None] = mapped_column(Float)
    shares_outstanding: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30))


class MacroSeries(Base):
    __tablename__ = "macro_series"

    code: Mapped[str] = mapped_column(String(30), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    frequency: Mapped[str | None] = mapped_column(String(20))
    units: Mapped[str | None] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(30), default="FRED")


class MacroObservation(Base):
    __tablename__ = "macro_observations"

    code: Mapped[str] = mapped_column(
        String(30), ForeignKey("macro_series.code", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    value: Mapped[float | None] = mapped_column(Float)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE")
    )
    published_at: Mapped[dt.datetime] = mapped_column(TZDateTime)
    headline: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1000), unique=True)
    source: Mapped[str | None] = mapped_column(String(100))

    sentiment_label: Mapped[str | None] = mapped_column(String(20))
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    scored_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)

    __table_args__ = (Index("idx_news_lookup", "instrument_id", "published_at"),)


# ═══════════════════════════════════════════════════════════════════════════
# Universes
# ═══════════════════════════════════════════════════════════════════════════
class Universe(Base):
    __tablename__ = "universes"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    description: Mapped[str | None] = mapped_column(Text)


class UniverseMember(Base):
    """Point-in-time membership.

    ``added_at`` / ``removed_at`` exist so backtests can reconstruct the index
    as it stood historically, avoiding survivorship bias.
    """

    __tablename__ = "universe_members"

    universe_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("universes.id", ondelete="CASCADE"), primary_key=True
    )
    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    removed_at: Mapped[dt.date | None] = mapped_column(Date)


# ═══════════════════════════════════════════════════════════════════════════
# ML lifecycle
# ═══════════════════════════════════════════════════════════════════════════
class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(20))
    #: Full feature spec (names + params). Reproduces the matrix exactly.
    spec: Mapped[dict[str, Any]] = mapped_column(JSONColumn)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("name", "version", name="uq_feature_set"),)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUIDColumn, primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE"), index=True
    )
    universe_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("universes.id", ondelete="SET NULL")
    )
    feature_set_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("feature_sets.id", ondelete="SET NULL")
    )

    model_name: Mapped[str] = mapped_column(String(60), index=True)
    model_version: Mapped[str] = mapped_column(String(20), default="1")
    horizon_days: Mapped[int] = mapped_column(SmallInteger)
    target_type: Mapped[str] = mapped_column(String(30), default=TargetType.RETURN)

    hyperparams: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    #: Walk-forward configuration: train/test sizes, step, purge, embargo, mode.
    split_config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)

    status: Mapped[str] = mapped_column(String(20), default=RunStatus.QUEUED, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text)

    # Reproducibility
    git_sha: Mapped[str | None] = mapped_column(String(40))
    random_seed: Mapped[int | None] = mapped_column(Integer)

    started_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    finished_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())

    folds: Mapped[list[FoldMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    aggregate: Mapped[RunMetric | None] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False, passive_deletes=True
    )

    __table_args__ = (
        Index("idx_runs_lookup", "instrument_id", "model_name", "horizon_days", "created_at"),
    )


class FoldMetric(Base):
    """One row per walk-forward fold. The audit trail for every headline number."""

    __tablename__ = "fold_metrics"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("model_runs.id", ondelete="CASCADE"), index=True
    )
    fold_index: Mapped[int] = mapped_column(SmallInteger)

    train_start: Mapped[dt.date | None] = mapped_column(Date)
    train_end: Mapped[dt.date | None] = mapped_column(Date)
    test_start: Mapped[dt.date | None] = mapped_column(Date)
    test_end: Mapped[dt.date | None] = mapped_column(Date)
    n_train: Mapped[int | None] = mapped_column(Integer)
    n_test: Mapped[int | None] = mapped_column(Integer)

    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    mase: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    directional_accuracy: Mapped[float | None] = mapped_column(Float)
    mcc: Mapped[float | None] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float)
    brier: Mapped[float | None] = mapped_column(Float)
    #: 1 - MSE_model / MSE_baseline. The number the resume claim rests on.
    skill_vs_naive: Mapped[float | None] = mapped_column(Float)
    sharpe: Mapped[float | None] = mapped_column(Float)

    run: Mapped[ModelRun] = relationship(back_populates="folds")

    __table_args__ = (UniqueConstraint("run_id", "fold_index", name="uq_fold"),)


class RunMetric(Base):
    """Fold-aggregated metrics with confidence intervals and significance tests."""

    __tablename__ = "run_metrics"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("model_runs.id", ondelete="CASCADE"), primary_key=True
    )
    #: {"rmse": {"mean":…, "std":…, "ci_low":…, "ci_high":…}, …}
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn)
    skill_vs_naive: Mapped[float | None] = mapped_column(Float)
    #: Diebold-Mariano test of equal predictive accuracy against the baseline.
    dm_stat: Mapped[float | None] = mapped_column(Float)
    dm_pvalue: Mapped[float | None] = mapped_column(Float)
    n_folds: Mapped[int | None] = mapped_column(SmallInteger)

    run: Mapped[ModelRun] = relationship(back_populates="aggregate")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("model_runs.id", ondelete="CASCADE"), index=True
    )
    instrument_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("instruments.id", ondelete="CASCADE")
    )

    #: Last bar the model was permitted to see.
    as_of_date: Mapped[dt.date] = mapped_column(Date)
    #: The bar being predicted. Always > as_of_date.
    target_date: Mapped[dt.date] = mapped_column(Date)

    fold_index: Mapped[int | None] = mapped_column(SmallInteger)
    is_out_of_sample: Mapped[bool] = mapped_column(Boolean, default=True)

    y_pred: Mapped[float] = mapped_column(Float)
    y_pred_lower: Mapped[float | None] = mapped_column(Float)  # conformal P10
    y_pred_upper: Mapped[float | None] = mapped_column(Float)  # conformal P90
    prob_up: Mapped[float | None] = mapped_column(Float)
    #: NULL until the target bar is realised; backfilled by the nightly job.
    y_true: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("run_id", "instrument_id", "as_of_date", "target_date", name="uq_pred"),
        CheckConstraint("target_date > as_of_date", name="ck_pred_is_forward_looking"),
        Index("idx_pred_lookup", "instrument_id", "target_date"),
    )


class FeatureImportance(Base):
    __tablename__ = "feature_importance"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("model_runs.id", ondelete="CASCADE"), primary_key=True
    )
    feature_name: Mapped[str] = mapped_column(String(120), primary_key=True)
    method: Mapped[str] = mapped_column(String(20), primary_key=True)
    importance: Mapped[float] = mapped_column(Float)


# ═══════════════════════════════════════════════════════════════════════════
# Backtesting
# ═══════════════════════════════════════════════════════════════════════════
class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUIDColumn, primary_key=True, default=uuid.uuid4)
    model_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUIDColumn, ForeignKey("model_runs.id", ondelete="CASCADE"), index=True
    )
    strategy_name: Mapped[str] = mapped_column(String(60))
    config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)

    start_date: Mapped[dt.date | None] = mapped_column(Date)
    end_date: Mapped[dt.date | None] = mapped_column(Date)
    initial_capital: Mapped[float | None] = mapped_column(MoneyColumn)

    total_return: Mapped[float | None] = mapped_column(Float)
    cagr: Mapped[float | None] = mapped_column(Float)
    sharpe: Mapped[float | None] = mapped_column(Float)
    sortino: Mapped[float | None] = mapped_column(Float)
    calmar: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    volatility: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)
    profit_factor: Mapped[float | None] = mapped_column(Float)
    n_trades: Mapped[int | None] = mapped_column(Integer)
    turnover: Mapped[float | None] = mapped_column(Float)
    total_costs: Mapped[float | None] = mapped_column(MoneyColumn)

    #: Buy-and-hold over the identical window. A strategy is only interesting
    #: relative to this, so it is never optional.
    benchmark_return: Mapped[float | None] = mapped_column(Float)
    benchmark_sharpe: Mapped[float | None] = mapped_column(Float)
    alpha: Mapped[float | None] = mapped_column(Float)
    beta: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now())


class BacktestEquity(Base):
    __tablename__ = "backtest_equity"

    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("backtest_runs.id", ondelete="CASCADE"), primary_key=True
    )
    ts: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(MoneyColumn)
    cash: Mapped[float | None] = mapped_column(MoneyColumn)
    position: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    benchmark_equity: Mapped[float | None] = mapped_column(MoneyColumn)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUIDColumn, ForeignKey("backtest_runs.id", ondelete="CASCADE"), index=True
    )
    entry_date: Mapped[dt.date | None] = mapped_column(Date)
    exit_date: Mapped[dt.date | None] = mapped_column(Date)
    side: Mapped[str | None] = mapped_column(String(10))
    entry_price: Mapped[float | None] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float)
    quantity: Mapped[float | None] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(MoneyColumn)
    pnl_pct: Mapped[float | None] = mapped_column(Float)
    costs: Mapped[float | None] = mapped_column(MoneyColumn)
    exit_reason: Mapped[str | None] = mapped_column(String(40))


# ═══════════════════════════════════════════════════════════════════════════
# Ops
# ═══════════════════════════════════════════════════════════════════════════
class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(30), index=True)
    symbol: Mapped[str | None] = mapped_column(String(20), index=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    finished_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    rows_written: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(20))
    error: Mapped[str | None] = mapped_column(Text)


class ApiQuota(Base):
    """Persisted rate-limit counters so quotas survive process restarts."""

    __tablename__ = "api_quota"

    provider: Mapped[str] = mapped_column(String(30), primary_key=True)
    calls_used: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[dt.datetime] = mapped_column(TZDateTime)
    limit_per_window: Mapped[int] = mapped_column(Integer)
