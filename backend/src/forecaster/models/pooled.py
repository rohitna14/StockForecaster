"""Cross-sectional (pooled) forecasting for stocks with little or no history.

**The problem.** A company that listed five weeks ago has ~30 bars. You cannot
walk-forward validate that: there is no past to train on and no future to test
against. Per-stock modelling simply does not apply.

**Why it is solvable anyway.** The thing being predicted -- volatility
clustering -- is not a property of *a company*. It is a property of *financial
price series*. Calm follows calm and turbulence follows turbulence in Apple, in
a 2026 IPO, and in wheat futures. So the mapping from "recent return behaviour"
to "next period's volatility" can be learned from hundreds of stocks and applied
to one the model has never seen.

This is standard practice for new listings, and it is honestly measurable: hold
out *entire symbols*, train on the rest, and score on the held-out ones. That
answers exactly the question a new IPO poses -- "how well does this work on a
stock the model has never seen?" -- rather than the easier question a random
split would answer.

**What it cannot do.** It cannot learn anything company-specific: that this
issuer has a jumpy earnings pattern, or that its float is tiny. Expect skill
somewhat below a well-fitted per-stock model, and the harness labels it as
lower-confidence rather than presenting it as equivalent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecaster.exceptions import InsufficientDataError
from forecaster.logging import get_logger
from forecaster.models.base import Forecaster, ModelContext

log = get_logger(__name__)

#: Bars a target symbol needs before the pooled model will produce a forecast.
#: The cold-start feature set warms up in 21, so this leaves a few valid rows.
MIN_TARGET_BARS = 25

#: Symbols required in the training pool for the cross-sectional fit to be
#: meaningful rather than a handful of idiosyncratic series.
MIN_POOL_SYMBOLS = 6


@dataclass
class PooledTrainingData:
    """Stacked feature/target matrix with symbol labels for group-aware splits."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # symbol per row
    feature_names: list[str]
    symbols: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_symbols(self) -> int:
        return len(set(self.groups.tolist()))


def build_pool(
    frames: dict[str, pd.DataFrame],
    *,
    feature_set: str = "cold_start",
    horizon: int = 5,
    target_type: str = "vol_ratio",
) -> PooledTrainingData:
    """Stack many symbols into one training matrix.

    Every feature in the cold-start set is scale-free (returns, ratios, bounded
    oscillators), which is what makes pooling valid at all -- a $3 stock and a
    $3,000 stock produce comparable rows. Stacking raw prices would just teach
    the model which company it was looking at.
    """
    from forecaster import features as F
    from forecaster.features.targets import build_target

    names = F.feature_set(feature_set)
    blocks_X: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    blocks_g: list[np.ndarray] = []
    used: list[str] = []
    columns: list[str] = []

    for symbol, frame in frames.items():
        if frame is None or len(frame) < 60:
            continue
        try:
            features = F.build(frame, names)
            target = build_target(frame, target_type, horizon)
        except Exception as exc:  # noqa: BLE001 -- one bad symbol must not stop the pool
            log.debug("pool_symbol_skipped", symbol=symbol, error=str(exc))
            continue

        combined = features.join(target.rename("__y__")).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(combined) < 30:
            continue

        if not columns:
            columns = [c for c in combined.columns if c != "__y__"]

        blocks_X.append(combined[columns].to_numpy(dtype="float64"))
        blocks_y.append(combined["__y__"].to_numpy(dtype="float64"))
        blocks_g.append(np.full(len(combined), symbol, dtype=object))
        used.append(symbol)

    if len(used) < MIN_POOL_SYMBOLS:
        raise InsufficientDataError(
            f"Pooled training needs at least {MIN_POOL_SYMBOLS} symbols with "
            f"history; only {len(used)} were usable.",
            symbols_available=len(used),
        )

    return PooledTrainingData(
        X=np.vstack(blocks_X),
        y=np.concatenate(blocks_y),
        groups=np.concatenate(blocks_g),
        feature_names=columns,
        symbols=used,
    )


class PooledForecaster(Forecaster):
    """Gradient-boosted model fitted across many symbols at once."""

    name = "pooled_lightgbm"
    display_name = "Pooled (cross-sectional)"
    family = "pooled"
    requires_scaling = False

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self._estimator: Any = None
        self._fallback = 0.0

    def _build(self) -> Any:
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=self.hyperparams.get("n_estimators", 400),
                learning_rate=self.hyperparams.get("learning_rate", 0.03),
                max_depth=self.hyperparams.get("max_depth", 5),
                num_leaves=self.hyperparams.get("num_leaves", 24),
                min_child_samples=self.hyperparams.get("min_child_samples", 60),
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.hyperparams.get("random_state", 42),
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_depth=5, random_state=42
            )

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean, y_clean = X[mask], y[mask]
        if len(X_clean) < 100:
            raise InsufficientDataError(
                f"Pooled fit needs >=100 clean rows, got {len(X_clean)}"
            )
        self._fallback = float(np.median(y_clean))
        self._estimator = self._build()
        self._estimator.fit(X_clean, y_clean)

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        if self._estimator is None:
            return np.full(len(X), self._fallback, dtype="float64")
        clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(self._estimator.predict(clean), dtype="float64")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        if self._estimator is None or not hasattr(self._estimator, "feature_importances_"):
            return None
        values = np.asarray(self._estimator.feature_importances_, dtype="float64")
        if len(values) != len(feature_names):
            return None
        total = float(values.sum())
        if total <= 0:
            return dict.fromkeys(feature_names, 0.0)
        return {n: float(v / total) for n, v in zip(feature_names, values, strict=True)}


@dataclass
class TransferResult:
    """Leave-one-symbol-out skill: how well this transfers to an unseen stock."""

    n_symbols: int
    n_rows: int
    rmse_skill_pct: float
    mae_skill_pct: float
    r2: float
    per_symbol: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def symbols_positive(self) -> int:
        return sum(1 for v in self.per_symbol.values() if v > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "leave_one_symbol_out",
            "n_symbols": self.n_symbols,
            "n_rows": self.n_rows,
            "rmse_skill_pct": self.rmse_skill_pct,
            "mae_skill_pct": self.mae_skill_pct,
            "r2": self.r2,
            "symbols_positive": self.symbols_positive,
            "per_symbol": self.per_symbol,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def validate_transfer(
    pool: PooledTrainingData, *, max_symbols: int | None = None
) -> TransferResult:
    """Leave-one-symbol-out validation.

    Hold out every row of one symbol, train on the rest, score the held-out
    symbol. Repeat. This is the only honest way to state expected accuracy for a
    stock the model has never seen -- a random row split would leak, because
    rows from the same symbol are highly correlated and adjacent rows share
    almost all their feature history.
    """
    from forecaster.validation.metrics import mae, r2 as r2_score, rmse

    started = time.perf_counter()
    symbols = sorted(set(pool.groups.tolist()))
    if max_symbols:
        symbols = symbols[:max_symbols]

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_symbol: dict[str, float] = {}

    for symbol in symbols:
        held_out = pool.groups == symbol
        if held_out.sum() < 20 or (~held_out).sum() < 200:
            continue

        model = PooledForecaster()
        try:
            model.fit(pool.X[~held_out], pool.y[~held_out])
            predictions = model.predict(pool.X[held_out])
        except Exception as exc:  # noqa: BLE001
            log.debug("transfer_fold_failed", symbol=symbol, error=str(exc))
            continue

        truth = pool.y[held_out]
        all_true.append(truth)
        all_pred.append(predictions)

        # Baseline for vol_ratio is 0 ("volatility persists").
        baseline_rmse = rmse(truth, np.zeros_like(truth))
        model_rmse = rmse(truth, predictions)
        if np.isfinite(baseline_rmse) and baseline_rmse > 0:
            per_symbol[symbol] = float((1 - model_rmse / baseline_rmse) * 100)

    if not all_true:
        raise InsufficientDataError("Leave-one-symbol-out produced no usable folds")

    truth = np.concatenate(all_true)
    predictions = np.concatenate(all_pred)
    zeros = np.zeros_like(truth)

    return TransferResult(
        n_symbols=len(per_symbol),
        n_rows=len(truth),
        rmse_skill_pct=float((1 - rmse(truth, predictions) / rmse(truth, zeros)) * 100),
        mae_skill_pct=float((1 - mae(truth, predictions) / mae(truth, zeros)) * 100),
        r2=float(r2_score(truth, predictions)),
        per_symbol=per_symbol,
        duration_seconds=time.perf_counter() - started,
    )
