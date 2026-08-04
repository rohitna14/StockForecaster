"""Evaluation harness.

Runs ``models x folds`` under strict walk-forward discipline and produces every
number the project publishes. The rules it enforces, none of which the
prototype did:

* The reference baseline is evaluated on **every fold, always**, so a skill
  score always has a denominator computed on identical data.
* Preprocessing is fitted **inside** each fold on training rows only.
* Model selection never touches test data. The leaderboard is a *report*, not a
  selection step; picking the max is explicitly not how a production model gets
  chosen (see :func:`select_model_honestly`).
* Every fold is re-checked for leakage before it is used.
* Aggregation across folds carries block-bootstrap confidence intervals and a
  Diebold-Mariano test against the baseline, so "better" is qualified.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecaster import features as F
from forecaster.exceptions import InsufficientDataError
from forecaster.features.targets import TargetType, build_target, is_classification
from forecaster.logging import get_logger, log_context
from forecaster.models.base import ModelContext
from forecaster.models.baselines import REFERENCE_BASELINE
from forecaster.models.registry import compatible_models, create, model_set
from forecaster.validation import metrics as M
from forecaster.validation.skill import all_skill_scores, rmse_skill
from forecaster.validation.splitters import Fold, SplitMode, WalkForwardSplit
from forecaster.validation.stats import block_bootstrap_ci, diebold_mariano

log = get_logger(__name__)


# ═══════════════════════════════ configuration ═════════════════════════════
@dataclass
class EvaluationConfig:
    symbol: str
    models: list[str] = field(default_factory=lambda: model_set("fast"))
    horizon: int = 1
    target_type: TargetType = TargetType.RETURN
    feature_set: str = "core"

    # Walk-forward geometry
    train_size: int = 504  # ~2 years
    test_size: int = 63  # ~1 quarter
    step: int | None = None
    embargo: int = 2
    mode: SplitMode = SplitMode.ROLLING
    max_folds: int | None = None

    # Evaluation options
    cost_bps: float = 5.0  # round-trip cost assumption for strategy metrics
    conformal_alpha: float = 0.2  # 0.2 -> 80% prediction interval (P10-P90)
    random_seed: int = 42

    def splitter(self) -> WalkForwardSplit:
        return WalkForwardSplit(
            train_size=self.train_size,
            test_size=self.test_size,
            step=self.step,
            horizon=self.horizon,
            embargo=self.embargo,
            mode=self.mode,
            max_folds=self.max_folds,
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["target_type"] = str(self.target_type)
        out["mode"] = str(self.mode)
        return out


# ═══════════════════════════════ results ═══════════════════════════════════
@dataclass
class FoldResult:
    fold_index: int
    metrics: M.MetricSet
    y_true: np.ndarray
    y_pred: np.ndarray
    y_lower: np.ndarray | None
    y_upper: np.ndarray | None
    probabilities: np.ndarray | None
    dates: pd.DatetimeIndex
    train_start: Any = None
    train_end: Any = None
    test_start: Any = None
    test_end: Any = None
    n_train: int = 0
    n_test: int = 0


@dataclass
class ModelResult:
    model_name: str
    display_name: str
    family: str
    folds: list[FoldResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    skill: dict[str, Any] = field(default_factory=dict)
    dm_test: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: str | None = None
    interval_coverage: float = float("nan")

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.folds)

    def pooled(self) -> tuple[np.ndarray, np.ndarray]:
        """All out-of-sample predictions concatenated across folds."""
        if not self.folds:
            return np.array([]), np.array([])
        return (
            np.concatenate([f.y_true for f in self.folds]),
            np.concatenate([f.y_pred for f in self.folds]),
        )

    def metric_series(self, name: str) -> np.ndarray:
        return np.array([getattr(f.metrics, name) for f in self.folds], dtype="float64")


@dataclass
class EvaluationReport:
    config: EvaluationConfig
    results: dict[str, ModelResult] = field(default_factory=dict)
    n_folds: int = 0
    n_samples: int = 0
    feature_names: list[str] = field(default_factory=list)
    fold_table: pd.DataFrame | None = None
    duration_seconds: float = 0.0

    def leaderboard(self, sort_by: str = "rmse") -> pd.DataFrame:
        """Ranked table. Baselines are included and cannot be filtered out."""
        rows = []
        for result in self.results.values():
            if not result.succeeded:
                rows.append(
                    {"model": result.model_name, "family": result.family, "error": result.error}
                )
                continue
            agg = result.aggregate
            rows.append(
                {
                    "model": result.model_name,
                    "display_name": result.display_name,
                    "family": result.family,
                    "is_baseline": result.family == "baseline",
                    "rmse": agg.get("rmse", {}).get("mean"),
                    "mae": agg.get("mae", {}).get("mean"),
                    "mase": agg.get("mase", {}).get("mean"),
                    "r2": agg.get("r2", {}).get("mean"),
                    "hit_rate": agg.get("directional_accuracy", {}).get("mean"),
                    "hit_rate_ci_low": agg.get("directional_accuracy", {}).get("ci_low"),
                    "hit_rate_ci_high": agg.get("directional_accuracy", {}).get("ci_high"),
                    "call_coverage": agg.get("directional_coverage", {}).get("mean"),
                    "base_rate": agg.get("base_rate", {}).get("mean"),
                    "mcc": agg.get("mcc", {}).get("mean"),
                    "sharpe": agg.get("sharpe", {}).get("mean"),
                    "rmse_skill_pct": result.skill.get("rmse", {}).get("improvement_pct"),
                    "directional_skill_pct": result.skill.get("directional", {}).get(
                        "improvement_pct"
                    ),
                    "dm_pvalue": result.dm_test.get("dm_pvalue"),
                    "interval_coverage": result.interval_coverage,
                    "seconds": round(result.duration_seconds, 2),
                }
            )
        frame = pd.DataFrame(rows)
        if sort_by in frame.columns and not frame.empty:
            ascending = sort_by in {"rmse", "mae", "mase", "dm_pvalue"}
            frame = frame.sort_values(sort_by, ascending=ascending, na_position="last")
        return frame.reset_index(drop=True)

    @property
    def baseline(self) -> ModelResult | None:
        return self.results.get(REFERENCE_BASELINE)


# ═══════════════════════════════ the harness ═══════════════════════════════
class EvaluationHarness:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def prepare(self, ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build the aligned (features, target, close) triple.

        Alignment happens once, here, on the intersection of valid feature rows
        and valid target rows. Every downstream index is positional into these
        three objects, which removes an entire class of off-by-one bugs.
        """
        names = F.feature_set(self.config.feature_set)
        X = F.build(ohlcv, names)
        y = build_target(ohlcv, self.config.target_type, self.config.horizon)

        combined = X.join(y.rename("__target__")).join(ohlcv["close"].rename("__close__"))
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        if combined.empty:
            raise InsufficientDataError(
                f"No usable rows for {self.config.symbol} after feature warm-up "
                f"({F.required_history(names)} bars) and target alignment",
                symbol=self.config.symbol,
            )

        features = combined.drop(columns=["__target__", "__close__"])
        return features, combined["__target__"], combined["__close__"]

    def run(self, ohlcv: pd.DataFrame, *, progress: Any = None) -> EvaluationReport:
        started = time.perf_counter()
        cfg = self.config
        np.random.seed(cfg.random_seed)

        features, target, close = self.prepare(ohlcv)
        splitter = cfg.splitter()
        folds = splitter.split(features)

        # Every fold re-validated even though the splitter already checked.
        for fold in folds:
            fold.assert_no_leakage(cfg.horizon)

        model_names = compatible_models(
            list(dict.fromkeys([REFERENCE_BASELINE, *cfg.models])),
            classification=is_classification(cfg.target_type),
        )
        # The reference baseline must be evaluated first and always -- every
        # other model's skill score uses it as the denominator.
        if REFERENCE_BASELINE in model_names:
            model_names.remove(REFERENCE_BASELINE)
        model_names.insert(0, REFERENCE_BASELINE)

        log.info(
            "evaluation_started",
            symbol=cfg.symbol,
            models=len(model_names),
            folds=len(folds),
            samples=len(features),
            features=features.shape[1],
            horizon=cfg.horizon,
        )

        report = EvaluationReport(
            config=cfg,
            n_folds=len(folds),
            n_samples=len(features),
            feature_names=list(features.columns),
            fold_table=pd.DataFrame([f.as_dict() for f in folds]),
        )

        # Baseline first -- every other model's skill score needs it.
        baseline_preds: dict[int, np.ndarray] = {}

        for position, name in enumerate(model_names):
            with log_context(model=name):
                result = self._evaluate_model(name, features, target, close, folds, baseline_preds)
            report.results[name] = result

            if name == REFERENCE_BASELINE and result.succeeded:
                baseline_preds = {f.fold_index: f.y_pred for f in result.folds}

            if progress is not None and hasattr(progress, "advance"):
                progress.advance(1)
            log.info(
                "model_evaluated",
                model=name,
                ok=result.succeeded,
                rmse=result.aggregate.get("rmse", {}).get("mean"),
                skill_pct=result.skill.get("rmse", {}).get("improvement_pct"),
                seconds=round(result.duration_seconds, 2),
            )

        report.duration_seconds = time.perf_counter() - started
        return report

    # ── per-model evaluation ──────────────────────────────────────────────
    def _evaluate_model(
        self,
        name: str,
        features: pd.DataFrame,
        target: pd.Series,
        close: pd.Series,
        folds: list[Fold],
        baseline_preds: dict[int, np.ndarray],
    ) -> ModelResult:
        cfg = self.config
        started = time.perf_counter()

        try:
            prototype = create(name)
        except Exception as exc:  # noqa: BLE001
            return ModelResult(name, name, "unknown", error=f"instantiation failed: {exc}")

        result = ModelResult(
            model_name=name,
            display_name=prototype.display_name,
            family=prototype.family,
        )

        X_all = features.to_numpy(dtype="float64")
        y_all = target.to_numpy(dtype="float64")
        importances: list[dict[str, float]] = []
        coverage_hits: list[float] = []

        for fold in folds:
            try:
                fold_result = self._run_fold(
                    name, fold, X_all, y_all, close, features.index, features.columns.tolist()
                )
            except Exception as exc:  # noqa: BLE001 -- one bad fold must not kill the model
                log.warning("fold_failed", model=name, fold=fold.index, error=str(exc))
                continue

            fold_res, importance, coverage = fold_result
            result.folds.append(fold_res)
            if importance:
                importances.append(importance)
            if np.isfinite(coverage):
                coverage_hits.append(coverage)

        if not result.folds:
            result.error = "all folds failed"
            result.duration_seconds = time.perf_counter() - started
            return result

        result.aggregate = self._aggregate(result)
        result.interval_coverage = float(np.mean(coverage_hits)) if coverage_hits else float("nan")

        if importances:
            keys = importances[0].keys()
            result.feature_importance = {
                k: float(np.mean([imp.get(k, 0.0) for imp in importances])) for k in keys
            }

        # Skill and significance versus the reference baseline.
        if baseline_preds and name != REFERENCE_BASELINE:
            y_true_all, y_pred_all, y_base_all = [], [], []
            for fold_res in result.folds:
                base = baseline_preds.get(fold_res.fold_index)
                if base is None or len(base) != len(fold_res.y_true):
                    continue
                y_true_all.append(fold_res.y_true)
                y_pred_all.append(fold_res.y_pred)
                y_base_all.append(base)

            if y_true_all:
                yt = np.concatenate(y_true_all)
                yp = np.concatenate(y_pred_all)
                yb = np.concatenate(y_base_all)
                result.skill = {k: v.as_dict() for k, v in all_skill_scores(yt, yp, yb).items()}
                result.dm_test = diebold_mariano(yt, yp, yb, horizon=cfg.horizon).as_dict()
        elif name == REFERENCE_BASELINE:
            result.skill = {
                "rmse": {
                    "improvement_pct": 0.0,
                    "skill": 0.0,
                    "metric": "rmse",
                    "baseline": REFERENCE_BASELINE,
                },
                "directional": {
                    "improvement_pct": 0.0,
                    "skill": 0.0,
                    "metric": "directional_accuracy",
                    "baseline": REFERENCE_BASELINE,
                },
            }

        result.duration_seconds = time.perf_counter() - started
        return result

    def _run_fold(
        self,
        model_name: str,
        fold: Fold,
        X_all: np.ndarray,
        y_all: np.ndarray,
        close: pd.Series,
        index: pd.Index,
        feature_names: list[str],
    ) -> tuple[FoldResult, dict[str, float] | None, float]:
        """Fit and score one fold. All preprocessing happens inside."""
        cfg = self.config
        model = create(model_name)

        tr, te = fold.train_idx, fold.test_idx
        X_train_raw, y_train = X_all[tr], y_all[tr]
        X_test_raw, y_test = X_all[te], y_all[te]

        # ── split-conformal calibration split ────────────────────────────
        # Proper training set / calibration set. The model is fitted on the
        # proper set only; residual quantiles from the untouched calibration
        # set give intervals with finite-sample coverage guarantees. This costs
        # ~20% of training rows, which is the honest price of a calibrated
        # interval rather than a fabricated one.
        n_calib = int(min(max(30, 0.2 * len(tr)), 250))
        use_conformal = cfg.conformal_alpha > 0 and len(tr) - n_calib >= 100

        if use_conformal:
            fit_slice = slice(0, len(tr) - n_calib)
            calib_slice = slice(len(tr) - n_calib, len(tr))
        else:
            fit_slice = slice(0, len(tr))
            calib_slice = None

        scaler = None
        if model.requires_scaling:
            from sklearn.preprocessing import StandardScaler

            # Fitted on the proper training rows ONLY -- never on calib or test.
            scaler = StandardScaler().fit(np.nan_to_num(X_train_raw[fit_slice]))

        def _prep(matrix: np.ndarray) -> np.ndarray:
            clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            return scaler.transform(clean) if scaler is not None else clean

        train_close = close.iloc[tr]
        fit_context = ModelContext(
            close=train_close.iloc[fit_slice],
            history=close.iloc[: int(tr.max()) + 1],
            horizon=cfg.horizon,
        )
        model.fit(_prep(X_train_raw[fit_slice]), y_train[fit_slice], fit_context)

        # ── prediction ────────────────────────────────────────────────────
        predict_context = ModelContext(
            close=close.iloc[te],
            history=close.iloc[: int(tr.max()) + 1],
            realized=y_test,
            horizon=cfg.horizon,
        )
        y_pred = model.predict(_prep(X_test_raw), predict_context)
        proba = model.predict_proba(_prep(X_test_raw), predict_context)

        # ── conformal intervals ───────────────────────────────────────────
        y_lower = y_upper = None
        coverage = float("nan")
        if use_conformal and calib_slice is not None:
            calib_pred = model.predict(
                _prep(X_train_raw[calib_slice]), ModelContext(horizon=cfg.horizon)
            )
            residuals = np.abs(y_train[calib_slice] - calib_pred)
            residuals = residuals[np.isfinite(residuals)]
            if len(residuals) >= 20:
                # Finite-sample conformal quantile.
                k = int(np.ceil((len(residuals) + 1) * (1 - cfg.conformal_alpha)))
                k = min(k, len(residuals))
                q = float(np.sort(residuals)[k - 1])
                y_lower, y_upper = y_pred - q, y_pred + q
                inside = (y_test >= y_lower) & (y_test <= y_upper)
                coverage = float(np.mean(inside[np.isfinite(y_test)]))

        fold_metrics = M.compute_all(y_test, y_pred, probabilities=proba, cost_bps=cfg.cost_bps)
        if model_name != REFERENCE_BASELINE:
            fold_metrics.skill_vs_naive = rmse_skill(y_test, y_pred, np.zeros_like(y_pred)).skill

        fold_result = FoldResult(
            fold_index=fold.index,
            metrics=fold_metrics,
            y_true=y_test,
            y_pred=y_pred,
            y_lower=y_lower,
            y_upper=y_upper,
            probabilities=proba,
            dates=pd.DatetimeIndex(index[te]),
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            n_train=fold.n_train,
            n_test=fold.n_test,
        )
        return fold_result, model.feature_importance(feature_names), coverage

    def _aggregate(self, result: ModelResult) -> dict[str, Any]:
        """Mean + block-bootstrap CI for each metric across folds."""
        out: dict[str, Any] = {}
        for metric_name in M.MetricSet().as_dict():
            if metric_name == "n":
                continue
            values = result.metric_series(metric_name)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                out[metric_name] = {"mean": None, "std": None, "ci_low": None, "ci_high": None}
                continue
            ci = block_bootstrap_ci(finite, seed=self.config.random_seed)
            out[metric_name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                "median": float(np.median(finite)),
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "n_folds": int(len(finite)),
            }
        return out


def select_model_honestly(report: EvaluationReport, metric: str = "rmse") -> str:
    """Choose a model without peeking at the final test period.

    Selecting ``argmax(metric)`` over the same folds you then report is how the
    prototype crowned a "champion" -- with fifteen models and ~100 test days,
    the winner is whichever one got luckiest. This instead selects on all folds
    **except the last**, and the held-out final fold gives an unbiased estimate
    of the chosen model's performance.

    Returns the selected model name.
    """
    candidates = {
        name: res for name, res in report.results.items() if res.succeeded and len(res.folds) >= 2
    }
    if not candidates:
        return REFERENCE_BASELINE

    scored: dict[str, float] = {}
    for name, res in candidates.items():
        values = res.metric_series(metric)[:-1]  # exclude the final fold
        finite = values[np.isfinite(values)]
        if len(finite):
            scored[name] = float(finite.mean())

    if not scored:
        return REFERENCE_BASELINE

    lower_is_better = metric in {"rmse", "mae", "mase", "brier"}
    return (
        min(scored, key=lambda k: scored[k])
        if lower_is_better
        else max(scored, key=lambda k: scored[k])
    )


def evaluate_symbol(
    ohlcv: pd.DataFrame, config: EvaluationConfig, **kwargs: Any
) -> EvaluationReport:
    """Convenience wrapper."""
    return EvaluationHarness(config).run(ohlcv, **kwargs)
