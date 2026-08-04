"""SHAP-based model explanation.

Two levels:

* **Global** -- mean absolute SHAP value per feature, i.e. which inputs the
  model relies on overall.
* **Local** -- the contribution of each feature to one specific prediction,
  which is what the UI's waterfall chart renders.

SHAP is preferred over a tree's built-in ``feature_importances_`` because gain-
based importance is biased toward high-cardinality continuous features and says
nothing about *direction*. SHAP values are signed and additive: they sum to the
difference between the prediction and the base value, so a waterfall chart
actually reconciles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecaster.logging import get_logger
from forecaster.models.base import Forecaster

log = get_logger(__name__)

#: Background sample size for kernel/permutation explainers. Larger is more
#: accurate and quadratically slower; 100 is the usual practical compromise.
BACKGROUND_SAMPLES = 100


@dataclass
class ShapExplanation:
    feature_names: list[str] = field(default_factory=list)
    #: Mean |SHAP| per feature, normalised to sum to 1.
    global_importance: dict[str, float] = field(default_factory=dict)
    #: Signed mean SHAP per feature -- the average *direction* of influence.
    mean_signed: dict[str, float] = field(default_factory=dict)
    base_value: float = 0.0
    #: Per-row SHAP values, rows x features.
    values: np.ndarray | None = None
    method: str = "none"
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.global_importance)

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.global_importance.items(), key=lambda kv: -kv[1])[:n]

    def local_contributions(self, row: int = -1, n: int = 8) -> list[dict[str, Any]]:
        """Signed contributions for a single prediction, largest first."""
        if self.values is None or len(self.values) == 0:
            return []
        idx = row if row >= 0 else len(self.values) + row
        if not 0 <= idx < len(self.values):
            return []
        contributions = [
            {"feature": name, "contribution": float(value)}
            for name, value in zip(self.feature_names, self.values[idx], strict=False)
        ]
        contributions.sort(key=lambda c: -abs(c["contribution"]))
        return contributions[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "base_value": self.base_value,
            "global_importance": self.global_importance,
            "mean_signed": self.mean_signed,
            "top_features": [{"feature": f, "importance": v} for f, v in self.top_features(15)],
            "error": self.error,
        }


def explain_model(
    model: Forecaster,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    *,
    max_rows: int = 300,
) -> ShapExplanation:
    """Compute SHAP values for a fitted model.

    Falls back gracefully: TreeExplainer for tree ensembles (exact and fast),
    LinearExplainer for linear models, and permutation importance for anything
    else -- including the sequence models, where a 3-D input makes standard SHAP
    explainers inapplicable.
    """
    if not model.is_fitted:
        return ShapExplanation(error="model is not fitted")

    matrix = X.to_numpy(dtype="float64") if isinstance(X, pd.DataFrame) else np.asarray(X, dtype="float64")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if len(matrix) > max_rows:
        matrix = matrix[-max_rows:]

    estimator = getattr(model, "_estimator", None)

    if estimator is not None and model.family == "tree":
        result = _tree_shap(estimator, matrix, feature_names)
        if result.succeeded:
            return result

    if estimator is not None and model.family == "linear":
        result = _linear_shap(estimator, matrix, feature_names)
        if result.succeeded:
            return result

    return _permutation_importance(model, matrix, feature_names)


def _tree_shap(estimator: Any, matrix: np.ndarray, names: list[str]) -> ShapExplanation:
    try:
        import shap

        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix, check_additivity=False)
        if isinstance(values, list):  # multiclass
            values = values[-1]
        values = np.asarray(values, dtype="float64")
        base = float(np.mean(np.atleast_1d(explainer.expected_value)))
        return _summarise(values, base, names, "tree_shap")
    except Exception as exc:  # noqa: BLE001 -- explanation is best-effort
        log.debug("tree_shap_failed", error=str(exc))
        return ShapExplanation(error=str(exc))


def _linear_shap(estimator: Any, matrix: np.ndarray, names: list[str]) -> ShapExplanation:
    try:
        coefficients = np.asarray(estimator.coef_, dtype="float64").ravel()
        if len(coefficients) != matrix.shape[1]:
            return ShapExplanation(error="coefficient/feature count mismatch")
        # For a linear model SHAP has a closed form: phi_i = beta_i * (x_i - E[x_i]).
        centred = matrix - matrix.mean(axis=0, keepdims=True)
        values = centred * coefficients
        base = float(getattr(estimator, "intercept_", 0.0))
        return _summarise(values, base, names, "linear_shap")
    except Exception as exc:  # noqa: BLE001
        log.debug("linear_shap_failed", error=str(exc))
        return ShapExplanation(error=str(exc))


def _permutation_importance(
    model: Forecaster, matrix: np.ndarray, names: list[str], n_repeats: int = 5
) -> ShapExplanation:
    """Model-agnostic fallback.

    Shuffle one column at a time and measure how much the predictions move.
    Works for anything with a ``predict``, including neural sequence models.
    Gives magnitudes but not signed per-row contributions, so ``values`` stays
    None and the UI hides the waterfall for these models rather than faking it.
    """
    try:
        baseline = model.predict(matrix)
        rng = np.random.default_rng(42)
        scores: dict[str, float] = {}

        for j, name in enumerate(names[: matrix.shape[1]]):
            deltas = []
            for _ in range(n_repeats):
                permuted = matrix.copy()
                permuted[:, j] = rng.permutation(permuted[:, j])
                deltas.append(float(np.mean(np.abs(model.predict(permuted) - baseline))))
            scores[name] = float(np.mean(deltas))

        total = sum(scores.values())
        normalised = (
            {k: v / total for k, v in scores.items()} if total > 0 else dict.fromkeys(scores, 0.0)
        )
        return ShapExplanation(
            feature_names=list(names),
            global_importance=normalised,
            mean_signed={},
            base_value=float(np.mean(baseline)),
            values=None,
            method="permutation",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("permutation_importance_failed", error=str(exc))
        return ShapExplanation(error=str(exc))


def _summarise(
    values: np.ndarray, base: float, names: list[str], method: str
) -> ShapExplanation:
    if values.ndim != 2 or values.shape[1] != len(names):
        return ShapExplanation(error=f"unexpected SHAP shape {values.shape}")

    magnitude = np.abs(values).mean(axis=0)
    total = float(magnitude.sum())
    importance = (
        {n: float(v / total) for n, v in zip(names, magnitude, strict=True)}
        if total > 0
        else dict.fromkeys(names, 0.0)
    )
    signed = {n: float(v) for n, v in zip(names, values.mean(axis=0), strict=True)}

    return ShapExplanation(
        feature_names=list(names),
        global_importance=importance,
        mean_signed=signed,
        base_value=base,
        values=values,
        method=method,
    )
