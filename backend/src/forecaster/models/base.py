"""Forecaster interface.

Every model -- naive baseline, gradient-boosted tree, LSTM -- implements this
one interface, which is what lets the harness treat them identically. The
baselines are not special-cased anywhere: they are fitted, scored and ranked
through exactly the same code path as everything else, so a leaderboard where
a tree model loses to ``naive_last_price`` is impossible to hide.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd

from forecaster.exceptions import NotFittedError


@dataclass(frozen=True)
class ModelContext:
    """Series a model may need beyond the feature matrix.

    Naive baselines forecast from the price path itself rather than from
    engineered features, so they need the close series. Passing it explicitly
    keeps price out of the feature matrix, where it would let every other model
    memorise ticker-specific levels.

    Fields:
        close: close prices aligned to the rows being fitted or predicted.
        history: full close series up to the start of this window, for
            lookbacks reaching further back than the window itself.
        realized: **backtest only.** Target values for the prediction window,
            used exclusively by rolling-refit models (ARIMA/SARIMA) to update
            their filter with bars that have *already happened* as the walk
            proceeds. At prediction step ``i`` a model may consume
            ``realized[:i]`` and never ``realized[i]``; that invariant is
            enforced by ``tests/unit/test_models.py::test_rolling_refit_is_causal``.
            ``None`` in live inference, where nothing has realised yet.
        horizon: forecast horizon in bars.
    """

    close: pd.Series | None = None
    history: pd.Series | None = None
    realized: np.ndarray | None = None
    horizon: int = 1


class Forecaster(abc.ABC):
    """Base class for all models."""

    #: Registry key, also written to ``model_runs.model_name``.
    name: str = "forecaster"
    #: Human-readable label for the UI.
    display_name: str = "Forecaster"
    #: Grouping for the leaderboard: baseline | linear | tree | deep | classical | ensemble
    family: str = "baseline"
    #: Whether ``predict`` returns class labels/probabilities rather than a value.
    is_classifier: bool = False
    #: Whether the harness should standardise features before fitting.
    requires_scaling: bool = False
    #: Whether the model consumes 3-D (samples, lookback, features) input.
    is_sequence_model: bool = False

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams = hyperparams
        self._fitted = False

    # ── lifecycle ─────────────────────────────────────────────────────────
    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None: ...

    @abc.abstractmethod
    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray: ...

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        context: ModelContext | None = None,
    ) -> Self:
        X_arr = _to_array(X)
        y_arr = _to_array(y).ravel()
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X has {len(X_arr)} rows but y has {len(y_arr)}")
        self._fit(X_arr, y_arr, context)
        self._fitted = True
        return self

    def predict(
        self, X: np.ndarray | pd.DataFrame, context: ModelContext | None = None
    ) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError(f"{self.name} has not been fitted")
        out = np.asarray(self._predict(_to_array(X), context), dtype="float64").ravel()
        # A model returning NaN silently corrupts every downstream metric.
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame, context: ModelContext | None = None
    ) -> np.ndarray | None:
        """Probability of the positive class, or None for regressors."""
        return None

    # ── introspection ─────────────────────────────────────────────────────
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        """Native importances when the model exposes them."""
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "family": self.family,
            "is_classifier": self.is_classifier,
            "requires_scaling": self.requires_scaling,
            "hyperparams": self.hyperparams,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} fitted={self._fitted}>"


def _to_array(data: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy(dtype="float64")
    return np.asarray(data, dtype="float64")
