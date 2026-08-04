"""Adapter wrapping any scikit-learn-style estimator as a :class:`Forecaster`.

Keeps the concrete model classes to a few declarative lines each and guarantees
they all handle NaNs, scaling flags and importance extraction identically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from forecaster.models.base import Forecaster, ModelContext


class SklearnForecaster(Forecaster):
    """Wraps an estimator exposing ``fit`` / ``predict``.

    Subclasses supply :meth:`_build`, which is called fresh for every fold --
    refitting a previously fitted estimator is a subtle way to carry state
    across folds.
    """

    #: Default hyperparameters, overridden by anything passed to __init__.
    defaults: dict[str, Any] = {}  # noqa: RUF012

    def __init__(self, **hyperparams: Any) -> None:
        merged = {**self.defaults, **hyperparams}
        super().__init__(**merged)
        self._estimator: Any = None

    def _build(self) -> Any:
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        X_clean, y_clean = _drop_nonfinite(X, y)
        if len(X_clean) == 0:
            raise ValueError(f"{self.name}: no finite training rows")
        self._estimator = self._build()
        self._estimator.fit(X_clean, y_clean)

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        return np.asarray(self._estimator.predict(_impute(X)), dtype="float64")

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame, context: ModelContext | None = None
    ) -> np.ndarray | None:
        if not self.is_classifier or self._estimator is None:
            return None
        proba_fn = getattr(self._estimator, "predict_proba", None)
        if proba_fn is None:
            return None
        arr = np.asarray(X, dtype="float64") if not isinstance(X, pd.DataFrame) else X.to_numpy()
        proba = np.asarray(proba_fn(_impute(arr)), dtype="float64")
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()

    def feature_importance(self, feature_names: list[str]) -> dict[str, float] | None:
        if self._estimator is None:
            return None
        values: np.ndarray | None = None
        if hasattr(self._estimator, "feature_importances_"):
            values = np.asarray(self._estimator.feature_importances_, dtype="float64")
        elif hasattr(self._estimator, "coef_"):
            values = np.abs(np.asarray(self._estimator.coef_, dtype="float64")).ravel()
        if values is None or len(values) != len(feature_names):
            return None
        total = float(values.sum())
        if total <= 0:
            return dict.fromkeys(feature_names, 0.0)
        return {name: float(v / total) for name, v in zip(feature_names, values, strict=True)}


def _drop_nonfinite(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _impute(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values with 0.

    Features are standardised before reaching a model, so 0 is the column mean
    -- the least-information substitute. Tree models tolerate it as a split
    point like any other value.
    """
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def make_sklearn_model(
    *,
    name: str,
    display_name: str,
    family: str,
    builder: Callable[[dict[str, Any]], Any],
    defaults: dict[str, Any] | None = None,
    requires_scaling: bool = False,
    is_classifier: bool = False,
) -> type[SklearnForecaster]:
    """Factory producing a concrete :class:`SklearnForecaster` subclass."""

    class _Model(SklearnForecaster):
        pass

    _Model.name = name
    _Model.display_name = display_name
    _Model.family = family
    _Model.requires_scaling = requires_scaling
    _Model.is_classifier = is_classifier
    _Model.defaults = dict(defaults or {})
    _Model._build = lambda self: builder(self.hyperparams)  # type: ignore[assignment,method-assign]
    _Model.__name__ = "".join(part.title() for part in name.split("_")) + "Forecaster"
    _Model.__qualname__ = _Model.__name__
    return _Model
