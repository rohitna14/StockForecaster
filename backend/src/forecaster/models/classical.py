"""Classical time-series models (ARIMA, SARIMAX, ETS).

**What was wrong before.** The prototype fitted ARIMA once on the training data
and then called ``forecast(steps=len(test))`` -- a single static forecast up to
250 bars ahead. An ARIMA forecast converges to the unconditional mean within a
few steps, so beyond the first handful of bars it emitted a flat line. Worse,
the evaluation guarded on ``len(forecast) == len(test_prices) - 1``, which was
essentially never true, so ``predictions`` stayed empty, the block was skipped,
and the exception was swallowed. ARIMA never actually scored anything.

**What happens here.** A genuine rolling one-step-ahead forecast. At each test
bar the model is *updated* with the actual observations that have since been
realised, then asked for exactly one step. That is how the model would be
operated live, and it is the only comparison against other one-step models that
means anything.

``refit=False`` on the update path applies the Kalman filter to the new
observations without re-estimating coefficients. Refitting from scratch at every
bar would be more accurate in principle and roughly 200x slower; coefficients
are re-estimated once per fold, which is the standard compromise.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from forecaster.logging import get_logger
from forecaster.models.base import Forecaster, ModelContext

log = get_logger(__name__)


class ClassicalForecaster(Forecaster):
    """Base for statsmodels state-space models operating on the return series."""

    family = "classical"

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self._result: Any = None
        self._train_endog: np.ndarray = np.array([])
        self._fallback: float = 0.0

    def _build(self, endog: np.ndarray) -> Any:
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        endog = y[np.isfinite(y)]
        self._fallback = float(endog.mean()) if len(endog) else 0.0
        self._train_endog = endog
        self._result = None

        if len(endog) < 60:
            log.debug("classical_insufficient_history", model=self.name, n=len(endog))
            return

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._result = self._build(endog).fit(disp=False)
        except Exception as exc:  # noqa: BLE001 -- convergence failure is expected sometimes
            log.warning("classical_fit_failed", model=self.name, error=str(exc))
            self._result = None

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        n = len(X)
        if self._result is None:
            return np.full(n, self._fallback, dtype="float64")

        actuals = self._test_actuals(context, n)

        # No actuals available (pure future forecast): fall back to a multi-step
        # forecast, which is the honest thing to do when nothing has realised.
        if actuals is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return np.asarray(self._result.forecast(steps=n), dtype="float64")
            except Exception:  # noqa: BLE001
                return np.full(n, self._fallback, dtype="float64")

        preds = np.empty(n, dtype="float64")
        state = self._result

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(n):
                    preds[i] = float(np.asarray(state.forecast(steps=1))[0])
                    # Feed in the bar that has now been realised, without
                    # re-estimating coefficients.
                    observed = actuals[i]
                    if np.isfinite(observed):
                        state = state.append([observed], refit=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("classical_rolling_failed", model=self.name, error=str(exc), at=i)
            preds[i:] = self._fallback

        return np.nan_to_num(preds, nan=self._fallback)

    @staticmethod
    def _test_actuals(context: ModelContext | None, n: int) -> np.ndarray | None:
        """Realised target values for the test window, if the harness supplied them.

        Used only to *update the filter with the past*. Look at the loop in
        :meth:`_predict`: ``preds[i]`` is produced **before** ``actuals[i]`` is
        appended, so at step ``i`` the model has consumed bars ``0..i-1`` only.
        This is exactly how the model would run live, where each day's close
        becomes available after that day's forecast was made.
        """
        if context is None or context.realized is None:
            return None
        values = np.asarray(context.realized, dtype="float64")
        return values[:n] if len(values) >= n else None


class ARIMAForecaster(ClassicalForecaster):
    """ARIMA on the return series.

    ``d=0`` because returns are already differenced -- applying d=1 to returns
    over-differences and injects a spurious MA(1) component.
    """

    name = "arima"
    display_name = "ARIMA"

    def __init__(self, order: tuple[int, int, int] = (2, 0, 2), **kwargs: Any) -> None:
        super().__init__(order=order, **kwargs)
        self.order = order

    def _build(self, endog: np.ndarray) -> Any:
        from statsmodels.tsa.arima.model import ARIMA

        return ARIMA(endog, order=self.order, enforce_stationarity=True, enforce_invertibility=True)


class SARIMAForecaster(ClassicalForecaster):
    """Seasonal ARIMA with a weekly (5 trading day) period."""

    name = "sarima"
    display_name = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 5),
        **kwargs: Any,
    ) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order

    def _build(self, endog: np.ndarray) -> Any:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(
            endog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )


class ETSForecaster(ClassicalForecaster):
    """Exponential smoothing (innovations state space)."""

    name = "ets"
    display_name = "Exponential smoothing"

    def _build(self, endog: np.ndarray) -> Any:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        return ETSModel(pd.Series(endog), error="add", trend=None, seasonal=None)


CLASSICAL_MODELS = {
    m.name: m for m in (ARIMAForecaster, SARIMAForecaster, ETSForecaster)
}
