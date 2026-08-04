"""Walk-forward windows must adapt to the history a symbol actually has."""

from __future__ import annotations

import pytest

from forecaster.exceptions import InsufficientDataError
from forecaster.validation.harness import (
    ABSOLUTE_MIN_TRAIN,
    MIN_FOLDS,
    MIN_TEST,
    EvaluationConfig,
    EvaluationHarness,
)
from tests.conftest import make_ohlcv


def test_full_history_keeps_requested_windows() -> None:
    config = EvaluationConfig(symbol="TEST", train_size=504, test_size=63)
    fitted = config.fitted_to(1500)
    assert fitted.train_size == 504
    assert fitted.test_size == 63


def test_short_history_shrinks_windows_instead_of_failing() -> None:
    """A 2024 IPO has ~1.5 years, not the ~5 the defaults assume.

    Less history means wider error bars, not "no answer possible". Failing here
    made valid companies (RDDT, ARM) unusable.
    """
    config = EvaluationConfig(symbol="NEWCO", train_size=504, test_size=63, horizon=5)
    fitted = config.fitted_to(390)

    assert fitted.train_size < 504
    assert fitted.train_size >= ABSOLUTE_MIN_TRAIN
    assert fitted.test_size >= MIN_TEST

    # And the shrunken config must actually produce folds.
    folds = fitted.splitter().split(390)
    assert len(folds) >= 2


def test_adapted_config_still_purges_correctly() -> None:
    """Shrinking windows must not weaken the leakage guarantee."""
    config = EvaluationConfig(symbol="NEWCO", train_size=504, test_size=63, horizon=5, embargo=2)
    fitted = config.fitted_to(420)

    for fold in fitted.splitter().split(420):
        gap = int(fold.test_idx.min()) - int(fold.train_idx.max()) - 1
        assert gap >= fitted.horizon + fitted.embargo
        fold.assert_no_leakage(fitted.horizon)


def test_genuinely_insufficient_history_fails_with_a_clear_message() -> None:
    """There is a floor. A model fitted on a few weeks is not a forecast."""
    config = EvaluationConfig(symbol="JUSTLISTED", train_size=504, test_size=63)
    with pytest.raises(InsufficientDataError, match="usable bars"):
        config.fitted_to(80)


def test_adaptation_can_be_disabled() -> None:
    config = EvaluationConfig(symbol="X", train_size=504, test_size=63, adapt_to_history=False)
    assert config.fitted_to(300).train_size == 504


def test_harness_runs_end_to_end_on_short_history() -> None:
    """The whole pipeline, not just the config maths."""
    frame = make_ohlcv(n=620, seed=5)
    config = EvaluationConfig(symbol="SHORT", models=["ridge"], horizon=5)

    report = EvaluationHarness(config).run(frame)

    assert report.n_folds >= 2
    assert "naive_last_price" in report.results
    # The recorded split config must reflect what actually ran, so a metric on
    # shortened windows is never silently compared to one on full windows.
    assert report.config.train_size <= 504
