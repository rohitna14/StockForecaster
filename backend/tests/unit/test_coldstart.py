"""Recent listings must still get a forecast."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the [ml] extra")

from forecaster.exceptions import InsufficientDataError  # noqa: E402
from forecaster.models.pooled import (  # noqa: E402
    MIN_TARGET_BARS,
    PooledForecaster,
    build_pool,
    validate_transfer,
)
from forecaster.validation.coldstart import (  # noqa: E402
    Confidence,
    Method,
    choose_method,
    confidence_for,
)
from tests.conftest import make_ohlcv


@pytest.fixture(scope="module")
def pool():
    frames = {f"SYM{i}": make_ohlcv(n=600, seed=i) for i in range(10)}
    return build_pool(frames, feature_set="cold_start", horizon=5)


# ── method selection ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("bars", "expected"),
    [
        (1254, Method.WALK_FORWARD),   # mature listing
        (700, Method.WALK_FORWARD),
        (339, Method.ADAPTIVE),        # 2025 IPO
        (150, Method.ADAPTIVE),
        (36, Method.TRANSFER),         # listed weeks ago
        (25, Method.TRANSFER),
        (10, Method.NONE),             # cannot build one feature row
    ],
)
def test_method_matches_available_history(bars: int, expected: Method) -> None:
    assert choose_method(bars) is expected


def test_confidence_tracks_evidence_not_headline() -> None:
    """A big number from thin data is weaker evidence than a modest one from thick."""
    strong = confidence_for(Method.WALK_FORWARD, 1254, 15.0)
    thin = confidence_for(Method.TRANSFER, 36, 25.0)
    assert strong is Confidence.HIGH
    assert thin is Confidence.VERY_LOW

    # A negative skill is never high confidence, however much history exists.
    assert confidence_for(Method.WALK_FORWARD, 2000, -3.0) is Confidence.LOW


# ── pooling ───────────────────────────────────────────────────────────────
def test_pool_stacks_multiple_symbols(pool) -> None:
    assert pool.n_symbols >= 6
    assert len(pool) > 1000
    assert pool.X.shape[0] == len(pool.y) == len(pool.groups)
    assert pool.X.shape[1] == len(pool.feature_names)


def test_pool_rejects_too_few_symbols() -> None:
    with pytest.raises(InsufficientDataError, match="at least"):
        build_pool({"ONE": make_ohlcv(n=600)}, feature_set="cold_start")


def test_pooled_model_predicts_for_an_unseen_symbol(pool) -> None:
    """The whole premise: a model that never saw this stock can still score it."""
    held_out = pool.groups == pool.symbols[0]

    model = PooledForecaster()
    model.fit(pool.X[~held_out], pool.y[~held_out])
    predictions = model.predict(pool.X[held_out])

    assert len(predictions) == int(held_out.sum())
    assert np.isfinite(predictions).all()


def test_leave_one_symbol_out_is_the_validation_used(pool) -> None:
    """Random row splits leak: adjacent rows of one symbol share their history.

    Holding out whole symbols answers the question a new IPO actually poses.
    """
    result = validate_transfer(pool, max_symbols=5)

    assert result.n_symbols >= 3
    assert result.n_rows > 500
    assert np.isfinite(result.rmse_skill_pct)
    assert set(result.per_symbol) <= set(pool.symbols)


def test_min_target_bars_allows_a_five_week_listing() -> None:
    """SPCX listed with 36 bars; the floor must sit below that."""
    assert MIN_TARGET_BARS <= 36
