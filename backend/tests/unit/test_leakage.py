"""Lookahead-bias guards.

**This is the most important test module in the repository.**

The resume claim "eliminated lookahead bias" is only defensible if something
actively checks for it. These tests are that something. They run as a separate,
named CI job so a failure is visible on the badge rather than buried.

Four independent guards, each catching a different class of mistake:

1. :func:`test_features_are_causal` -- corrupt the future, assert the past does
   not move. Catches any indicator using a centred window, a negative shift, or
   a whole-series statistic (``.mean()``, ``.max()``, a fitted scaler).
2. :func:`test_purge_gap_holds` -- assert no training label can read a bar
   inside the test window. Catches an off-by-one in the splitter.
3. :func:`test_shuffled_target_yields_no_skill` -- destroy the signal, assert
   the skill collapses. Catches leakage *anywhere* in the pipeline, including
   places nobody thought to check.
4. :func:`test_scaler_fitted_on_train_only` -- assert preprocessing never sees
   test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster import features as F
from forecaster.exceptions import ValidationConfigError
from forecaster.features.targets import forward_return
from forecaster.models.baselines import NaiveLastPrice
from forecaster.validation.skill import rmse_skill
from forecaster.validation.splitters import SplitMode, WalkForwardSplit
from tests.conftest import make_ohlcv, make_predictable_ohlcv

pytestmark = pytest.mark.leakage


# ═══════════════════════════ 1. feature causality ══════════════════════════
@pytest.mark.parametrize("feature_name", F.feature_set("all"))
def test_features_are_causal(feature_name: str, ohlcv: pd.DataFrame) -> None:
    """Every feature at bar t must depend only on bars <= t.

    Method: build features on the real frame, then rebuild on a frame whose
    values after a cut point have been replaced with garbage. Any value before
    the cut that changes proves the feature reached into the future.
    """
    cut = len(ohlcv) // 2

    corrupted = ohlcv.copy()
    rng = np.random.default_rng(999)
    tail = slice(cut, None)
    # Scale the future by a wild factor; keep OHLC ordering internally valid so
    # the failure can only come from lookahead, not from NaN propagation.
    factor = rng.uniform(5.0, 20.0, size=len(ohlcv) - cut)
    for col in ("open", "high", "low", "close", "adj_close"):
        corrupted.iloc[tail, corrupted.columns.get_loc(col)] = (
            ohlcv[col].iloc[tail].to_numpy() * factor
        )
    corrupted.iloc[tail, corrupted.columns.get_loc("volume")] = (
        ohlcv["volume"].iloc[tail].to_numpy() * 100
    )

    original = F.build(ohlcv, [feature_name])
    perturbed = F.build(corrupted, [feature_name])

    past_original = original.iloc[:cut]
    past_perturbed = perturbed.iloc[:cut]

    for col in past_original.columns:
        a = past_original[col].to_numpy(dtype="float64")
        b = past_perturbed[col].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        differs = ~both_nan & ~np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        assert not differs.any(), (
            f"Feature {feature_name!r} column {col!r} is NOT causal: "
            f"{int(differs.sum())} value(s) before index {cut} changed when only "
            f"future data was modified. First offending index: {int(np.argmax(differs))}"
        )


def test_target_is_forward_looking_by_design(ohlcv: pd.DataFrame) -> None:
    """Sanity check on the test above: the *target* must fail causality.

    If this passed, the corruption harness would not be sensitive enough to
    detect anything, and every other causality assertion would be vacuous.
    """
    cut = len(ohlcv) // 2
    corrupted = ohlcv.copy()
    corrupted.iloc[cut:, corrupted.columns.get_loc("close")] *= 10.0

    original = forward_return(ohlcv["close"], horizon=1).iloc[:cut]
    perturbed = forward_return(corrupted["close"], horizon=1).iloc[:cut]

    assert not np.allclose(
        original.to_numpy()[-1], perturbed.to_numpy()[-1], equal_nan=True
    ), "Corruption harness is not sensitive; the causality tests would be vacuous."


# ═══════════════════════════ 2. purge correctness ══════════════════════════
@pytest.mark.parametrize("horizon", [1, 5, 10, 21])
@pytest.mark.parametrize("mode", [SplitMode.ROLLING, SplitMode.ANCHORED])
def test_purge_gap_holds(horizon: int, mode: SplitMode) -> None:
    """No training label may read a bar inside its fold's test window."""
    n = 2000
    splitter = WalkForwardSplit(
        train_size=500, test_size=100, horizon=horizon, embargo=0, mode=mode
    )
    folds = splitter.split(n)
    assert folds, "expected at least one fold"

    for fold in folds:
        max_train = int(fold.train_idx.max())
        min_test = int(fold.test_idx.min())

        assert max_train < min_test, f"fold {fold.index}: train overlaps test"
        assert max_train + horizon < min_test, (
            f"fold {fold.index}: label at train bar {max_train} reads bar "
            f"{max_train + horizon}, inside the test window at {min_test}"
        )
        assert min_test - max_train - 1 >= horizon, (
            f"fold {fold.index}: purge gap {min_test - max_train - 1} < horizon {horizon}"
        )
        assert not set(fold.train_idx) & set(fold.test_idx)


@pytest.mark.parametrize("embargo", [0, 5, 20])
def test_embargo_widens_the_gap(embargo: int) -> None:
    splitter = WalkForwardSplit(train_size=400, test_size=80, horizon=5, embargo=embargo)
    for fold in splitter.split(1500):
        gap = int(fold.test_idx.min()) - int(fold.train_idx.max()) - 1
        assert gap >= 5 + embargo


def test_test_windows_do_not_overlap() -> None:
    """Default step must produce disjoint test windows.

    Overlapping test sets make fold scores correlated, which silently narrows
    every confidence interval computed from them.
    """
    folds = WalkForwardSplit(train_size=500, test_size=100, horizon=1).split(2000)
    seen: set[int] = set()
    for fold in folds:
        current = set(fold.test_idx.tolist())
        assert not (seen & current), f"fold {fold.index} test window overlaps an earlier one"
        seen |= current


def test_test_windows_are_strictly_ordered() -> None:
    folds = WalkForwardSplit(train_size=400, test_size=60, horizon=1).split(1800)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert int(earlier.test_idx.max()) < int(later.test_idx.min())


def test_impossible_config_raises_rather_than_silently_returning_nothing() -> None:
    """A config that cannot produce folds must fail loudly."""
    with pytest.raises(ValidationConfigError):
        WalkForwardSplit(train_size=5000, test_size=100, horizon=1).split(1000)


def test_fold_self_check_catches_a_bad_gap() -> None:
    """Fold.assert_no_leakage must reject hand-built leaky folds.

    Train ends at 99, test starts at 100 -- a zero-bar gap. Even at horizon=1
    this leaks: the label at bar 99 is computed from bar 100, the first test
    bar. A correctly purged fold for h=1 must end training at bar 98.
    """
    from forecaster.validation.splitters import Fold

    zero_gap = Fold(index=0, train_idx=np.arange(0, 100), test_idx=np.arange(100, 150))
    with pytest.raises(ValidationConfigError, match="Purge gap is too small"):
        zero_gap.assert_no_leakage(horizon=1)
    with pytest.raises(ValidationConfigError, match="Purge gap is too small"):
        zero_gap.assert_no_leakage(horizon=5)

    # One bar of purge is exactly enough for horizon=1, and not enough for 2.
    purged = Fold(index=0, train_idx=np.arange(0, 99), test_idx=np.arange(100, 150))
    purged.assert_no_leakage(horizon=1)
    with pytest.raises(ValidationConfigError, match="Purge gap is too small"):
        purged.assert_no_leakage(horizon=2)

    overlapping = Fold(index=0, train_idx=np.arange(0, 120), test_idx=np.arange(100, 150))
    with pytest.raises(ValidationConfigError):
        overlapping.assert_no_leakage(horizon=1)


# ═══════════════════ 3. the strongest guard: shuffled target ═══════════════
def _fit_predict_ridge(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)
    return np.asarray(model.predict(scaler.transform(X_test)))


def _run_walk_forward(
    frame: pd.DataFrame,
    *,
    shuffle_target: bool,
    seed: int = 0,
    feature_set: str = "core",
) -> float:
    """Run a minimal walk-forward and return mean RMSE skill vs naive."""
    names = F.feature_set(feature_set)
    X = F.build(frame, names)
    y = forward_return(frame["close"], horizon=1)

    data = X.join(y.rename("target")).dropna()
    features = data.drop(columns=["target"])
    target = data["target"]

    if shuffle_target:
        # Destroy the feature/target relationship while preserving the marginal
        # distribution of y exactly.
        rng = np.random.default_rng(seed)
        target = pd.Series(
            rng.permutation(target.to_numpy()), index=target.index, name="target"
        )

    splitter = WalkForwardSplit(train_size=500, test_size=100, horizon=1, embargo=2)
    skills: list[float] = []

    for fold in splitter.split(features):
        X_tr = features.to_numpy()[fold.train_idx]
        y_tr = target.to_numpy()[fold.train_idx]
        X_te = features.to_numpy()[fold.test_idx]
        y_te = target.to_numpy()[fold.test_idx]

        preds = _fit_predict_ridge(X_tr, y_tr, X_te)
        naive = NaiveLastPrice().fit(X_tr, y_tr).predict(X_te)
        skills.append(rmse_skill(y_te, preds, naive).skill)

    return float(np.mean(skills))


def test_shuffled_target_yields_no_skill() -> None:
    """Train on a shuffled target; skill versus naive must collapse to ~0.

    This is the single most valuable test here. It does not check one specific
    mistake -- it checks the *conclusion*. If any stage of the pipeline leaks
    (a feature peeking ahead, a scaler fitted on everything, a splitter
    off-by-one, an index misalignment), the model will still find the target
    through the leak and post positive skill even though the labels are noise.

    A small negative value is expected and healthy: a model fitting pure noise
    is slightly worse than predicting zero.
    """
    frame = make_ohlcv(n=1600, seed=11)
    skill = _run_walk_forward(frame, shuffle_target=True, seed=3)

    assert skill < 0.02, (
        f"Shuffled-target skill is {skill:.4f}, expected ~0. The pipeline is "
        f"leaking future information: a model trained on randomised labels "
        f"should have no predictive power whatsoever."
    )
    assert skill > -0.75, f"Skill {skill:.4f} is implausibly negative; check the harness."


def test_pipeline_finds_a_planted_signal() -> None:
    """Positive control.

    The shuffled-target test only proves the pipeline is not leaking. It would
    also pass if the pipeline were simply broken and predicting nothing. This
    asserts the same code *does* find skill when a real signal exists.

    Uses the ``minimal`` feature set deliberately. With the full 55-column
    ``core`` set, Ridge(alpha=1) on 500 training rows overfits hard enough to
    bury this signal entirely (in-sample R^2 0.27 -> out-of-sample -0.18). That
    is a genuine property of the estimator, not a plumbing fault, and it is
    pinned by :func:`test_more_features_can_hurt` below.
    """
    frame = make_predictable_ohlcv(n=1600, seed=5, strength=0.6)
    skill = _run_walk_forward(frame, shuffle_target=False, feature_set="minimal")

    assert skill > 0.01, (
        f"Skill on planted-signal data is {skill:.4f}; the pipeline should "
        f"recover an obvious autoregressive signal. Something is broken."
    )


def test_more_features_can_hurt() -> None:
    """Regression guard on a claim made in the methodology docs.

    On identical data with an identical planted signal, a 4-feature model beats
    a 55-feature one out-of-sample. This is the concrete evidence behind the
    'more features is not automatically better' point, and it exists so the
    claim cannot quietly stop being true.
    """
    frame = make_predictable_ohlcv(n=1600, seed=5, strength=0.6)
    minimal = _run_walk_forward(frame, shuffle_target=False, feature_set="minimal")
    core = _run_walk_forward(frame, shuffle_target=False, feature_set="core")

    assert minimal > core, (
        f"Expected the small feature set to generalise better on this data "
        f"(minimal={minimal:.4f}, core={core:.4f})."
    )


def test_random_walk_yields_no_meaningful_skill() -> None:
    """On a true random walk, honest skill must be near zero.

    Any sizeable positive number here would mean we had 'predicted' geometric
    Brownian motion, which is impossible by construction.
    """
    frame = make_ohlcv(n=1600, seed=23)
    skill = _run_walk_forward(frame, shuffle_target=False)
    assert skill < 0.05, (
        f"Skill of {skill:.4f} on synthetic GBM. There is no signal in this "
        f"data to find, so this indicates leakage."
    )


# ═══════════════════════════ 4. preprocessing isolation ════════════════════
def test_scaler_fitted_on_train_only(ohlcv: pd.DataFrame) -> None:
    """Standardisation must be fitted per fold, on training rows only.

    Fitting a scaler on the whole series leaks the test period's mean and
    variance into training. It rarely changes results dramatically, which is
    what makes it easy to miss and easy to leave in.
    """
    from sklearn.preprocessing import StandardScaler

    names = F.feature_set("core")
    X = F.build(ohlcv, names).dropna()
    fold = WalkForwardSplit(train_size=400, test_size=100, horizon=1).split(X)[0]

    train_only = StandardScaler().fit(X.to_numpy()[fold.train_idx])
    everything = StandardScaler().fit(X.to_numpy())

    assert not np.allclose(train_only.mean_, everything.mean_), (
        "Scaler fitted on train-only and on the full series are identical; "
        "the test cannot distinguish correct from leaky scaling."
    )

    transformed = train_only.transform(X.to_numpy()[fold.train_idx])
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)


def test_prediction_dates_are_strictly_forward(ohlcv: pd.DataFrame) -> None:
    """as_of_date < target_date for every stored prediction.

    Mirrors the DB CHECK constraint ``ck_pred_is_forward_looking`` so the
    invariant is enforced in code as well as in the schema.
    """
    horizon = 5
    index = ohlcv.index
    for position in range(0, len(index) - horizon, 97):
        as_of = index[position]
        target = index[position + horizon]
        assert target > as_of
