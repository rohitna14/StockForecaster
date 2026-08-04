"""Walk-forward and purged cross-validation splitters.

This module is the reason the project exists. The original prototype used a
single 80/20 chronological split and selected its "champion" model by the best
R-squared *on that same test set*, across fifteen models. That is not
validation; it is noise mining with extra steps.

Three ideas, in order of importance:

**1. Walk-forward.** Train on the past, test on the immediate future, roll
forward, repeat. Every fold's test window is strictly after its train window.
You get a distribution of out-of-sample scores across many market regimes
rather than a single number that depends entirely on where you happened to cut.

**2. Purging.** With an ``h``-bar forward target, the label at bar ``t`` is
computed from the close at ``t+h``. The final ``h`` training labels are
therefore built from bars that live inside the test window. Left alone, the
model has seen the test period's prices *through its own labels*. So we drop
those ``h`` bars from the end of every training window::

    train:  |=====================|  purge  |  test  |
                                   <-- h -->

**3. Embargo.** Financial returns are serially correlated, so bars immediately
after a test window still carry information about it. An embargo of ``e`` bars
widens the gap beyond the purge. Total gap = ``horizon + embargo``.

Everything is positional (integer offsets into the frame), not date-based, so
the same splitter works for daily, weekly or intraday bars without any calendar
arithmetic.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from forecaster.exceptions import ValidationConfigError
from forecaster.logging import get_logger

log = get_logger(__name__)


class SplitMode(StrEnum):
    #: Fixed-length training window that slides forward. Adapts to regime
    #: change; discards old data.
    ROLLING = "rolling"
    #: Training window grows from a fixed origin. Uses all history; adapts
    #: more slowly.
    ANCHORED = "anchored"


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, expressed as positional indices."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: dt.date | None = None
    train_end: dt.date | None = None
    test_start: dt.date | None = None
    test_end: dt.date | None = None
    purged: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def assert_no_leakage(self, horizon: int) -> None:
        """Fail loudly if this fold could leak.

        Called by the harness on every fold. It is cheap, and it converts a
        silent methodological error into a crash.
        """
        if self.n_train == 0 or self.n_test == 0:
            raise ValidationConfigError(f"Fold {self.index} has an empty side")

        max_train = int(self.train_idx.max())
        min_test = int(self.test_idx.min())

        if max_train >= min_test:
            raise ValidationConfigError(
                f"Fold {self.index}: train index {max_train} >= test index {min_test}"
            )
        # The label at max_train reads the bar at max_train + horizon. That bar
        # must fall strictly before the test window.
        if max_train + horizon >= min_test:
            raise ValidationConfigError(
                f"Fold {self.index}: label at train bar {max_train} reads bar "
                f"{max_train + horizon}, which is inside the test window "
                f"starting at {min_test}. Purge gap is too small for horizon={horizon}."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.index,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "purged": self.purged,
        }


@dataclass(frozen=True)
class WalkForwardSplit:
    """Rolling or anchored walk-forward splitter with purge and embargo.

    Args:
        train_size: training bars per fold. Ignored in anchored mode, where
            it acts as the *minimum* initial window.
        test_size: test bars per fold.
        step: bars to advance between folds. Defaults to ``test_size``, giving
            non-overlapping test windows -- which is what you want, because
            overlapping test windows make fold scores correlated and the
            resulting confidence intervals too narrow.
        horizon: forward horizon of the target. Sets the purge width.
        embargo: extra bars dropped between train and test.
        mode: rolling or anchored.
        max_folds: cap for quick experiments.
    """

    train_size: int = 756  # ~3 years of daily bars
    test_size: int = 63  # ~1 quarter
    step: int | None = None
    horizon: int = 1
    embargo: int = 0
    mode: SplitMode = SplitMode.ROLLING
    max_folds: int | None = None
    min_train_size: int = 252

    def __post_init__(self) -> None:
        if self.train_size < 1 or self.test_size < 1:
            raise ValidationConfigError("train_size and test_size must be >= 1")
        if self.horizon < 1:
            raise ValidationConfigError("horizon must be >= 1")
        if self.embargo < 0:
            raise ValidationConfigError("embargo must be >= 0")

    @property
    def gap(self) -> int:
        """Bars dropped between the end of train and the start of test."""
        return self.horizon + self.embargo

    @property
    def effective_step(self) -> int:
        return self.step if self.step is not None else self.test_size

    def n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self._iter_bounds(n_samples))

    def _iter_bounds(self, n_samples: int) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(train_start, train_stop, test_start, test_stop)`` exclusive-stop."""
        first_test_start = self.train_size + self.gap
        if first_test_start + self.test_size > n_samples:
            return

        fold = 0
        test_start = first_test_start
        while test_start + self.test_size <= n_samples:
            if self.max_folds is not None and fold >= self.max_folds:
                return

            test_stop = test_start + self.test_size
            # Purge: training stops `gap` bars before the test window opens.
            train_stop = test_start - self.gap
            train_start = (
                0 if self.mode is SplitMode.ANCHORED else max(0, train_stop - self.train_size)
            )

            if train_stop - train_start >= self.min_train_size:
                yield train_start, train_stop, test_start, test_stop
                fold += 1

            test_start += self.effective_step

    def split(self, data: pd.DataFrame | pd.Series | np.ndarray | int) -> list[Fold]:
        """Produce the folds for a dataset.

        Accepts a frame/series (dates are recorded on each fold), a numpy array,
        or a plain sample count.
        """
        index: pd.Index | None = None
        if isinstance(data, int):
            n_samples = data
        elif isinstance(data, np.ndarray):
            n_samples = len(data)
        else:
            n_samples = len(data)
            index = data.index

        folds: list[Fold] = []
        for i, (tr0, tr1, te0, te1) in enumerate(self._iter_bounds(n_samples)):
            train_idx = np.arange(tr0, tr1)
            test_idx = np.arange(te0, te1)

            dates: dict[str, dt.date | None] = {
                "train_start": None,
                "train_end": None,
                "test_start": None,
                "test_end": None,
            }
            if index is not None and len(index):
                dates = {
                    "train_start": _as_date(index[tr0]),
                    "train_end": _as_date(index[tr1 - 1]),
                    "test_start": _as_date(index[te0]),
                    "test_end": _as_date(index[te1 - 1]),
                }

            fold = Fold(index=i, train_idx=train_idx, test_idx=test_idx, purged=self.gap, **dates)
            fold.assert_no_leakage(self.horizon)
            folds.append(fold)

        if not folds:
            raise ValidationConfigError(
                "Walk-forward configuration produced zero folds",
                n_samples=n_samples,
                train_size=self.train_size,
                test_size=self.test_size,
                gap=self.gap,
                required_minimum=self.train_size + self.gap + self.test_size,
            )

        log.debug(
            "walk_forward_split",
            folds=len(folds),
            mode=self.mode.value,
            n_samples=n_samples,
            gap=self.gap,
        )
        return folds

    def describe(self) -> dict[str, Any]:
        """Serialisable config, persisted on ``model_runs.split_config``."""
        return {
            "splitter": "walk_forward",
            "mode": self.mode.value,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "step": self.effective_step,
            "horizon": self.horizon,
            "embargo": self.embargo,
            "purge_gap": self.gap,
            "min_train_size": self.min_train_size,
        }


@dataclass(frozen=True)
class PurgedKFold:
    """K-fold CV with purging and a symmetric embargo.

    Unlike walk-forward, training data may lie on *both* sides of the test
    window, so contamination can flow backwards as well as forwards. Both edges
    are purged, and an embargo is applied after the test block.

    Use this for hyperparameter search inside a training window, never for
    reporting headline performance -- it trains on data that postdates the test
    fold, which no live system could do.
    """

    n_splits: int = 5
    horizon: int = 1
    embargo_pct: float = 0.01

    def split(self, data: pd.DataFrame | pd.Series | np.ndarray | int) -> list[Fold]:
        n_samples = data if isinstance(data, int) else len(data)
        index = getattr(data, "index", None) if not isinstance(data, (int, np.ndarray)) else None

        if self.n_splits < 2:
            raise ValidationConfigError("n_splits must be >= 2")

        embargo = int(n_samples * self.embargo_pct)
        bounds = np.linspace(0, n_samples, self.n_splits + 1).astype(int)
        folds: list[Fold] = []

        for i in range(self.n_splits):
            te0, te1 = bounds[i], bounds[i + 1]
            test_idx = np.arange(te0, te1)

            # Purge horizon bars before the test block (their labels reach into
            # it) and horizon+embargo bars after (serial correlation).
            left = np.arange(0, max(0, te0 - self.horizon))
            right = np.arange(min(n_samples, te1 + self.horizon + embargo), n_samples)
            train_idx = np.concatenate([left, right])

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            dates: dict[str, dt.date | None] = {
                "train_start": None,
                "train_end": None,
                "test_start": None,
                "test_end": None,
            }
            if index is not None and len(index):
                dates = {
                    "train_start": _as_date(index[int(train_idx.min())]),
                    "train_end": _as_date(index[int(train_idx.max())]),
                    "test_start": _as_date(index[te0]),
                    "test_end": _as_date(index[te1 - 1]),
                }

            folds.append(
                Fold(
                    index=i,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    purged=self.horizon + embargo,
                    **dates,
                )
            )

        if not folds:
            raise ValidationConfigError("PurgedKFold produced zero usable folds")
        return folds

    def describe(self) -> dict[str, Any]:
        return {
            "splitter": "purged_kfold",
            "n_splits": self.n_splits,
            "horizon": self.horizon,
            "embargo_pct": self.embargo_pct,
        }


def _as_date(value: Any) -> dt.date | None:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def summarize_folds(folds: list[Fold]) -> pd.DataFrame:
    """Human-readable fold table -- used by the CLI and the methodology page."""
    return pd.DataFrame([f.as_dict() for f in folds])
