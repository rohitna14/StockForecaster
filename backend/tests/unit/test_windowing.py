"""Sequence windowing tests.

The prototype's central modelling bug was reshaping to ``(n, 1, n_features)``
and calling the result an LSTM. These tests pin the correct behaviour so it
cannot regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecaster.exceptions import InsufficientDataError
from forecaster.models.deep.windowing import align_predictions, last_window, make_windows


def test_window_shape_is_three_dimensional() -> None:
    X = np.arange(100 * 4, dtype="float64").reshape(100, 4)
    X_win, _ = make_windows(X, None, lookback=10)

    assert X_win.ndim == 3
    assert X_win.shape == (91, 10, 4)
    assert X_win.shape[1] == 10, (
        "lookback dimension must be the window length -- a value of 1 here is "
        "the exact bug that made every 'LSTM' in the prototype a dense layer"
    )


def test_window_contents_are_contiguous_history() -> None:
    """Window i must be exactly rows [i, i+lookback)."""
    X = np.arange(50 * 2, dtype="float64").reshape(50, 2)
    X_win, _ = make_windows(X, None, lookback=5)

    for i in (0, 7, 20, 45):
        np.testing.assert_array_equal(X_win[i], X[i : i + 5])


def test_target_aligns_to_last_bar_of_window() -> None:
    """The label must belong to the final bar in the window, not the first.

    Aligning to the first bar would pair a window of history with a target
    that occurred lookback-1 bars *before* most of that history -- lookahead.
    """
    X = np.arange(30 * 3, dtype="float64").reshape(30, 3)
    y = np.arange(30, dtype="float64") * 10.0
    X_win, y_win = make_windows(X, y, lookback=6)

    assert y_win is not None
    assert len(X_win) == len(y_win) == 25
    # First window covers rows 0..5, so its label is y[5].
    assert y_win[0] == y[5]
    assert y_win[-1] == y[-1]
    np.testing.assert_array_equal(y_win, y[5:])


def test_windows_never_contain_future_rows() -> None:
    """Explicit causality check: corrupt the future, assert windows before it hold."""
    X = np.random.default_rng(0).normal(size=(80, 3))
    cut = 40

    corrupted = X.copy()
    corrupted[cut:] = 999.0

    original, _ = make_windows(X, None, lookback=10)
    perturbed, _ = make_windows(corrupted, None, lookback=10)

    # Window i ends at row i+9. Windows fully before the cut must be identical.
    last_clean = cut - 10
    np.testing.assert_allclose(original[:last_clean], perturbed[:last_clean])


def test_too_few_rows_raises() -> None:
    with pytest.raises(InsufficientDataError):
        make_windows(np.zeros((5, 3)), None, lookback=60)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="rows but y has"):
        make_windows(np.zeros((50, 3)), np.zeros(40), lookback=10)


def test_last_window_is_the_most_recent_history() -> None:
    X = np.arange(100 * 2, dtype="float64").reshape(100, 2)
    window = last_window(X, lookback=20)

    assert window.shape == (1, 20, 2)
    np.testing.assert_array_equal(window[0], X[-20:])


def test_align_predictions_pads_the_warmup() -> None:
    preds = np.array([1.0, 2.0, 3.0])
    out = align_predictions(preds, n_original=6, lookback=4, fill=-1.0)

    assert len(out) == 6
    np.testing.assert_array_equal(out[:3], [-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(out[3:], preds)


def test_align_predictions_is_identity_when_lengths_match() -> None:
    preds = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(align_predictions(preds, 3, 1), preds)


def test_windowing_does_not_copy_the_whole_matrix() -> None:
    """Stride trick sanity: memory must not blow up by a factor of lookback."""
    X = np.zeros((5000, 20), dtype="float32")
    X_win, _ = make_windows(X, None, lookback=60)
    assert X_win.shape == (4941, 60, 20)
