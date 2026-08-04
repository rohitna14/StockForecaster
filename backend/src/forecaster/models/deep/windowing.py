"""Sequence windowing -- the fix for the prototype's central modelling bug.

The original code did this before every "deep learning" model::

    X_train_lstm = features['X_train'].reshape((n_samples, 1, n_features))
    lstm = Sequential([LSTM(100, input_shape=(1, n_features)), ...])

That is a sequence of length **one**. An LSTM unrolled over a single timestep
has no previous hidden state to carry, so its recurrence never fires: the gates
reduce to fixed affine transforms of a single input vector and the whole layer
collapses to a dense layer with extra steps. The same was true of the GRU,
Bi-LSTM, SimpleRNN and CNN-LSTM. Six "deep sequence models", none of which saw
a sequence, all of them slower than the dense layer they were equivalent to.

:func:`make_windows` produces genuine ``(samples, lookback, features)`` tensors
so the recurrence has something to recur over.

Windowing is applied **inside each fold, after the split**, so a window never
spans the train/test boundary -- doing it before splitting would hand every
test window a tail of training bars.
"""

from __future__ import annotations

import numpy as np

from forecaster.exceptions import InsufficientDataError

#: Default lookback. ~3 trading months: long enough for volatility clustering
#: and momentum to be visible, short enough that a fold of ~500 training bars
#: still yields ~440 usable windows.
DEFAULT_LOOKBACK = 60


def make_windows(
    X: np.ndarray, y: np.ndarray | None = None, lookback: int = DEFAULT_LOOKBACK
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a 2-D feature matrix into 3-D sequence windows.

    Window ``i`` spans rows ``[i, i + lookback)`` and is paired with the target
    at row ``i + lookback - 1`` -- the *last* bar in the window. That alignment
    is the point: the model sees ``lookback`` bars of history and predicts the
    target belonging to the final one, using no information beyond it.

    Args:
        X: ``(n_samples, n_features)``.
        y: optional ``(n_samples,)`` target.
        lookback: bars per window.

    Returns:
        ``(X_windows, y_windows)`` with shapes
        ``(n_samples - lookback + 1, lookback, n_features)`` and
        ``(n_samples - lookback + 1,)``.

    Raises:
        InsufficientDataError: fewer rows than the lookback.
    """
    X = np.asarray(X, dtype="float32")
    if X.ndim != 2:
        raise ValueError(f"expected 2-D feature matrix, got shape {X.shape}")

    n_samples, n_features = X.shape
    if n_samples < lookback:
        raise InsufficientDataError(
            f"Need at least {lookback} rows to build a window, got {n_samples}",
            required=lookback,
            available=n_samples,
        )

    n_windows = n_samples - lookback + 1
    # Stride trick: a view, not a copy. A naive loop would materialise
    # lookback x more memory than the source matrix.
    strides = (X.strides[0], X.strides[0], X.strides[1])
    windows = np.lib.stride_tricks.as_strided(
        X, shape=(n_windows, lookback, n_features), strides=strides, writeable=False
    )
    X_windows = np.ascontiguousarray(windows)

    if y is None:
        return X_windows, None

    y = np.asarray(y, dtype="float32").ravel()
    if len(y) != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {len(y)}")
    return X_windows, y[lookback - 1 :]


def last_window(X: np.ndarray, lookback: int = DEFAULT_LOOKBACK) -> np.ndarray:
    """The single most recent window, for live inference."""
    X = np.asarray(X, dtype="float32")
    if len(X) < lookback:
        raise InsufficientDataError(
            f"Need {lookback} rows for inference, got {len(X)}",
            required=lookback,
            available=len(X),
        )
    return X[-lookback:][np.newaxis, ...]


def align_predictions(
    predictions: np.ndarray, n_original: int, lookback: int, fill: float = 0.0
) -> np.ndarray:
    """Pad windowed predictions back to the original row count.

    Windowing costs the first ``lookback - 1`` rows, which have no complete
    history. They are filled rather than dropped so that every model in the
    harness returns exactly ``len(X_test)`` predictions and the leaderboard
    compares like with like.
    """
    predictions = np.asarray(predictions, dtype="float64").ravel()
    if len(predictions) == n_original:
        return predictions
    out = np.full(n_original, fill, dtype="float64")
    offset = n_original - len(predictions)
    if offset < 0:
        return predictions[-n_original:]
    out[offset:] = predictions
    return out
