"""TensorFlow sequence models: LSTM, GRU, TCN, Transformer.

All four consume real ``(samples, lookback, features)`` tensors -- see
:mod:`forecaster.models.deep.windowing` for why that sentence needs saying.

Capacity is deliberately small (32-64 units, 1-2 layers). With ~450 training
windows per fold and a signal-to-noise ratio this low, a 3-layer 100-unit stack
memorises the training set within a handful of epochs. The prototype trained
100/50/25-unit stacked LSTMs for a fixed 30 epochs with no early stopping and
no validation split, which is a recipe for exactly that.

Every model here uses early stopping on a *chronological* validation tail of the
training window -- never a random split, which would let the model validate on
bars that precede its training data.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Silence TF's C++ logging before the first import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from forecaster.logging import get_logger
from forecaster.models.base import Forecaster, ModelContext
from forecaster.models.deep.windowing import (
    DEFAULT_LOOKBACK,
    align_predictions,
    make_windows,
)

log = get_logger(__name__)


def _tf() -> Any:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    return tf


class SequenceForecaster(Forecaster):
    """Base class for windowed neural forecasters."""

    family = "deep"
    requires_scaling = True
    is_sequence_model = True

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
            **kwargs,
        )
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self._model: Any = None
        self._fallback = 0.0
        self._history: dict[str, list[float]] = {}

    def _build_network(self, lookback: int, n_features: int) -> Any:
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray, context: ModelContext | None) -> None:
        tf = _tf()
        tf.keras.utils.set_random_seed(self.random_state)

        self._fallback = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            X_win, y_win = make_windows(X, y, self.lookback)
        except Exception as exc:  # noqa: BLE001
            log.warning("windowing_failed", model=self.name, error=str(exc))
            self._model = None
            return

        assert y_win is not None
        finite = np.isfinite(y_win)
        X_win, y_win = X_win[finite], y_win[finite]
        if len(X_win) < 80:
            log.warning("deep_insufficient_windows", model=self.name, n=len(X_win))
            self._model = None
            return

        # Chronological validation tail. A random split would validate on bars
        # that precede training bars, which is leakage in a time series.
        n_val = max(20, int(len(X_win) * self.validation_fraction))
        n_val = min(n_val, len(X_win) // 3)
        X_tr, y_tr = X_win[:-n_val], y_win[:-n_val]
        X_val, y_val = X_win[-n_val:], y_win[-n_val:]

        self._model = self._build_network(self.lookback, X.shape[1])
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(delta=1.0),  # robust to fat-tailed returns
            metrics=["mae"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(3, self.patience // 3), min_lr=1e-6
            ),
        ]

        history = self._model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,  # order matters; keep batches chronological
        )
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        log.debug(
            "deep_fit_complete",
            model=self.name,
            epochs_run=len(self._history.get("loss", [])),
            best_val=min(self._history.get("val_loss", [float("nan")])),
        )

    def _predict(self, X: np.ndarray, context: ModelContext | None) -> np.ndarray:
        if self._model is None:
            return np.full(len(X), self._fallback, dtype="float64")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if len(X) < self.lookback:
            return np.full(len(X), self._fallback, dtype="float64")

        X_win, _ = make_windows(X, None, self.lookback)
        raw = self._model.predict(X_win, verbose=0).ravel()
        # The first lookback-1 test rows have no complete window; fill with the
        # training mean so the output length matches every other model's.
        return align_predictions(raw, len(X), self.lookback, fill=self._fallback)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["training_history"] = self._history
        return out


class LSTMForecaster(SequenceForecaster):
    """Stacked LSTM over a real 60-bar lookback."""

    name = "lstm"
    display_name = "LSTM"

    def _build_network(self, lookback: int, n_features: int) -> Any:
        tf = _tf()
        L = tf.keras.layers
        return tf.keras.Sequential(
            [
                L.Input(shape=(lookback, n_features)),
                L.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
                L.LSTM(32, dropout=0.2),
                L.Dense(16, activation="relu"),
                L.Dropout(0.2),
                L.Dense(1),
            ],
            name="lstm_forecaster",
        )


class GRUForecaster(SequenceForecaster):
    """GRU -- fewer parameters than LSTM, usually better on short series."""

    name = "gru"
    display_name = "GRU"

    def _build_network(self, lookback: int, n_features: int) -> Any:
        tf = _tf()
        L = tf.keras.layers
        return tf.keras.Sequential(
            [
                L.Input(shape=(lookback, n_features)),
                L.GRU(64, return_sequences=True, dropout=0.2),
                L.GRU(32, dropout=0.2),
                L.Dense(16, activation="relu"),
                L.Dense(1),
            ],
            name="gru_forecaster",
        )


class TCNForecaster(SequenceForecaster):
    """Temporal Convolutional Network -- dilated *causal* convolutions.

    ``padding="causal"`` is what makes this valid for forecasting: output at
    timestep *t* depends only on inputs at ``<= t``. A plain ``"same"`` padding
    convolution is centred and would read the future, which is a lookahead bug
    that no test on the *features* would ever catch, because it lives inside the
    model.

    Dilations 1,2,4,8 give a receptive field of ~31 bars with four layers, and
    it trains several times faster than an LSTM because the convolutions
    parallelise across time.
    """

    name = "tcn"
    display_name = "Temporal CNN"

    def _build_network(self, lookback: int, n_features: int) -> Any:
        tf = _tf()
        L = tf.keras.layers

        inputs = L.Input(shape=(lookback, n_features))
        x = inputs
        for dilation in (1, 2, 4, 8):
            residual = x
            x = L.Conv1D(
                filters=32,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
            )(x)
            x = L.LayerNormalization()(x)
            x = L.Dropout(0.15)(x)
            # Residual connection; 1x1 conv matches channel counts on the first
            # block where the input still has n_features channels.
            if residual.shape[-1] != x.shape[-1]:
                residual = L.Conv1D(32, 1, padding="same")(residual)
            x = L.Add()([x, residual])

        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(16, activation="relu")(x)
        outputs = L.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name="tcn_forecaster")


class TransformerForecaster(SequenceForecaster):
    """Encoder-only temporal Transformer with causal self-attention.

    The ``use_causal_mask=True`` flag is load-bearing for the same reason as
    causal padding in the TCN: without it every position attends to every other
    position, including future ones, and the model reads bars it has no right
    to see.
    """

    name = "transformer"
    display_name = "Transformer"

    def __init__(
        self, num_heads: int = 4, key_dim: int = 16, ff_dim: int = 64, **kwargs: Any
    ) -> None:
        super().__init__(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim

    def _build_network(self, lookback: int, n_features: int) -> Any:
        tf = _tf()
        L = tf.keras.layers

        class PositionalEmbedding(L.Layer):
            """Learned positional encoding, as a tracked layer.

            Attention is permutation-invariant, so without positions the model
            cannot tell bar 1 from bar 60 and the 'temporal' Transformer is a
            bag-of-timesteps.

            This must be a real Layer rather than an Embedding applied to a
            ``tf.range`` constant outside the graph: Keras only tracks weights
            reachable through layers, so the outside-the-graph version produced
            an embedding that was never registered, never appeared in
            ``trainable_weights``, and therefore stayed frozen at its random
            initialisation for the entire run.
            """

            def __init__(self, sequence_length: int, d_model: int, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.sequence_length = sequence_length
                self.d_model = d_model
                self.embedding = L.Embedding(input_dim=sequence_length, output_dim=d_model)

            def call(self, inputs: Any) -> Any:
                positions = tf.range(start=0, limit=self.sequence_length, delta=1)
                return inputs + self.embedding(positions)

            def get_config(self) -> dict[str, Any]:
                return {
                    **super().get_config(),
                    "sequence_length": self.sequence_length,
                    "d_model": self.d_model,
                }

        inputs = L.Input(shape=(lookback, n_features))
        x = L.Dense(self.ff_dim)(inputs)  # project to model dimension
        x = PositionalEmbedding(lookback, self.ff_dim, name="positional_embedding")(x)

        for _ in range(2):
            attn = L.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.key_dim, dropout=0.1
            )(x, x, use_causal_mask=True)
            x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, attn]))

            ff = L.Dense(self.ff_dim * 2, activation="relu")(x)
            ff = L.Dropout(0.1)(ff)
            ff = L.Dense(self.ff_dim)(ff)
            x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, ff]))

        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.15)(x)
        outputs = L.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name="transformer_forecaster")


DEEP_MODELS = {
    m.name: m for m in (LSTMForecaster, GRUForecaster, TCNForecaster, TransformerForecaster)
}
