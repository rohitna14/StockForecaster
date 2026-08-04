"""Sequence-model tests.

Marked ``deep`` so CI can run them separately -- they need TensorFlow and are
slow relative to everything else.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="requires the [deep] extra")
pytestmark = pytest.mark.deep

from forecaster.models.deep.sequence import (  # noqa: E402
    DEEP_MODELS,
    GRUForecaster,
    LSTMForecaster,
    TCNForecaster,
    TransformerForecaster,
)


@pytest.fixture(scope="module")
def training_data() -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive signal a sequence model should be able to represent."""
    rng = np.random.default_rng(0)
    n, n_features = 500, 6
    X = rng.normal(size=(n, n_features)).astype("float32")
    y = np.zeros(n, dtype="float32")
    for t in range(10, n):
        y[t] = 0.5 * X[t - 1, 0] + 0.3 * X[t - 5, 1] + 0.1 * rng.normal()
    return X, y


def test_lstm_input_shape_has_a_real_lookback(training_data) -> None:
    """The regression guard for the prototype's central bug.

    A model whose input timestep dimension is 1 is not a sequence model. This
    asserts the built network actually expects `lookback` timesteps.
    """
    X, y = training_data
    model = LSTMForecaster(lookback=30, epochs=2, patience=1)
    model.fit(X, y)

    assert model._model is not None
    input_shape = model._model.input_shape
    assert input_shape[1] == 30, (
        f"LSTM input timestep dimension is {input_shape[1]}, expected 30. "
        f"A value of 1 collapses the recurrence into a dense layer."
    )
    assert input_shape[2] == X.shape[1]


def test_lstm_has_recurrent_weights(training_data) -> None:
    """An LSTM layer must carry recurrent kernels, and they must be trained."""
    X, y = training_data
    model = LSTMForecaster(lookback=20, epochs=2, patience=1)
    model.fit(X, y)

    lstm_layers = [layer for layer in model._model.layers if "lstm" in type(layer).__name__.lower()]
    assert lstm_layers, "no LSTM layer found"

    recurrent = [w for w in lstm_layers[0].weights if "recurrent" in w.name]
    assert recurrent, "LSTM layer has no recurrent kernel"
    assert not np.allclose(recurrent[0].numpy(), 0.0)


@pytest.mark.parametrize("model_cls", list(DEEP_MODELS.values()))
def test_predict_returns_one_value_per_input_row(model_cls, training_data) -> None:
    """Every model must return len(X) predictions so the leaderboard aligns."""
    X, y = training_data
    model = model_cls(lookback=20, epochs=2, patience=1)
    model.fit(X[:400], y[:400])

    preds = model.predict(X[400:])
    assert len(preds) == 100
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("model_cls", list(DEEP_MODELS.values()))
def test_prediction_is_causal(model_cls, training_data) -> None:
    """Corrupting row t must not change the prediction for any row < t.

    This catches non-causal padding in the TCN and a missing causal mask in the
    Transformer -- lookahead that lives inside the *model* and which no test on
    the feature matrix could ever detect.
    """
    X, y = training_data
    model = model_cls(lookback=20, epochs=3, patience=2)
    model.fit(X[:400], y[:400])

    X_test = X[400:].copy()
    baseline = model.predict(X_test)

    corrupted = X_test.copy()
    corrupted[60:] = 50.0  # wildly out-of-distribution future
    perturbed = model.predict(corrupted)

    # Predictions for rows strictly before the corruption must be unchanged.
    # Row i uses window [i-lookback+1, i], so rows < 60 are unaffected only
    # from index lookback-1 onward (earlier rows are warm-up fill).
    np.testing.assert_allclose(
        baseline[19:60],
        perturbed[19:60],
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{model_cls.name} is not causal: past predictions moved when "
        f"future inputs changed",
    )


def test_models_learn_a_recoverable_signal(training_data) -> None:
    """Sanity: a sequence model should beat predicting the mean on AR data."""
    X, y = training_data
    model = GRUForecaster(lookback=20, epochs=40, patience=8)
    model.fit(X[:400], y[:400])

    preds = model.predict(X[400:])[19:]
    actual = y[400:][19:]

    model_mse = float(np.mean((actual - preds) ** 2))
    mean_mse = float(np.mean((actual - y[:400].mean()) ** 2))
    assert model_mse < mean_mse, (
        f"GRU MSE {model_mse:.5f} did not beat predicting the training mean "
        f"({mean_mse:.5f}) on data with an explicit autoregressive signal."
    )


def test_insufficient_data_falls_back_without_crashing() -> None:
    """Too few rows to window must degrade gracefully, not raise."""
    X = np.random.default_rng(1).normal(size=(30, 5)).astype("float32")
    y = np.random.default_rng(2).normal(size=30).astype("float32")

    model = TCNForecaster(lookback=60, epochs=2)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 30
    assert np.isfinite(preds).all()


def test_transformer_positional_encoding_is_trainable(training_data) -> None:
    """Positional encoding must be a tracked layer with weights that actually train.

    The failure mode this guards is subtle: applying an Embedding to a
    ``tf.range`` constant *outside* the functional graph still adds positions
    numerically, so the model trains and predicts without error -- but Keras
    never registers the weights, they never appear in ``trainable_weights``,
    and the encoding stays frozen at random initialisation forever. The model
    looks fine and is quietly worse.
    """
    model = TransformerForecaster(lookback=20)
    network = model._build_network(20, 5)

    pos_layers = [layer for layer in network.layers if layer.name == "positional_embedding"]
    assert pos_layers, "Transformer has no positional embedding layer"

    trainable = {w.path for w in network.trainable_weights}
    assert any("positional_embedding" in path for path in trainable), (
        "positional embedding weights are not in trainable_weights -- they will "
        "never be updated by the optimiser"
    )

    # And confirm they genuinely move during training rather than sitting at init.
    X, y = training_data
    fitted = TransformerForecaster(lookback=20, epochs=6, patience=5)
    initial = (
        fitted._build_network(20, X.shape[1])
        .get_layer("positional_embedding")
        .get_weights()[0]
        .copy()
    )

    fitted.fit(X[:300], y[:300])
    trained = fitted._model.get_layer("positional_embedding").get_weights()[0]

    assert trained.shape == initial.shape
    assert not np.allclose(trained, initial), (
        "positional embedding weights are identical before and after training"
    )
