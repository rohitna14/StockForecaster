"""Linear and regularised-linear models.

Underrated for this problem. With a genuine signal-to-noise ratio near zero,
heavily regularised linear models frequently beat gradient-boosted trees
out-of-sample, because there is very little non-linear structure to find and
trees spend their capacity memorising noise. Having them on the leaderboard is
what makes that visible rather than assumed.
"""

from __future__ import annotations

from typing import Any

from forecaster.models.adapters import make_sklearn_model


def _ridge(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(alpha=hp.get("alpha", 10.0), random_state=hp.get("random_state", 42))


def _lasso(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import Lasso

    return Lasso(
        alpha=hp.get("alpha", 1e-4),
        max_iter=hp.get("max_iter", 5000),
        random_state=hp.get("random_state", 42),
    )


def _elastic_net(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import ElasticNet

    return ElasticNet(
        alpha=hp.get("alpha", 1e-4),
        l1_ratio=hp.get("l1_ratio", 0.5),
        max_iter=hp.get("max_iter", 5000),
        random_state=hp.get("random_state", 42),
    )


def _huber(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import HuberRegressor

    return HuberRegressor(
        epsilon=hp.get("epsilon", 1.35),
        alpha=hp.get("alpha", 1e-3),
        max_iter=hp.get("max_iter", 500),
    )


def _ols(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _logistic(hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=hp.get("C", 0.1),
        max_iter=hp.get("max_iter", 1000),
        random_state=hp.get("random_state", 42),
    )


RidgeForecaster = make_sklearn_model(
    name="ridge",
    display_name="Ridge regression",
    family="linear",
    builder=_ridge,
    defaults={"alpha": 10.0},
    requires_scaling=True,
)

LassoForecaster = make_sklearn_model(
    name="lasso",
    display_name="Lasso regression",
    family="linear",
    builder=_lasso,
    defaults={"alpha": 1e-4},
    requires_scaling=True,
)

ElasticNetForecaster = make_sklearn_model(
    name="elastic_net",
    display_name="Elastic Net",
    family="linear",
    builder=_elastic_net,
    defaults={"alpha": 1e-4, "l1_ratio": 0.5},
    requires_scaling=True,
)

HuberForecaster = make_sklearn_model(
    name="huber",
    display_name="Huber regression",
    family="linear",
    builder=_huber,
    defaults={"epsilon": 1.35, "alpha": 1e-3},
    requires_scaling=True,
)

OLSForecaster = make_sklearn_model(
    name="ols",
    display_name="Ordinary least squares",
    family="linear",
    builder=_ols,
    requires_scaling=True,
)

LogisticForecaster = make_sklearn_model(
    name="logistic",
    display_name="Logistic regression",
    family="linear",
    builder=_logistic,
    defaults={"C": 0.1},
    requires_scaling=True,
    is_classifier=True,
)

LINEAR_MODELS = {
    m.name: m
    for m in (
        RidgeForecaster,
        LassoForecaster,
        ElasticNetForecaster,
        HuberForecaster,
        OLSForecaster,
        LogisticForecaster,
    )
}
