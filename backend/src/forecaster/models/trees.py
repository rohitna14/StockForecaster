"""Tree ensembles.

Hyperparameter defaults are deliberately far more conservative than library
defaults. On daily equity returns the signal-to-noise ratio is brutal, and a
LightGBM with 200 unconstrained trees will fit the noise perfectly and
generalise not at all -- which is exactly what happened in the prototype.

Concretely: shallow depth, strong leaf minimums, heavy subsampling, and
explicit L1/L2 regularisation.
"""

from __future__ import annotations

from typing import Any

from forecaster.models.adapters import make_sklearn_model


def _random_forest(hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=hp.get("n_estimators", 300),
        max_depth=hp.get("max_depth", 5),
        min_samples_leaf=hp.get("min_samples_leaf", 50),
        max_features=hp.get("max_features", "sqrt"),
        random_state=hp.get("random_state", 42),
        n_jobs=hp.get("n_jobs", -1),
    )


def _extra_trees(hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=hp.get("n_estimators", 300),
        max_depth=hp.get("max_depth", 6),
        min_samples_leaf=hp.get("min_samples_leaf", 50),
        max_features=hp.get("max_features", "sqrt"),
        random_state=hp.get("random_state", 42),
        n_jobs=hp.get("n_jobs", -1),
    )


def _gradient_boosting(hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=hp.get("max_iter", 200),
        learning_rate=hp.get("learning_rate", 0.03),
        max_depth=hp.get("max_depth", 3),
        min_samples_leaf=hp.get("min_samples_leaf", 40),
        l2_regularization=hp.get("l2_regularization", 1.0),
        early_stopping=hp.get("early_stopping", True),
        validation_fraction=hp.get("validation_fraction", 0.15),
        random_state=hp.get("random_state", 42),
    )


def _lightgbm(hp: dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=hp.get("n_estimators", 300),
        learning_rate=hp.get("learning_rate", 0.02),
        max_depth=hp.get("max_depth", 4),
        num_leaves=hp.get("num_leaves", 15),
        min_child_samples=hp.get("min_child_samples", 40),
        subsample=hp.get("subsample", 0.8),
        subsample_freq=hp.get("subsample_freq", 1),
        colsample_bytree=hp.get("colsample_bytree", 0.7),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.0),
        random_state=hp.get("random_state", 42),
        n_jobs=hp.get("n_jobs", -1),
        verbose=-1,
    )


def _xgboost(hp: dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=hp.get("n_estimators", 300),
        learning_rate=hp.get("learning_rate", 0.02),
        max_depth=hp.get("max_depth", 3),
        min_child_weight=hp.get("min_child_weight", 20),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.7),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.0),
        random_state=hp.get("random_state", 42),
        n_jobs=hp.get("n_jobs", -1),
        verbosity=0,
    )


def _catboost(hp: dict[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=hp.get("iterations", 300),
        learning_rate=hp.get("learning_rate", 0.02),
        depth=hp.get("depth", 4),
        l2_leaf_reg=hp.get("l2_leaf_reg", 5.0),
        subsample=hp.get("subsample", 0.8),
        random_seed=hp.get("random_state", 42),
        verbose=0,
        # Prevents CatBoost writing a catboost_info/ directory next to the CWD
        # on every fit -- the prototype committed that folder to git.
        allow_writing_files=False,
    )


RandomForestForecaster = make_sklearn_model(
    name="random_forest", display_name="Random Forest", family="tree", builder=_random_forest
)
ExtraTreesForecaster = make_sklearn_model(
    name="extra_trees", display_name="Extra Trees", family="tree", builder=_extra_trees
)
GradientBoostingForecaster = make_sklearn_model(
    name="gradient_boosting",
    display_name="Hist Gradient Boosting",
    family="tree",
    builder=_gradient_boosting,
)
LightGBMForecaster = make_sklearn_model(
    name="lightgbm", display_name="LightGBM", family="tree", builder=_lightgbm
)
XGBoostForecaster = make_sklearn_model(
    name="xgboost", display_name="XGBoost", family="tree", builder=_xgboost
)
CatBoostForecaster = make_sklearn_model(
    name="catboost", display_name="CatBoost", family="tree", builder=_catboost
)

TREE_MODELS = {
    m.name: m
    for m in (
        RandomForestForecaster,
        ExtraTreesForecaster,
        GradientBoostingForecaster,
        LightGBMForecaster,
        XGBoostForecaster,
        CatBoostForecaster,
    )
}
