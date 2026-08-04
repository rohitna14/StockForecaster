"""Model registry.

One lookup table from name -> class, plus named model sets. Deep-learning
models are imported lazily so that importing this module does not drag in
TensorFlow -- the prototype imported TF, XGBoost, LightGBM, CatBoost and
statsmodels at module scope, which pushed cold start past a minute and made the
container enormous.
"""

from __future__ import annotations

from typing import Any

from forecaster.exceptions import UnknownModelError
from forecaster.logging import get_logger
from forecaster.models.base import Forecaster
from forecaster.models.baselines import BASELINES, REFERENCE_BASELINE

log = get_logger(__name__)

_REGISTRY: dict[str, type[Forecaster]] = {}
_LAZY: dict[str, str] = {}


def register(model_cls: type[Forecaster]) -> type[Forecaster]:
    _REGISTRY[model_cls.name] = model_cls
    return model_cls


def register_lazy(name: str, import_path: str) -> None:
    """Defer an import until the model is actually requested."""
    _LAZY[name] = import_path


def _bootstrap() -> None:
    if _REGISTRY:
        return

    for cls in BASELINES.values():
        register(cls)

    try:
        from forecaster.models.linear import LINEAR_MODELS
        from forecaster.models.trees import TREE_MODELS

        for cls in (*LINEAR_MODELS.values(), *TREE_MODELS.values()):
            register(cls)
    except ImportError as exc:  # pragma: no cover -- [ml] extra not installed
        log.warning("ml_models_unavailable", error=str(exc), hint="pip install -e '.[ml]'")

    try:
        from forecaster.models.classical import CLASSICAL_MODELS

        for cls in CLASSICAL_MODELS.values():
            register(cls)
    except ImportError as exc:  # pragma: no cover
        log.debug("classical_models_unavailable", error=str(exc))

    # TensorFlow is heavy; only import it when a sequence model is requested.
    for name in ("lstm", "gru", "tcn", "transformer"):
        register_lazy(name, "forecaster.models.deep.registry")


def get_model_class(name: str) -> type[Forecaster]:
    _bootstrap()
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name in _LAZY:
        import importlib

        try:
            module = importlib.import_module(_LAZY[name])
        except ImportError as exc:
            raise UnknownModelError(
                f"Model {name!r} requires the [deep] extra: pip install -e '.[deep]'",
                model=name,
            ) from exc
        for cls in getattr(module, "DEEP_MODELS", {}).values():
            register(cls)
        if name in _REGISTRY:
            return _REGISTRY[name]

    raise UnknownModelError(f"Unknown model {name!r}", model=name, available=available_models())


def create(name: str, **hyperparams: Any) -> Forecaster:
    return get_model_class(name)(**hyperparams)


def available_models(*, family: str | None = None) -> list[str]:
    _bootstrap()
    names = sorted(set(_REGISTRY) | set(_LAZY))
    if family:
        names = [n for n in names if n in _REGISTRY and _REGISTRY[n].family == family]
    return names


def describe_all() -> list[dict[str, Any]]:
    """Catalog for ``GET /models``."""
    _bootstrap()
    out: list[dict[str, Any]] = []
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        out.append(
            {
                "name": cls.name,
                "display_name": cls.display_name,
                "family": cls.family,
                "is_classifier": cls.is_classifier,
                "requires_scaling": cls.requires_scaling,
                "is_sequence_model": cls.is_sequence_model,
                "is_baseline": cls.family == "baseline",
            }
        )
    for name in sorted(set(_LAZY) - set(_REGISTRY)):
        out.append(
            {
                "name": name,
                "display_name": name.upper(),
                "family": "deep",
                "is_classifier": False,
                "requires_scaling": True,
                "is_sequence_model": True,
                "is_baseline": False,
                "requires_extra": "deep",
            }
        )
    return out


# ── named model sets ──────────────────────────────────────────────────────
#: Always included in every evaluation. Not configurable -- a leaderboard
#: without a baseline is meaningless, so it cannot be turned off.
#: ``always_long`` and ``coin_flip`` emit class labels, so they are added only
#: for classification targets (see :func:`compatible_models`).
MANDATORY_BASELINES = [REFERENCE_BASELINE, "historical_mean"]
MANDATORY_CLASSIFIER_BASELINES = ["always_long", "coin_flip"]


def compatible_models(names: list[str], *, classification: bool) -> list[str]:
    """Filter a model list to those valid for the target type.

    Scoring a classifier against a return target produces garbage:
    ``always_long`` emits 1.0, which as a *return* means +100% per bar, and its
    RMSE then dwarfs everything else on the leaderboard while telling you
    nothing. Conversely a regressor's continuous output is not a class label.
    """
    _bootstrap()
    kept: list[str] = []
    for name in names:
        if name not in _REGISTRY:
            kept.append(name)  # lazy/deep models resolve later
            continue
        if _REGISTRY[name].is_classifier == classification:
            kept.append(name)
    mandatory = MANDATORY_CLASSIFIER_BASELINES if classification else MANDATORY_BASELINES
    return list(dict.fromkeys([*[m for m in mandatory if m in _REGISTRY], *kept]))


def model_set(name: str) -> list[str]:
    """Resolve a named model set, always including the mandatory baselines."""
    _bootstrap()
    sets: dict[str, list[str]] = {
        "baselines": list(BASELINES),
        "fast": ["ridge", "elastic_net", "lightgbm"],
        "linear": available_models(family="linear"),
        "trees": available_models(family="tree"),
        "classical": available_models(family="classical"),
        "standard": ["ridge", "elastic_net", "huber", "random_forest",
                     "gradient_boosting", "lightgbm", "xgboost", "catboost"],
        "deep": ["lstm", "gru", "tcn", "transformer"],
        "all": [n for n in available_models() if n not in _LAZY],
    }
    if name not in sets:
        raise UnknownModelError(f"Unknown model set {name!r}", available=sorted(sets))

    chosen = sets[name]
    merged = list(dict.fromkeys([*MANDATORY_BASELINES, *chosen]))
    return [m for m in merged if m in _REGISTRY or m in _LAZY]
