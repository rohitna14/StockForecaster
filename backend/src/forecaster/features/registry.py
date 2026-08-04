"""Declarative feature registry.

Features are registered by name with the number of bars of history they need
before producing a valid value. That ``min_history`` number is not decoration:
the walk-forward harness uses it to size the warm-up period so that no fold
ever trains on a partially-formed indicator.

Adding a feature is one decorated function; it then becomes available to every
model, every fold, and the ``/features`` endpoint automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forecaster.exceptions import UnknownFeatureError
from forecaster.logging import get_logger

log = get_logger(__name__)

#: A builder takes the OHLCV frame and returns a Series or a DataFrame.
FeatureBuilder = Callable[[pd.DataFrame], "pd.Series[Any] | pd.DataFrame"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    builder: FeatureBuilder
    group: str
    min_history: int
    description: str = ""
    #: Excluded from the default set (expensive, or experimental).
    optional: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, FeatureSpec] = {}


def feature(
    name: str,
    *,
    group: str,
    min_history: int,
    description: str = "",
    optional: bool = False,
    tags: tuple[str, ...] = (),
) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Register a feature builder."""

    def decorator(fn: FeatureBuilder) -> FeatureBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Feature {name!r} is already registered")
        _REGISTRY[name] = FeatureSpec(
            name=name,
            builder=fn,
            group=group,
            min_history=min_history,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            optional=optional,
            tags=tags,
        )
        return fn

    return decorator


def get_spec(name: str) -> FeatureSpec:
    if name not in _REGISTRY:
        raise UnknownFeatureError(
            f"Unknown feature {name!r}", available=sorted(_REGISTRY)[:20]
        )
    return _REGISTRY[name]


def list_features(*, group: str | None = None, include_optional: bool = True) -> list[FeatureSpec]:
    specs = list(_REGISTRY.values())
    if group:
        specs = [s for s in specs if s.group == group]
    if not include_optional:
        specs = [s for s in specs if not s.optional]
    return sorted(specs, key=lambda s: (s.group, s.name))


def groups() -> list[str]:
    return sorted({s.group for s in _REGISTRY.values()})


def required_history(names: list[str]) -> int:
    """Longest warm-up among the requested features.

    The harness reserves this many bars before the first training fold, so no
    model ever sees a half-formed 200-day average.
    """
    return max((get_spec(n).min_history for n in names), default=0)


def build(
    frame: pd.DataFrame, names: list[str], *, drop_na: bool = False
) -> pd.DataFrame:
    """Materialise the named features from an OHLCV frame.

    Args:
        frame: date-indexed OHLCV (adjusted). Must contain open/high/low/
            close/volume.
        drop_na: drop warm-up rows. Left False by default so the caller can
            align the feature matrix against targets before deciding what to
            discard.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")

    pieces: list[pd.DataFrame] = []
    for name in names:
        spec = get_spec(name)
        result = spec.builder(frame)
        if isinstance(result, pd.Series):
            result = result.to_frame(name if result.name is None else str(result.name))
        pieces.append(result)

    if not pieces:
        return pd.DataFrame(index=frame.index)

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    out.index.name = frame.index.name or "ts"

    # Indicators divide by rolling denominators that can be exactly zero on
    # flat bars; that produces inf, which silently poisons StandardScaler.
    out = out.replace([float("inf"), float("-inf")], pd.NA).astype("float64")

    return out.dropna() if drop_na else out


def registry_spec(names: list[str]) -> dict[str, Any]:
    """Serialisable description of a feature set, stored on ``feature_sets.spec``.

    Persisting this is what makes a published metric reproducible: the exact
    feature list and warm-up are recoverable from the run row alone.
    """
    return {
        "features": names,
        "min_history": required_history(names),
        "groups": sorted({get_spec(n).group for n in names}),
        "count": len(names),
    }
