"""Feature engineering.

Importing this package registers the whole catalog, so
``forecaster.features.build(...)`` works without any further imports.
"""

from __future__ import annotations

from forecaster.features import catalog  # noqa: F401 -- import registers features
from forecaster.features.catalog import feature_set
from forecaster.features.registry import (
    FeatureSpec,
    build,
    get_spec,
    groups,
    list_features,
    registry_spec,
    required_history,
)

__all__ = [
    "FeatureSpec",
    "build",
    "feature_set",
    "get_spec",
    "groups",
    "list_features",
    "registry_spec",
    "required_history",
]
