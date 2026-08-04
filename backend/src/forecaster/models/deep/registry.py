"""Lazy entry point for the deep-model family.

The model registry imports this module only when a sequence model is actually
requested, so TensorFlow never loads for a run that does not use it.
"""

from __future__ import annotations

from forecaster.models.deep.sequence import DEEP_MODELS

__all__ = ["DEEP_MODELS"]
