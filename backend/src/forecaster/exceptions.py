"""Domain exception hierarchy.

Every error raised by this package descends from :class:`ForecasterError`, so
the API layer can translate the whole tree into RFC 7807 ``problem+json``
responses in one place instead of scattering ``try/except Exception`` around.

Each exception carries a stable ``code`` (used as the problem *type* slug) and
an optional ``details`` mapping that is safe to serialise to clients.
"""

from __future__ import annotations

from typing import Any


class ForecasterError(Exception):
    """Base class for all application errors."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_problem(self) -> dict[str, Any]:
        problem: dict[str, Any] = {
            "type": f"https://stockforecaster.dev/errors/{self.code}",
            "title": self.code.replace("_", " ").title(),
            "status": self.http_status,
            "detail": self.message,
        }
        if self.details:
            problem["details"] = self.details
        return problem


# ── Configuration ─────────────────────────────────────────────────────────
class ConfigurationError(ForecasterError):
    code = "configuration_error"


# ── Data / ingestion ──────────────────────────────────────────────────────
class DataError(ForecasterError):
    code = "data_error"
    http_status = 400


class InstrumentNotFoundError(DataError):
    code = "instrument_not_found"
    http_status = 404


class InsufficientDataError(DataError):
    """Not enough history to do what was asked (train, backtest, indicate)."""

    code = "insufficient_data"
    http_status = 422


class DataValidationError(DataError):
    """A provider returned bars that fail OHLC sanity checks."""

    code = "data_validation_failed"
    http_status = 502


class ProviderError(ForecasterError):
    code = "provider_error"
    http_status = 502

    def __init__(self, message: str, provider: str, **details: Any) -> None:
        super().__init__(message, provider=provider, **details)
        self.provider = provider


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    http_status = 429


class ProviderUnavailableError(ProviderError):
    """Circuit breaker is open, or the provider is not configured."""

    code = "provider_unavailable"


class AllProvidersFailedError(DataError):
    code = "all_providers_failed"
    http_status = 503


# ── Features / modelling ──────────────────────────────────────────────────
class FeatureError(ForecasterError):
    code = "feature_error"
    http_status = 422


class UnknownFeatureError(FeatureError):
    code = "unknown_feature"


class ModelError(ForecasterError):
    code = "model_error"
    http_status = 422


class UnknownModelError(ModelError):
    code = "unknown_model"
    http_status = 404


class NotFittedError(ModelError):
    code = "model_not_fitted"


class ValidationConfigError(ForecasterError):
    """A walk-forward configuration that cannot produce any usable fold."""

    code = "invalid_validation_config"
    http_status = 422


class LeakageError(ForecasterError):
    """Raised when a guard detects future information in a training window.

    This should never surface at runtime -- it exists so that the guards fail
    loudly rather than silently producing impressive, wrong numbers.
    """

    code = "lookahead_leakage_detected"


# ── Jobs ──────────────────────────────────────────────────────────────────
class RunNotFoundError(ForecasterError):
    code = "run_not_found"
    http_status = 404


class RunFailedError(ForecasterError):
    code = "run_failed"
