"""Liveness, readiness, and the methodology endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from forecaster.api.deps import InstrumentRepoDep, get_provider_router
from forecaster.api.schemas.common import HealthResponse, ReadinessResponse
from forecaster.config import get_settings
from forecaster.db.models import Tier
from forecaster.db.session import healthcheck
from forecaster.lake import writer as lake_writer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness. Cheap, no dependencies -- safe for a container probe."""
    settings = get_settings()
    return HealthResponse(status="ok", version="0.1.0", environment=settings.environment.value)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(instruments: InstrumentRepoDep) -> ReadinessResponse:
    """Readiness: database, lake and provider chain all reachable."""
    db_ok = await healthcheck()

    hot_symbols = 0
    if db_ok:
        try:
            hot_symbols = await instruments.count(tier=Tier.HOT)
        except Exception:  # noqa: BLE001 -- readiness must not raise
            db_ok = False

    stats = lake_writer.lake_stats()
    lake_symbols = int(stats.get("symbols", 0))
    provider_status = get_provider_router().provider_status()

    any_provider = any(p.get("available") for p in provider_status.values())
    status = "ready" if (db_ok and any_provider) else "degraded"

    return ReadinessResponse(
        status=status,
        database=db_ok,
        lake=lake_symbols > 0,
        providers=provider_status,
        hot_symbols=hot_symbols,
        lake_symbols=lake_symbols,
    )


@router.get("/methodology")
async def methodology() -> dict[str, Any]:
    """Machine-readable summary of how results here are validated.

    Exposed as an endpoint so the frontend's methodology page and the Streamlit
    client render the same claims from one source, and so the guarantees are
    inspectable without reading the repository.
    """
    return {
        "validation": {
            "scheme": "walk-forward",
            "modes": ["rolling", "anchored"],
            "purge": "training window ends `horizon` bars before the test window opens",
            "embargo": "additional bars dropped to break serial correlation",
            "preprocessing": "fitted inside each fold on training rows only",
            "model_selection": "on all folds except the last; the leaderboard is a report, not a selection step",
        },
        "baselines": [
            "naive_last_price",
            "historical_mean",
            "drift",
            "ewma",
            "seasonal_naive",
            "always_long",
            "coin_flip",
        ],
        "reference_baseline": "naive_last_price",
        "significance_tests": [
            "diebold_mariano",
            "block_bootstrap_ci",
            "probability_of_backtest_overfitting",
            "deflated_sharpe_ratio",
        ],
        "prediction_intervals": {
            "method": "split conformal",
            "nominal_coverage": 0.8,
        },
        "leakage_guards": [
            "per-feature causality: corrupt the future, assert the past is unchanged",
            "purge correctness: max(train) + horizon < min(test) on every fold",
            "shuffled target: permute labels, assert skill collapses to ~0",
            "positive control: assert the same pipeline finds a planted signal",
            "preprocessing isolation: scalers fitted per fold on train rows only",
        ],
        "known_findings": {
            "directional_price_prediction": (
                "No model beat the naive baseline at any horizon tested. "
                "Hit rates sat below the base rate. This is reported, not hidden."
            ),
            "volatility_forecasting": (
                "Statistically significant improvement over a random-walk "
                "baseline, consistent across the tested universe and grounded "
                "in volatility clustering."
            ),
        },
        "limitations": [
            "feature set chosen once up front, not re-selected per fold",
            "universes seeded from present-day index membership (survivorship)",
            "single-name only; no cross-sectional training",
            "volatility is not directly tradable without an options layer",
        ],
        "disclaimer": "Educational and research use. Not financial advice.",
    }
