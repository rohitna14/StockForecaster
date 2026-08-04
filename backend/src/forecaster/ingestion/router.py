"""Provider failover router.

Tries each configured provider in priority order, respecting per-provider rate
limits and circuit breakers. A provider that fails repeatedly is taken out of
rotation for a cooldown rather than being retried on every request.

This is the only place that decides *where* price data comes from; callers ask
for a symbol and a date range and receive a normalised frame.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from forecaster.config import get_settings
from forecaster.exceptions import (
    AllProvidersFailedError,
    ProviderError,
    ProviderUnavailableError,
)
from forecaster.ingestion.base import PriceProvider
from forecaster.ingestion.ratelimit import CircuitBreaker, TokenBucket
from forecaster.ingestion.validators import ValidationReport, validate_bars
from forecaster.logging import get_logger

log = get_logger(__name__)


@dataclass
class FetchResult:
    symbol: str
    frame: pd.DataFrame
    provider: str
    report: ValidationReport
    attempts: list[tuple[str, str]]  # (provider, outcome)

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


class ProviderRouter:
    """Ordered failover across price providers."""

    def __init__(self, providers: list[PriceProvider] | None = None) -> None:
        settings = get_settings()
        self.providers: list[PriceProvider] = (
            providers if providers is not None else _default_providers()
        )
        self._buckets: dict[str, TokenBucket] = {
            p.name: TokenBucket(p.capabilities.requests_per_minute) for p in self.providers
        }
        self._breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(
                name=p.name,
                threshold=settings.provider_circuit_breaker_threshold,
                cooldown_seconds=settings.provider_circuit_breaker_cooldown_seconds,
            )
            for p in self.providers
        }

    @property
    def available_providers(self) -> list[PriceProvider]:
        return [
            p for p in self.providers if p.is_configured() and self._breakers[p.name].is_available
        ]

    async def fetch_daily(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        *,
        preferred: str | None = None,
    ) -> FetchResult:
        """Fetch daily bars, falling through providers until one succeeds.

        Args:
            preferred: force a specific provider to be tried first (used by the
                cross-check job to compare two sources on the same window).

        Raises:
            AllProvidersFailedError: every provider errored. Note that a
                provider legitimately returning *no data* for an unknown ticker
                is not a failure -- that yields an empty frame.
        """
        candidates = self.available_providers
        if preferred:
            candidates.sort(key=lambda p: p.name != preferred)

        if not candidates:
            raise AllProvidersFailedError(
                "No price providers are configured and available", symbol=symbol
            )

        attempts: list[tuple[str, str]] = []
        errors: dict[str, str] = {}

        for provider in candidates:
            breaker = self._breakers[provider.name]
            bucket = self._buckets[provider.name]
            try:
                await bucket.acquire(timeout=30.0)
                frame = await provider.fetch_daily(symbol, start, end)
            except ProviderError as exc:
                breaker.record_failure()
                errors[provider.name] = str(exc)
                attempts.append((provider.name, "error"))
                log.warning(
                    "provider_failed", provider=provider.name, symbol=symbol, error=str(exc)
                )
                continue
            except Exception as exc:  # noqa: BLE001 -- never let one provider kill the chain
                breaker.record_failure()
                errors[provider.name] = repr(exc)
                attempts.append((provider.name, "error"))
                log.warning(
                    "provider_crashed", provider=provider.name, symbol=symbol, error=repr(exc)
                )
                continue

            breaker.record_success()

            if frame.empty:
                # Not a provider fault -- the symbol may simply not exist here.
                attempts.append((provider.name, "empty"))
                log.debug("provider_empty", provider=provider.name, symbol=symbol)
                continue

            cleaned, report = validate_bars(frame, symbol)
            attempts.append((provider.name, "ok"))
            log.info(
                "fetched",
                symbol=symbol,
                provider=provider.name,
                rows=len(cleaned),
                start=str(cleaned.index.min().date()) if len(cleaned) else None,
                end=str(cleaned.index.max().date()) if len(cleaned) else None,
            )
            return FetchResult(
                symbol=symbol,
                frame=cleaned,
                provider=provider.name,
                report=report,
                attempts=attempts,
            )

        if errors and all(outcome == "error" for _, outcome in attempts):
            raise AllProvidersFailedError(
                f"All {len(attempts)} providers failed for {symbol}",
                symbol=symbol,
                errors=errors,
            )

        # Every provider answered cleanly but none had data for this symbol.
        log.info("no_data_anywhere", symbol=symbol, attempts=attempts)
        return FetchResult(
            symbol=symbol,
            frame=pd.DataFrame(),
            provider="none",
            report=ValidationReport(symbol=symbol),
            attempts=attempts,
        )

    def provider_status(self) -> dict[str, dict[str, object]]:
        """Health snapshot, surfaced by ``GET /health/ready``."""
        return {
            p.name: {
                "configured": p.is_configured(),
                "circuit": self._breakers[p.name].state.value,
                "available": p.is_configured() and self._breakers[p.name].is_available,
                "rate_per_minute": p.capabilities.requests_per_minute,
            }
            for p in self.providers
        }

    async def aclose(self) -> None:
        for provider in self.providers:
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()


def _default_providers() -> list[PriceProvider]:
    """Build the standard chain, skipping providers with no credentials.

    Order is deliberate: yfinance first (deepest history, adjusted closes),
    Stooq second (independent source, no key), keyed providers last so their
    scarce quota is preserved for when the free ones are down.
    """
    from forecaster.ingestion.providers.stooq_provider import StooqProvider
    from forecaster.ingestion.providers.tiingo_provider import TiingoProvider
    from forecaster.ingestion.providers.yfinance_provider import YFinanceProvider

    chain: list[PriceProvider] = [YFinanceProvider(), StooqProvider(), TiingoProvider()]
    configured = [p for p in chain if p.is_configured()]
    skipped = [p.name for p in chain if not p.is_configured()]
    if skipped:
        log.debug("providers_skipped_no_credentials", providers=skipped)
    if not configured:
        raise ProviderUnavailableError("No providers configured", provider="<router>")
    return configured
