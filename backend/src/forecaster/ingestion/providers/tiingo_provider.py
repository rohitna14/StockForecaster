"""Tiingo EOD API.

Keyed but genuinely free (~1,000 requests/day, 50 unique symbols/month on the
free tier). Highest data quality of the three price sources and it publishes a
proper adjusted series, so it is the tie-breaker when yfinance and Stooq
disagree. Skipped automatically when no key is set.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pandas as pd

from forecaster.config import get_settings
from forecaster.exceptions import ProviderError, ProviderRateLimitError
from forecaster.ingestion.base import PriceProvider, ProviderCapabilities, normalise_frame
from forecaster.logging import get_logger

log = get_logger(__name__)

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"


class TiingoProvider(PriceProvider):
    name = "tiingo"
    capabilities = ProviderCapabilities(
        daily=True,
        intraday=False,
        adjusted=True,
        corporate_actions=True,
        requests_per_minute=50.0,
        requests_per_day=1000,
        max_history_years=None,
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    def is_configured(self) -> bool:
        return get_settings().tiingo_api_key is not None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            settings = get_settings()
            key = settings.tiingo_api_key
            if key is None:
                raise ProviderError("Tiingo API key not configured", provider=self.name)
            self._client = httpx.AsyncClient(
                timeout=settings.http_timeout_seconds,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {key.get_secret_value()}",
                },
            )
        return self._client

    async def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        client = await self._get_client()
        url = f"{TIINGO_BASE}/{symbol.lower()}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
            "resampleFreq": "daily",
        }
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429:
                raise ProviderRateLimitError(
                    "Tiingo daily quota exhausted", provider=self.name, symbol=symbol
                )
            if response.status_code == 404:
                log.debug("tiingo_unknown_symbol", symbol=symbol)
                return pd.DataFrame()
            response.raise_for_status()
            payload = response.json()
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Tiingo request failed for {symbol}: {exc}", provider=self.name, symbol=symbol
            ) from exc

        if not payload:
            return pd.DataFrame()

        frame = pd.DataFrame(payload)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.set_index("date")

        # Tiingo names the adjusted columns adjOpen/adjHigh/... and keeps the
        # raw ones unprefixed. We want raw OHLC + adjClose so the adjustment
        # factor stays recoverable downstream.
        frame = frame.rename(columns={"adjClose": "adj_close"})
        keep = ["open", "high", "low", "close", "adj_close", "volume"]
        missing = [c for c in keep if c not in frame.columns]
        if missing:
            raise ProviderError(
                f"Tiingo payload for {symbol} missing {missing}",
                provider=self.name,
                symbol=symbol,
            )
        return normalise_frame(frame[keep], symbol=symbol, provider=self.name)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
