"""Stooq CSV endpoint.

The best free fallback: no API key, no registration, no documented quota, and
it serves clean split-adjusted daily history straight as CSV. Its weakness is
coverage (US equities and indices only, no fundamentals), which is exactly
complementary to yfinance's weakness (being an unofficial scrape).
"""

from __future__ import annotations

import datetime as dt
import io

import httpx
import pandas as pd

from forecaster.config import get_settings
from forecaster.exceptions import ProviderError, ProviderRateLimitError
from forecaster.ingestion.base import PriceProvider, ProviderCapabilities, normalise_frame
from forecaster.logging import get_logger

log = get_logger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/"


class StooqProvider(PriceProvider):
    name = "stooq"
    capabilities = ProviderCapabilities(
        daily=True,
        intraday=False,
        # Stooq serves split-adjusted prices but publishes no separate adjusted
        # close; normalise_frame sets adj_close = close, which is correct here.
        adjusted=False,
        corporate_actions=False,
        requests_per_minute=30.0,
        max_history_years=None,
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=get_settings().http_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; StockForecaster/0.1)"},
            )
        return self._client

    @staticmethod
    def _stooq_symbol(symbol: str) -> str:
        """Stooq namespaces US tickers with a ``.us`` suffix and lowercases them."""
        s = symbol.lower().replace("/", "-")
        return s if "." in s else f"{s}.us"

    async def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        client = await self._get_client()
        params = {
            "s": self._stooq_symbol(symbol),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
        try:
            response = await client.get(STOOQ_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Stooq request failed for {symbol}: {exc}", provider=self.name, symbol=symbol
            ) from exc

        text = response.text.strip()
        # Stooq answers unknown tickers and quota exhaustion with HTTP 200 and
        # a plain-text body rather than a status code, so the body has to be
        # sniffed. A valid response always begins with the CSV header.
        if not text.startswith("Date,"):
            if "exceeded" in text.lower() or "limit" in text.lower():
                raise ProviderRateLimitError(
                    f"Stooq refused request for {symbol}: {text[:120]}",
                    provider=self.name,
                    symbol=symbol,
                )
            # Unknown symbol / no coverage -- absence of data, not a fault.
            log.debug("stooq_no_data", symbol=symbol, body=text[:80])
            return pd.DataFrame()

        try:
            frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
        except Exception as exc:
            raise ProviderError(
                f"Stooq returned unparseable CSV for {symbol}: {exc}",
                provider=self.name,
                symbol=symbol,
            ) from exc

        return normalise_frame(frame, symbol=symbol, provider=self.name)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
