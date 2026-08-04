"""Yahoo Finance via the ``yfinance`` package.

Primary source: no API key, deep history, and it supplies both raw and adjusted
closes plus corporate actions. It is an unofficial scrape, so it is wrapped in
retries and paired with fallbacks in the router.

``yfinance`` is synchronous and does blocking network I/O, so every call is
pushed to a worker thread to avoid stalling the event loop.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from forecaster.exceptions import ProviderError
from forecaster.ingestion.base import PriceProvider, ProviderCapabilities, normalise_frame
from forecaster.logging import get_logger

log = get_logger(__name__)


class YFinanceProvider(PriceProvider):
    name = "yfinance"
    capabilities = ProviderCapabilities(
        daily=True,
        intraday=True,
        adjusted=True,
        corporate_actions=True,
        fundamentals=True,
        news=True,
        requests_per_minute=60.0,
        max_history_years=None,
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    async def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        raw = await asyncio.to_thread(self._fetch_sync, symbol, start, end)
        return normalise_frame(raw, symbol=symbol, provider=self.name)

    def _fetch_sync(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            # auto_adjust=False keeps BOTH 'Close' and 'Adj Close'. With the
            # modern default (True) yfinance overwrites Close in place and
            # drops Adj Close, which silently destroys the raw series we want
            # to display to users.
            frame = ticker.history(
                start=start,
                end=end + dt.timedelta(days=1),  # yfinance 'end' is exclusive
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=False,
            )
        except Exception as exc:
            raise ProviderError(
                f"yfinance request failed for {symbol}: {exc}", provider=self.name, symbol=symbol
            ) from exc

        if frame is None or frame.empty:
            log.debug("yfinance_empty", symbol=symbol, start=str(start), end=str(end))
            return pd.DataFrame()
        return frame

    async def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Splits and dividends, for the ``corporate_actions`` table."""
        return await asyncio.to_thread(self._actions_sync, symbol)

    def _actions_sync(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf

        try:
            actions = yf.Ticker(symbol).actions
        except Exception as exc:
            raise ProviderError(
                f"yfinance actions failed for {symbol}: {exc}", provider=self.name, symbol=symbol
            ) from exc
        return pd.DataFrame() if actions is None else actions

    async def fetch_profile(self, symbol: str) -> dict[str, Any]:
        """Company metadata used to enrich the ``instruments`` row."""
        return await asyncio.to_thread(self._profile_sync, symbol)

    def _profile_sync(self, symbol: str) -> dict[str, Any]:
        import yfinance as yf

        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:  # noqa: BLE001 -- profile is best-effort enrichment
            log.debug("yfinance_profile_failed", symbol=symbol, error=str(exc))
            return {}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
        }
