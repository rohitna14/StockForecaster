"""On-demand ingestion.

The catalog holds ~6,600 instruments but only a handful were pre-ingested, so
any other symbol produced "No data yet" -- a dead end for a valid, listed
company.

Pre-ingesting everything is the obvious fix and the wrong one: ~6,600 symbols at
a provider-safe rate takes hours, most of it wasted on tickers nobody opens, and
it goes stale immediately. Instead, the first request for a symbol fetches it,
stores it in both tiers, and serves it. Cost is a one-off ~1-2s on first view;
every later request hits the database.

Two things make this safe under load:

* **Per-symbol locking** -- ten concurrent requests for a cold symbol trigger
  one fetch, not ten. The rest wait for it.
* **Negative caching** -- a symbol the providers do not know (a typo, a delisted
  ticker) is remembered as absent for a while, so a crawler hitting bad URLs
  cannot hammer the upstream API.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass

from forecaster.config import get_settings
from forecaster.db.models import Tier
from forecaster.db.repositories.instruments import InstrumentRepository
from forecaster.db.session import session_scope
from forecaster.logging import get_logger
from forecaster.ingestion.service import IngestionService

log = get_logger(__name__)

#: How long a "this symbol has no data anywhere" verdict is trusted.
NEGATIVE_TTL_SECONDS = 1800

#: A fetch that takes longer than this is abandoned so a page load cannot hang.
FETCH_TIMEOUT_SECONDS = 45.0


@dataclass
class IngestOutcome:
    symbol: str
    ingested: bool
    rows: int = 0
    reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.ingested and self.rows > 0


class OnDemandIngestor:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._negative: dict[str, float] = {}
        self._guard = asyncio.Lock()
        self._service: IngestionService | None = None

    async def _get_service(self) -> IngestionService:
        if self._service is None:
            self._service = IngestionService()
        return self._service

    async def _lock_for(self, symbol: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(symbol)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[symbol] = lock
            return lock

    def _is_known_absent(self, symbol: str) -> bool:
        expiry = self._negative.get(symbol)
        if expiry is None:
            return False
        if time.monotonic() > expiry:
            self._negative.pop(symbol, None)
            return False
        return True

    def _remember_absent(self, symbol: str) -> None:
        self._negative[symbol] = time.monotonic() + NEGATIVE_TTL_SECONDS

    async def ensure(
        self, symbol: str, *, years: int | None = None, promote: bool = True
    ) -> IngestOutcome:
        """Fetch and store ``symbol`` if we do not already have it.

        Returns immediately when the symbol is already stored or is known to be
        absent upstream.
        """
        symbol = symbol.strip().upper()
        if not symbol:
            return IngestOutcome(symbol, False, reason="empty symbol")

        if self._is_known_absent(symbol):
            return IngestOutcome(symbol, False, reason="known absent (cached)")

        lock = await self._lock_for(symbol)
        async with lock:
            # Re-check inside the lock: a concurrent request may have just
            # finished the work we were queued behind.
            if await self._already_stored(symbol):
                return IngestOutcome(symbol, False, reason="already stored")
            if self._is_known_absent(symbol):
                return IngestOutcome(symbol, False, reason="known absent (cached)")

            settings = get_settings()
            end = dt.date.today()
            start = end - dt.timedelta(days=365 * (years or settings.default_history_years))

            service = await self._get_service()
            started = time.perf_counter()
            try:
                hot, lake = await asyncio.wait_for(
                    service.ingest_symbol(symbol, start=start, end=end, to_hot=promote),
                    timeout=FETCH_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                log.warning("on_demand_timeout", symbol=symbol)
                return IngestOutcome(symbol, False, reason="provider timeout")
            except Exception as exc:  # noqa: BLE001 -- a bad ticker must not 500
                log.warning("on_demand_failed", symbol=symbol, error=str(exc))
                self._remember_absent(symbol)
                return IngestOutcome(symbol, False, reason=str(exc))

            rows = max(hot, lake)
            elapsed = time.perf_counter() - started

            if rows == 0:
                self._remember_absent(symbol)
                log.info("on_demand_no_data", symbol=symbol, seconds=round(elapsed, 2))
                return IngestOutcome(symbol, False, reason="no data from any provider")

            # A symbol ingested just now must be searchable immediately; the
            # index otherwise carries a 15-minute TTL and a brand-new listing
            # would stay invisible to search right after someone fetched it.
            try:
                from forecaster.search.service import ensure_index

                await ensure_index(force=True)
            except Exception as exc:  # noqa: BLE001 -- indexing is not critical
                log.debug("index_refresh_failed", symbol=symbol, error=str(exc))

            log.info(
                "on_demand_ingested", symbol=symbol, rows=rows, seconds=round(elapsed, 2)
            )
            return IngestOutcome(symbol, True, rows=rows, reason="ingested on demand")

    @staticmethod
    async def _already_stored(symbol: str) -> bool:
        from forecaster.lake import query as lake_query

        if lake_query.has_symbol(symbol):
            return True
        async with session_scope() as session:
            instrument = await InstrumentRepository(session).get_by_symbol(symbol)
            return bool(
                instrument and (instrument.tier == Tier.HOT or instrument.first_date)
            )

    def forget(self, symbol: str) -> None:
        """Clear the negative cache for a symbol (used after a manual ingest)."""
        self._negative.pop(symbol.strip().upper(), None)

    async def aclose(self) -> None:
        if self._service is not None:
            await self._service.aclose()
            self._service = None


_ingestor = OnDemandIngestor()


def get_ingestor() -> OnDemandIngestor:
    return _ingestor
