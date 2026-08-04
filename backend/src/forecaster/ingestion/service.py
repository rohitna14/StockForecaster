"""Ingestion orchestration.

Ties together: router (where data comes from) -> validators (is it sane) ->
lake writer (cold tier, always) -> Postgres (hot tier, when promoted).

Write policy: **every fetched bar lands in the Parquet lake**, regardless of
tier. The lake is the system of record and is cheap. Postgres holds only the
hot working set, so it is a cache that can be rebuilt from the lake at any time.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field

from forecaster.config import get_settings
from forecaster.db.models import Instrument, Tier
from forecaster.db.repositories.instruments import InstrumentRepository
from forecaster.db.repositories.ohlcv import OHLCVRepository
from forecaster.db.session import session_scope
from forecaster.ingestion.router import ProviderRouter
from forecaster.lake import writer as lake_writer
from forecaster.logging import get_logger, log_context

log = get_logger(__name__)


@dataclass
class IngestSummary:
    requested: int = 0
    succeeded: int = 0
    empty: int = 0
    failed: int = 0
    rows_hot: int = 0
    rows_lake: int = 0
    failures: dict[str, str] = field(default_factory=dict)
    empties: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "succeeded": self.succeeded,
            "empty": self.empty,
            "failed": self.failed,
            "rows_hot": self.rows_hot,
            "rows_lake": self.rows_lake,
        }


class IngestionService:
    def __init__(self, router: ProviderRouter | None = None) -> None:
        self.router = router or ProviderRouter()
        self.settings = get_settings()

    async def ingest_symbol(
        self,
        symbol: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        to_hot: bool | None = None,
        create_missing: bool = True,
    ) -> tuple[int, int]:
        """Fetch and persist one symbol. Returns ``(rows_hot, rows_lake)``.

        Args:
            to_hot: force hot-tier write. When ``None`` (default) the
                instrument's existing tier decides, so a scheduled refresh does
                not accidentally promote the entire cold universe.
        """
        symbol = symbol.upper()
        end = end or dt.date.today()
        start = start or end - dt.timedelta(days=365 * self.settings.default_history_years)

        with log_context(symbol=symbol):
            result = await self.router.fetch_daily(symbol, start, end)
            if result.is_empty:
                return 0, 0

            rows_lake = lake_writer.write_bars(symbol, result.frame, result.provider)

            async with session_scope() as session:
                instruments = InstrumentRepository(session)
                instrument = await instruments.get_by_symbol(symbol)

                if instrument is None:
                    if not create_missing:
                        return 0, rows_lake
                    instrument = await self._create_instrument(session, symbol)

                should_write_hot = (
                    to_hot if to_hot is not None else instrument.tier == Tier.HOT
                )
                if not should_write_hot:
                    return 0, rows_lake

                ohlcv = OHLCVRepository(session)
                rows_hot = await ohlcv.upsert_bars(instrument.id, result.frame, result.provider)

                first, last, _ = await ohlcv.coverage(instrument.id)
                if first and last:
                    await instruments.update_coverage(instrument.id, first, last)
                if to_hot:
                    await instruments.set_tier(symbol, Tier.HOT)

            return rows_hot, rows_lake

    async def ingest_many(
        self,
        symbols: list[str],
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        to_hot: bool | None = None,
        concurrency: int = 4,
        progress: object | None = None,
    ) -> IngestSummary:
        """Ingest a list of symbols with bounded concurrency.

        Concurrency stays low by default: free providers throttle aggressively
        and a burst of 50 parallel requests gets the whole session rate-limited,
        which is slower than four steady workers.
        """
        summary = IngestSummary(requested=len(symbols))
        semaphore = asyncio.Semaphore(concurrency)

        async def one(sym: str) -> None:
            async with semaphore:
                try:
                    hot, lake = await self.ingest_symbol(
                        sym, start=start, end=end, to_hot=to_hot
                    )
                except Exception as exc:  # noqa: BLE001 -- one bad ticker must not abort the batch
                    summary.failed += 1
                    summary.failures[sym] = str(exc)
                    log.warning("ingest_failed", symbol=sym, error=str(exc))
                    return

                if hot == 0 and lake == 0:
                    summary.empty += 1
                    summary.empties.append(sym)
                else:
                    summary.succeeded += 1
                    summary.rows_hot += hot
                    summary.rows_lake += lake

                if progress is not None and hasattr(progress, "advance"):
                    progress.advance(1)  # type: ignore[attr-defined]

        await asyncio.gather(*(one(s) for s in symbols))
        log.info("ingest_batch_complete", **summary.as_dict())
        return summary

    async def _create_instrument(self, session: object, symbol: str) -> Instrument:
        """Create a minimal instrument row, enriched from the profile if available."""
        profile: dict[str, object] = {}
        for provider in self.router.providers:
            fetch_profile = getattr(provider, "fetch_profile", None)
            if fetch_profile is not None:
                try:
                    profile = await fetch_profile(symbol)
                except Exception:  # noqa: BLE001 -- enrichment is optional
                    profile = {}
                if profile:
                    break

        instrument = Instrument(
            symbol=symbol,
            name=str(profile.get("name") or symbol),
            exchange=profile.get("exchange"),  # type: ignore[arg-type]
            sector=profile.get("sector"),  # type: ignore[arg-type]
            industry=profile.get("industry"),  # type: ignore[arg-type]
            country=profile.get("country"),  # type: ignore[arg-type]
            currency=str(profile.get("currency") or "USD"),
            market_cap=profile.get("market_cap"),  # type: ignore[arg-type]
            tier=Tier.COLD,
        )
        session.add(instrument)  # type: ignore[attr-defined]
        await session.flush()  # type: ignore[attr-defined]
        log.info("instrument_created", symbol=symbol, name=instrument.name)
        return instrument

    async def aclose(self) -> None:
        await self.router.aclose()
