"""Command-line interface.

forecaster db init                  create the schema
forecaster seed                     load instruments + universes
forecaster ingest AAPL MSFT --hot   fetch history into both tiers
forecaster ingest --universe demo --hot
forecaster status                   coverage across hot + cold tiers
forecaster promote NVDA             cold -> hot
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from forecaster.config import get_settings
from forecaster.logging import configure_logging, get_logger

app = typer.Typer(
    name="forecaster",
    help="Leakage-free walk-forward stock forecasting platform.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database schema management.", no_args_is_help=True)
lake_app = typer.Typer(help="Parquet cold-tier management.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(lake_app, name="lake")

console = Console()
log = get_logger(__name__)


@app.callback()
def main() -> None:
    configure_logging()


# ══════════════════════════════════ db ══════════════════════════════════
@db_app.command("init")
def db_init(
    drop: Annotated[bool, typer.Option("--drop", help="Drop existing tables first.")] = False,
) -> None:
    """Create all tables from the ORM metadata."""

    async def run() -> None:
        from forecaster.db.session import create_all, dispose_engine, drop_all

        if drop:
            confirm = typer.confirm("This destroys all data. Continue?", default=False)
            if not confirm:
                raise typer.Abort
            await drop_all()
            console.print("[yellow]Dropped all tables.[/]")
        await create_all()
        await dispose_engine()

    asyncio.run(run())
    console.print(f"[green]Schema ready[/] at {get_settings().database_url}")


@db_app.command("check")
def db_check() -> None:
    """Verify connectivity."""

    async def run() -> bool:
        from forecaster.db.session import dispose_engine, healthcheck

        ok = await healthcheck()
        await dispose_engine()
        return ok

    ok = asyncio.run(run())
    console.print("[green]Database reachable[/]" if ok else "[red]Database unreachable[/]")
    raise typer.Exit(0 if ok else 1)


# ═════════════════════════════════ seed ═════════════════════════════════
@app.command("seed")
def seed(
    skip_indices: Annotated[
        bool, typer.Option("--skip-indices", help="Skip Wikipedia index scraping (offline).")
    ] = False,
) -> None:
    """Load instrument metadata and named universes."""

    async def run() -> dict[str, int]:
        from forecaster.db.session import create_all, dispose_engine
        from forecaster.ingestion.seed import seed_all

        await create_all()
        result = await seed_all(include_index_universes=not skip_indices)
        await dispose_engine()
        return result

    results = asyncio.run(run())
    table = Table(title="Seed results", header_style="bold cyan")
    table.add_column("Target")
    table.add_column("Rows", justify="right")
    for key, value in results.items():
        table.add_row(key, f"{value:,}")
    console.print(table)


# ════════════════════════════════ ingest ════════════════════════════════
@app.command("ingest")
def ingest(
    symbols: Annotated[list[str] | None, typer.Argument(help="Ticker symbols.")] = None,
    universe: Annotated[str | None, typer.Option("--universe", "-u")] = None,
    years: Annotated[int, typer.Option("--years", "-y", help="Years of history.")] = 5,
    hot: Annotated[
        bool, typer.Option("--hot", help="Also write to the Postgres hot tier.")
    ] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
    limit: Annotated[int | None, typer.Option("--limit", help="Cap symbols (testing).")] = None,
) -> None:
    """Fetch OHLCV history into the lake (and optionally the hot tier)."""

    async def run() -> None:
        from sqlalchemy import select

        from forecaster.db.models import Instrument, Universe, UniverseMember
        from forecaster.db.session import create_all, dispose_engine, session_scope
        from forecaster.ingestion.service import IngestionService

        await create_all()

        target: list[str] = [s.upper() for s in (symbols or [])]
        if universe:
            async with session_scope() as session:
                stmt = (
                    select(Instrument.symbol)
                    .join(UniverseMember, UniverseMember.instrument_id == Instrument.id)
                    .join(Universe, Universe.id == UniverseMember.universe_id)
                    .where(Universe.name == universe, UniverseMember.removed_at.is_(None))
                    .order_by(Instrument.symbol)
                )
                target += list((await session.execute(stmt)).scalars().all())

        target = sorted(set(target))
        if limit:
            target = target[:limit]
        if not target:
            console.print("[red]No symbols to ingest.[/] Pass symbols or --universe.")
            raise typer.Exit(1)

        end = dt.date.today()
        start = end - dt.timedelta(days=365 * years)
        console.print(
            f"Ingesting [bold]{len(target)}[/] symbols "
            f"({start} to {end}), hot={hot}, concurrency={concurrency}"
        )

        service = IngestionService()
        with console.status("[cyan]Fetching...") as status:
            done = 0

            class _Progress:
                def advance(self, n: int = 1) -> None:
                    nonlocal done
                    done += n
                    status.update(f"[cyan]Fetching... {done}/{len(target)}")

            summary = await service.ingest_many(
                target,
                start=start,
                end=end,
                to_hot=True if hot else None,
                concurrency=concurrency,
                progress=_Progress(),
            )
        await service.aclose()
        await dispose_engine()

        table = Table(title="Ingestion summary", header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in summary.as_dict().items():
            table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
        console.print(table)

        if summary.failures:
            console.print(f"[yellow]{len(summary.failures)} failures[/], first 5:")
            for sym, err in list(summary.failures.items())[:5]:
                console.print(f"  [red]{sym}[/]: {err[:110]}")
        if summary.empties:
            console.print(
                f"[dim]{len(summary.empties)} symbols had no data: "
                f"{', '.join(summary.empties[:10])}[/]"
            )

    asyncio.run(run())


@app.command("promote")
def promote(
    symbols: Annotated[list[str], typer.Argument(help="Symbols to promote to the hot tier.")],
    years: Annotated[int, typer.Option("--years", "-y")] = 5,
) -> None:
    """Move symbols from the cold lake into Postgres."""

    async def run() -> None:
        from forecaster.db.session import create_all, dispose_engine
        from forecaster.ingestion.service import IngestionService

        await create_all()
        service = IngestionService()
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * years)
        for symbol in symbols:
            hot_rows, lake_rows = await service.ingest_symbol(
                symbol, start=start, end=end, to_hot=True
            )
            console.print(f"[green]{symbol.upper()}[/]: {hot_rows:,} hot / {lake_rows:,} lake rows")
        await service.aclose()
        await dispose_engine()

    asyncio.run(run())


# ════════════════════════════════ status ════════════════════════════════
@app.command("status")
def status() -> None:
    """Show coverage across both storage tiers."""

    async def run() -> None:
        from forecaster.db.models import Tier
        from forecaster.db.repositories.instruments import InstrumentRepository
        from forecaster.db.repositories.ohlcv import OHLCVRepository
        from forecaster.db.session import create_all, dispose_engine, session_scope
        from forecaster.lake import query as lake_query
        from forecaster.lake import writer as lake_writer

        await create_all()
        async with session_scope() as session:
            instruments = InstrumentRepository(session)
            total = await instruments.count()
            hot = await instruments.count(tier=Tier.HOT)
            bars = await OHLCVRepository(session).total_bars()
        await dispose_engine()

        stats = lake_writer.lake_stats()
        lake_rows = lake_query.total_rows()

        settings = get_settings()
        table = Table(title="StockForecaster status", header_style="bold cyan")
        table.add_column("Tier")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("hot", "database", settings.database_url.split("///")[-1][-48:])
        table.add_row("hot", "instruments", f"{total:,}")
        table.add_row("hot", "promoted (hot tier)", f"{hot:,}")
        table.add_row("hot", "daily bars", f"{bars:,}")
        table.add_row("cold", "symbols in lake", f"{stats['symbols']:,}")
        table.add_row("cold", "parquet files", f"{stats['files']:,}")
        table.add_row("cold", "lake size", f"{int(stats['bytes']) / 1e6:.1f} MB")
        table.add_row("cold", "daily bars", f"{lake_rows:,}")
        console.print(table)

    asyncio.run(run())


@lake_app.command("summary")
def lake_summary(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
) -> None:
    """Per-symbol coverage in the Parquet lake."""
    from forecaster.lake import query as lake_query

    frame = lake_query.universe_summary()
    if frame.empty:
        console.print("[yellow]Lake is empty.[/] Run `forecaster ingest` first.")
        return

    table = Table(title=f"Lake coverage ({len(frame)} symbols)", header_style="bold cyan")
    for col in ("symbol", "first_date", "last_date", "rows"):
        table.add_column(col, justify="right" if col == "rows" else "left")
    for _, row in frame.head(limit).iterrows():
        table.add_row(
            row["symbol"], str(row["first_date"]), str(row["last_date"]), f"{row['rows']:,}"
        )
    console.print(table)
    if len(frame) > limit:
        console.print(f"[dim]... and {len(frame) - limit} more[/]")


from forecaster import cli_evaluate  # noqa: E402 -- avoids a circular import at module load

cli_evaluate.register(app)


if __name__ == "__main__":
    app()
