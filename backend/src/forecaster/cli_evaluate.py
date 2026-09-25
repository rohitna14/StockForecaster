"""``forecaster evaluate`` -- the command that produces every published number.

Registered onto the main Typer app in :mod:`forecaster.cli`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import warnings
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from forecaster.config import REPO_ROOT
from forecaster.features.targets import TargetType
from forecaster.logging import get_logger

console = Console()
log = get_logger(__name__)

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "SPY",
]
DEFAULT_MODELS = ["ridge", "elastic_net", "random_forest", "gradient_boosting", "lightgbm"]


def register(app: typer.Typer) -> None:
    app.command("evaluate")(evaluate)
    app.command("backtest")(backtest)
    app.command("ingest-all")(ingest_all)


def ingest_all(
    years: Annotated[int, typer.Option("--years", "-y")] = 5,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 3,
    limit: Annotated[int | None, typer.Option("--limit", help="Cap symbols (testing).")] = None,
    skip_existing: Annotated[bool, typer.Option("--skip-existing/--refresh")] = True,
    hot: Annotated[bool, typer.Option("--hot", help="Also write to the Postgres hot tier.")] = False,
    priority_only: Annotated[bool, typer.Option("--priority", help="Only well-known tickers.")] = False,
) -> None:
    """Ingest the entire catalog. Intended to run overnight.

    Note this is NOT required for a symbol to work -- opening any ticker
    ingests it on demand in about a second. This exists to warm the cache in
    bulk so the first visitor never waits.

    Realistic timing: ~6,600 symbols at concurrency 3 takes several hours, and
    free providers will throttle. --priority does the ~500 names people
    actually search first, which takes a few minutes and covers most traffic.
    """
    import asyncio as _asyncio

    warnings.filterwarnings("ignore")

    async def run() -> None:
        from sqlalchemy import select

        from forecaster.db.models import Instrument
        from forecaster.db.session import create_all, dispose_engine, session_scope
        from forecaster.ingestion.service import IngestionService
        from forecaster.lake import query as lake_query
        from forecaster.search.aliases import alias_targets

        await create_all()

        async with session_scope() as session:
            stmt = select(Instrument.symbol, Instrument.market_cap).where(
                Instrument.is_active.is_(True)
            )
            rows = (await session.execute(stmt)).all()

        # Biggest companies first: if the run is interrupted, the symbols people
        # actually search are already done.
        ordered = sorted(rows, key=lambda r: -(r[1] or 0))
        symbols = [r[0] for r in ordered]

        if priority_only:
            wanted = alias_targets()
            symbols = [s for s in symbols if s in wanted] + symbols[:500]
            symbols = list(dict.fromkeys(symbols))

        if skip_existing:
            before = len(symbols)
            symbols = [s for s in symbols if not lake_query.has_symbol(s)]
            console.print(f"[dim]{before - len(symbols)} already ingested, skipping[/]")

        if limit:
            symbols = symbols[:limit]

        if not symbols:
            console.print("[green]Nothing to ingest.[/]")
            return

        console.print(
            f"Ingesting [bold]{len(symbols)}[/] symbols, {years}y, "
            f"concurrency={concurrency}, hot={hot}"
        )
        console.print("[dim]Free providers throttle; this is deliberately unhurried.[/]")

        service = IngestionService()
        with console.status("[cyan]Starting...") as status:
            done = 0

            class _Progress:
                def advance(self, n: int = 1) -> None:
                    nonlocal done
                    done += n
                    pct = done / len(symbols) * 100
                    status.update(f"[cyan]{done}/{len(symbols)} ({pct:.1f}%)")

            summary = await service.ingest_many(
                symbols,
                start=dt.date.today() - dt.timedelta(days=365 * years),
                end=dt.date.today(),
                to_hot=True if hot else None,
                concurrency=concurrency,
                progress=_Progress(),
            )
        await service.aclose()
        await dispose_engine()

        table = Table(title="Bulk ingest", header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in summary.as_dict().items():
            table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
        console.print(table)

        if summary.failures:
            console.print(f"[yellow]{len(summary.failures)} failed[/] (likely throttling; re-run to resume)")

    _asyncio.run(run())


def evaluate(
    symbols: Annotated[list[str] | None, typer.Argument(help="Symbols to evaluate.")] = None,
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated model names.")
    ] = ",".join(DEFAULT_MODELS),
    horizons: Annotated[
        str, typer.Option("--horizons", help="Comma-separated horizons.")
    ] = "1,5,10,21",
    target: Annotated[
        str, typer.Option("--target", "-t", help="return | vol_ratio | direction | log_return")
    ] = "return",
    train_size: Annotated[int, typer.Option("--train-size")] = 504,
    test_size: Annotated[int, typer.Option("--test-size")] = 63,
    embargo: Annotated[int, typer.Option("--embargo")] = 2,
    feature_set: Annotated[str, typer.Option("--features", help="minimal|core|full|all")] = "core",
    persist: Annotated[
        bool, typer.Option("--persist/--no-persist", help="Write runs to the database.")
    ] = True,
    write_results: Annotated[
        bool,
        typer.Option("--write-results", help="Regenerate docs/RESULTS.md (runs both targets)."),
    ] = False,
    deep: Annotated[bool, typer.Option("--deep", help="Include sequence models (slow).")] = False,
) -> None:
    """Run walk-forward evaluation and report the leaderboard.

    With --write-results, runs BOTH the directional and volatility experiments
    across all horizons and regenerates docs/RESULTS.md from the output.
    """
    warnings.filterwarnings("ignore")

    from forecaster.validation.harness import EvaluationConfig, EvaluationHarness

    symbol_list = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
    if models.strip().lower() == "all":
        # Every registered model. Classifiers are filtered per-target by the
        # harness, so this is safe to pass for regression targets too.
        from forecaster.models.registry import available_models

        model_list = available_models()
        deep = True
    else:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    if deep:
        model_list = list(dict.fromkeys([*model_list, "lstm", "gru", "tcn", "transformer"]))
    horizon_list = [int(h) for h in horizons.split(",") if h.strip()]

    targets = [TargetType.RETURN, TargetType.VOL_RATIO] if write_results else [TargetType(target)]

    all_rows: dict[str, list[dict[str, Any]]] = {str(t): [] for t in targets}
    reports: list[Any] = []

    total = len(targets) * len(horizon_list) * len(symbol_list)
    console.print(
        f"[bold cyan]Evaluating[/] {len(symbol_list)} symbols x {len(horizon_list)} horizons "
        f"x {len(targets)} target(s) = {total} runs, {len(model_list)} models each"
    )

    with console.status("[cyan]Running walk-forward...") as status:
        done = 0
        for target_type in targets:
            for horizon in horizon_list:
                for symbol in symbol_list:
                    done += 1
                    status.update(f"[cyan]{target_type} h={horizon} {symbol} ({done}/{total})")

                    frame = _load_bars(symbol)
                    if frame is None or len(frame) < train_size + 200:
                        console.print(f"  [yellow]skip {symbol}[/]: insufficient history")
                        continue

                    cfg = EvaluationConfig(
                        symbol=symbol,
                        models=model_list,
                        horizon=horizon,
                        target_type=target_type,
                        feature_set=feature_set,
                        train_size=train_size,
                        test_size=test_size,
                        embargo=embargo,
                    )
                    try:
                        report = EvaluationHarness(cfg).run(frame)
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"  [red]fail {symbol} h={horizon}[/]: {exc}")
                        continue

                    reports.append(report)
                    all_rows[str(target_type)].extend(_flatten(report, symbol, horizon))

                    if persist:
                        from forecaster.validation.persistence import persist_report

                        asyncio.run(persist_report(report))

    if not any(all_rows.values()):
        console.print("[red]No successful runs.[/]")
        raise typer.Exit(1)

    for target_name, rows in all_rows.items():
        if rows:
            _print_leaderboard(pd.DataFrame(rows), target_name)

    if write_results:
        _write_results_doc(
            all_rows,
            symbol_list,
            {
                "train_size": train_size,
                "test_size": test_size,
                "embargo": embargo,
                "mode": "rolling",
            },
        )


def backtest(
    symbol: Annotated[str, typer.Argument(help="Symbol to backtest.")],
    model: Annotated[str, typer.Option("--model", "-m")] = "lightgbm",
    horizon: Annotated[int, typer.Option("--horizon", "-h")] = 1,
    costs: Annotated[
        str, typer.Option("--costs", help="zero|optimistic|realistic|conservative")
    ] = "realistic",
    sweep: Annotated[bool, typer.Option("--sweep", help="Run every cost preset.")] = False,
) -> None:
    """Backtest a model's out-of-sample predictions with transaction costs."""
    warnings.filterwarnings("ignore")

    import numpy as np

    from forecaster.backtest.costs import COST_PRESETS
    from forecaster.backtest.engine import BacktestConfig, Backtester, sweep_costs
    from forecaster.validation.harness import EvaluationConfig, EvaluationHarness

    frame = _load_bars(symbol)
    if frame is None:
        console.print(f"[red]No data for {symbol}.[/] Run `forecaster ingest {symbol}` first.")
        raise typer.Exit(1)

    cfg = EvaluationConfig(symbol=symbol.upper(), models=[model], horizon=horizon)
    report = EvaluationHarness(cfg).run(frame)
    result = report.results.get(model)
    if result is None or not result.succeeded:
        console.print(f"[red]Model {model} failed:[/] {result.error if result else 'missing'}")
        raise typer.Exit(1)

    # Stitch out-of-sample predictions back into one date-indexed series.
    dates = np.concatenate([f.dates.to_numpy() for f in result.folds])
    preds = np.concatenate([f.y_pred for f in result.folds])
    predictions = pd.Series(preds, index=pd.DatetimeIndex(dates)).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    if sweep:
        table = Table(
            title=f"{symbol.upper()} / {model} -- cost sensitivity", header_style="bold cyan"
        )
        for col in (
            "preset",
            "round_trip_bps",
            "total_return",
            "sharpe",
            "max_drawdown",
            "n_trades",
            "benchmark_return",
        ):
            table.add_column(col.replace("_", " "), justify="right")
        for _, row in sweep_costs(frame["close"], predictions, frame["open"]).iterrows():
            table.add_row(
                str(row["preset"]),
                f"{row['round_trip_bps']:.1f}",
                f"{row['total_return']:+.2%}",
                f"{row['sharpe']:.2f}",
                f"{row['max_drawdown']:.2%}",
                str(int(row["n_trades"])),
                f"{row['benchmark_return']:+.2%}",
            )
        console.print(table)
        return

    bt = Backtester(BacktestConfig(costs=COST_PRESETS[costs]))
    outcome = bt.run(frame["close"], predictions, opens=frame["open"])

    table = Table(
        title=f"{symbol.upper()} / {model} backtest ({costs} costs)", header_style="bold cyan"
    )
    table.add_column("Metric")
    table.add_column("Strategy", justify="right")
    table.add_column("Buy & hold", justify="right")
    s = outcome.stats
    table.add_row("Total return", f"{s['total_return']:+.2%}", f"{s['benchmark_return']:+.2%}")
    table.add_row("Sharpe", f"{s['sharpe']:.2f}", f"{s['benchmark_sharpe']:.2f}")
    table.add_row("Max drawdown", f"{s['max_drawdown']:.2%}", "--")
    table.add_row("Trades", str(s["n_trades"]), "1")
    table.add_row("Costs paid", f"{s['total_costs']:.2%}", "--")
    table.add_row("Cost drag on Sharpe", f"{s['cost_drag_on_sharpe']:.2f}", "--")
    console.print(table)
    console.print(f"\n[dim]{outcome.summary()}[/]")


# ── helpers ───────────────────────────────────────────────────────────────
def _load_bars(symbol: str) -> pd.DataFrame | None:
    from forecaster.lake import query as lake_query

    frame = lake_query.read_bars(symbol)
    return None if frame.empty else frame


def _flatten(report: Any, symbol: str, horizon: int) -> list[dict[str, Any]]:
    rows = []
    for name, result in report.results.items():
        if not result.succeeded:
            continue
        agg = result.aggregate
        rows.append(
            {
                "horizon": horizon,
                "symbol": symbol,
                "model": name,
                "family": result.family,
                "rmse": agg.get("rmse", {}).get("mean"),
                "mase": agg.get("mase", {}).get("mean"),
                "r2": agg.get("r2", {}).get("mean"),
                "hit_rate": agg.get("directional_accuracy", {}).get("mean"),
                "base_rate": agg.get("base_rate", {}).get("mean"),
                "rmse_skill_pct": result.skill.get("rmse", {}).get("improvement_pct"),
                "dir_skill_pct": result.skill.get("directional", {}).get("improvement_pct"),
                "dm_p": result.dm_test.get("dm_pvalue"),
                "coverage": result.interval_coverage,
                "folds": len(result.folds),
            }
        )
    return rows


def _print_leaderboard(frame: pd.DataFrame, target_name: str) -> None:
    from forecaster.validation.report import pool_results

    pooled = pool_results(frame)
    table = Table(title=f"Pooled leaderboard -- target={target_name}", header_style="bold cyan")
    for col in (
        "horizon",
        "model",
        "rmse_skill_pct",
        "hit_rate",
        "base_rate",
        "mase",
        "r2",
        "coverage",
        "n",
    ):
        if col in pooled.columns:
            table.add_column(col.replace("_", " "), justify="right")

    for _, row in pooled.sort_values(
        ["horizon", "rmse_skill_pct"], ascending=[True, False]
    ).iterrows():
        is_baseline = row["model"] in {"naive_last_price", "historical_mean"}
        style = "dim" if is_baseline else ""
        cells = [f"{int(row['horizon'])}d", str(row["model"])]
        for col in ("rmse_skill_pct", "hit_rate", "base_rate", "mase", "r2", "coverage"):
            if col in pooled.columns:
                value = row.get(col)
                cells.append(
                    "--"
                    if pd.isna(value)
                    else (f"{value:+.2f}%" if col == "rmse_skill_pct" else f"{value:.4f}")
                )
        if "n" in pooled.columns:
            cells.append(str(int(row["n"])))
        table.add_row(*cells, style=style)
    console.print(table)


def _write_results_doc(
    all_rows: dict[str, list[dict[str, Any]]],
    symbols: list[str],
    config_note: dict[str, Any],
) -> None:
    from forecaster.validation.report import pool_results, write_results

    directional = pool_results(pd.DataFrame(all_rows.get("return", [])))
    volatility_raw = pd.DataFrame(all_rows.get("vol_ratio", []))
    volatility = pool_results(volatility_raw)

    per_symbol = pd.DataFrame()
    if not volatility_raw.empty:
        headline_h = (
            5 if 5 in set(volatility_raw["horizon"]) else int(volatility_raw["horizon"].min())
        )
        subset = volatility_raw[
            (volatility_raw["horizon"] == headline_h)
            & (~volatility_raw["model"].isin(["naive_last_price", "historical_mean"]))
        ]
        if not subset.empty:
            per_symbol = subset.loc[subset.groupby("symbol")["rmse_skill_pct"].idxmax()]

    path = write_results(
        REPO_ROOT / "docs" / "RESULTS.md",
        directional=directional,
        volatility=volatility,
        symbols=symbols,
        config_note=config_note,
        per_symbol_vol=per_symbol,
    )
    console.print(f"\n[green]Wrote[/] {path.relative_to(REPO_ROOT)}")
