"""Parquet lake writer (cold tier).

Layout is Hive-partitioned so DuckDB can prune whole directories from a
predicate without opening the files:

``data/lake/ohlcv/symbol=AAPL/year=2024/data.parquet``

Why this exists: the full US equity universe is ~7,000 tickers x ~1,260
trading days ~= 8.9M rows, roughly 1.5 GB in Postgres with indexes. Every free
Postgres tier is 0.5 GB. Parquet + zstd puts the same data in ~120 MB at zero
hosting cost, and DuckDB scans it fast enough to serve a request directly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from forecaster.config import get_settings
from forecaster.db.repositories.ohlcv import OHLCV_COLUMNS
from forecaster.logging import get_logger

log = get_logger(__name__)

OHLCV_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.date32()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("adj_close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("source", pa.string()),
    ]
)


def ohlcv_root(root: Path | None = None) -> Path:
    return (root or get_settings().lake_root) / "ohlcv"


def symbol_path(symbol: str, year: int, root: Path | None = None) -> Path:
    return ohlcv_root(root) / f"symbol={symbol.upper()}" / f"year={year}" / "data.parquet"


def write_bars(symbol: str, frame: pd.DataFrame, source: str, *, root: Path | None = None) -> int:
    """Write (or merge into) the lake, partitioned by symbol and year.

    Existing partitions are read, merged on ``ts`` with the new rows winning,
    and rewritten -- so re-ingesting an overlapping window is idempotent, the
    same guarantee the Postgres upsert gives.
    """
    if frame.empty:
        return 0

    out = frame.copy()
    if "source" not in out.columns:
        out["source"] = source
    out = out.reset_index().rename(columns={"index": "ts"})
    out["ts"] = pd.to_datetime(out["ts"]).dt.date

    written = 0
    for year, chunk in out.groupby(pd.DatetimeIndex(out["ts"]).year):
        path = symbol_path(symbol, int(year), root)
        path.parent.mkdir(parents=True, exist_ok=True)

        merged = chunk[["ts", *OHLCV_COLUMNS, "source"]]
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            merged = pd.concat([existing, merged], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ts"], keep="last")

        merged = merged.sort_values("ts").reset_index(drop=True)
        table = pa.Table.from_pandas(merged, schema=OHLCV_SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression="zstd", compression_level=3)
        written += len(chunk)

    log.debug("lake_write", symbol=symbol, rows=written)
    return written


def delete_symbol(symbol: str, *, root: Path | None = None) -> int:
    """Remove every partition for a symbol. Returns files deleted."""
    base = ohlcv_root(root) / f"symbol={symbol.upper()}"
    if not base.exists():
        return 0
    files = list(base.rglob("*.parquet"))
    for file in files:
        file.unlink()
    for directory in sorted(base.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if base.exists() and not any(base.iterdir()):
        base.rmdir()
    return len(files)


def lake_stats(*, root: Path | None = None) -> dict[str, object]:
    base = ohlcv_root(root)
    if not base.exists():
        return {"symbols": 0, "files": 0, "bytes": 0}
    files = list(base.rglob("*.parquet"))
    return {
        "symbols": len(list(base.glob("symbol=*"))),
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "root": str(base),
    }


def available_symbols(*, root: Path | None = None) -> list[str]:
    base = ohlcv_root(root)
    if not base.exists():
        return []
    return sorted(p.name.removeprefix("symbol=") for p in base.glob("symbol=*") if p.is_dir())


def coverage(symbol: str, *, root: Path | None = None) -> tuple[dt.date, dt.date, int] | None:
    """(first, last, rows) for a symbol in the lake, or None if absent."""
    base = ohlcv_root(root) / f"symbol={symbol.upper()}"
    if not base.exists():
        return None
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return None
    frames = [pq.read_table(f, columns=["ts"]).to_pandas() for f in files]
    all_ts = pd.concat(frames, ignore_index=True)["ts"]
    return all_ts.min(), all_ts.max(), len(all_ts)
