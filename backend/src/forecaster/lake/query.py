"""DuckDB reader over the Parquet lake.

DuckDB runs in-process with no server, reads Hive partitions natively, and
pushes filters down so a single-symbol query touches only that symbol's
directory. This is what makes serving 7,000 tickers from a free tier viable.
"""

from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from forecaster.db.repositories.ohlcv import OHLCV_COLUMNS, apply_adjustment
from forecaster.lake.writer import ohlcv_root
from forecaster.logging import get_logger

log = get_logger(__name__)

_local = threading.local()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Thread-local connection. DuckDB connections are not thread-safe."""
    conn: duckdb.DuckDBPyConnection | None = getattr(_local, "conn", None)
    if conn is None:
        conn = duckdb.connect(database=":memory:")
        conn.execute("SET enable_progress_bar=false")
        conn.execute(f"SET threads TO {min(4, (duckdb.__standard_vector_size__ or 4))}")
        _local.conn = conn
    return conn


def _glob(root: Path | None = None, symbol: str | None = None) -> str:
    base = ohlcv_root(root)
    pattern = (
        f"{base}/symbol={symbol.upper()}/year=*/*.parquet"
        if symbol
        else f"{base}/symbol=*/year=*/*.parquet"
    )
    return pattern.replace("\\", "/")


def has_symbol(symbol: str, *, root: Path | None = None) -> bool:
    return (ohlcv_root(root) / f"symbol={symbol.upper()}").exists()


def read_bars(
    symbol: str,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    adjusted: bool = True,
    root: Path | None = None,
) -> pd.DataFrame:
    """Read one symbol's bars from the lake as a date-indexed frame.

    Mirrors ``OHLCVRepository.get_frame`` exactly so callers can switch tiers
    without noticing.
    """
    if not has_symbol(symbol, root=root):
        return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("ts")

    clauses: list[str] = []
    params: list[Any] = []
    if start:
        clauses.append("ts >= ?")
        params.append(start)
    if end:
        clauses.append("ts <= ?")
        params.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT ts, open, high, low, close, adj_close, volume
        FROM read_parquet('{_glob(root, symbol)}', hive_partitioning=true)
        {where}
        ORDER BY ts
    """
    frame = get_connection().execute(sql, params).df()
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS).rename_axis("ts")

    frame["ts"] = pd.to_datetime(frame["ts"])
    frame = frame.set_index("ts").sort_index()
    return apply_adjustment(frame) if adjusted else frame


def read_panel(
    symbols: list[str],
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    field: str = "adj_close",
    root: Path | None = None,
) -> pd.DataFrame:
    """Wide dates x symbols panel straight out of Parquet."""
    if not symbols:
        return pd.DataFrame()

    present = [s.upper() for s in symbols if has_symbol(s, root=root)]
    if not present:
        return pd.DataFrame()

    clauses = [f"symbol IN ({','.join('?' * len(present))})"]
    params: list[Any] = list(present)
    if start:
        clauses.append("ts >= ?")
        params.append(start)
    if end:
        clauses.append("ts <= ?")
        params.append(end)

    sql = f"""
        SELECT symbol, ts, {field}
        FROM read_parquet('{_glob(root)}', hive_partitioning=true)
        WHERE {" AND ".join(clauses)}
        ORDER BY ts
    """
    long = get_connection().execute(sql, params).df()
    if long.empty:
        return pd.DataFrame()
    long["ts"] = pd.to_datetime(long["ts"])
    return long.pivot(index="ts", columns="symbol", values=field).sort_index()


def universe_summary(*, root: Path | None = None) -> pd.DataFrame:
    """Per-symbol coverage across the whole lake, in one scan.

    Handy for the CLI and for the ``/health/ready`` payload.
    """
    base = ohlcv_root(root)
    if not base.exists() or not any(base.glob("symbol=*")):
        return pd.DataFrame(columns=["symbol", "first_date", "last_date", "rows"])

    sql = f"""
        SELECT symbol,
               MIN(ts)   AS first_date,
               MAX(ts)   AS last_date,
               COUNT(*)  AS rows
        FROM read_parquet('{_glob(root)}', hive_partitioning=true)
        GROUP BY symbol
        ORDER BY symbol
    """
    return get_connection().execute(sql).df()


def total_rows(*, root: Path | None = None) -> int:
    base = ohlcv_root(root)
    if not base.exists() or not any(base.glob("symbol=*")):
        return 0
    sql = f"SELECT COUNT(*) FROM read_parquet('{_glob(root)}', hive_partitioning=true)"
    return int(get_connection().execute(sql).fetchone()[0])  # type: ignore[index]
