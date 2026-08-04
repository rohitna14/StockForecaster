"""Universe seeding.

Two jobs:

1. **Migrate the legacy ``stocks.db``.** The original project shipped a NASDAQ
   screener CSV dumped into SQLite, with ``Last Sale`` stored as the text
   ``"$141.71"`` and ``% Change`` as ``"0.049%"``. We keep the instrument
   metadata (symbol, name, sector, industry, market cap) and *discard* the
   price snapshot columns entirely -- they are a stale point-in-time artifact,
   not a time series, and keeping them would invite someone to use them.

2. **Define named universes** (S&P 500, NASDAQ 100, a small demo set) so
   backtests and batch jobs have a stable, reproducible symbol list.
"""

from __future__ import annotations

import io
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from forecaster.config import REPO_ROOT, get_settings
from forecaster.db.models import Universe, UniverseMember
from forecaster.db.repositories.instruments import InstrumentRepository
from forecaster.db.session import session_scope
from forecaster.logging import get_logger

log = get_logger(__name__)

LEGACY_DB_PATH = REPO_ROOT / "data" / "legacy" / "stocks.db"

#: Small, liquid, sector-diverse set used for demos, tests and CI.
DEMO_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "XOM", "JNJ", "WMT", "SPY",
]

_MONEY_RE = re.compile(r"[^0-9.\-]")


def _parse_money(value: Any) -> float | None:
    """``"$141.71"`` -> ``141.71``. Returns None for blanks and junk."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = _MONEY_RE.sub("", str(value))
    if not text or text in {"-", "."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_percent(value: Any) -> float | None:
    """``"0.049%"`` -> ``0.049``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _clean_name(raw: Any, symbol: str) -> str:
    """Strip the boilerplate share-class suffixes NASDAQ appends to every name."""
    name = str(raw or symbol).strip()
    for suffix in (
        " Common Stock", " Common Shares", " Ordinary Shares", " Class A Ordinary Shares",
        " Class B Ordinary Shares", " Common Stock (", " American Depositary Shares",
    ):
        if suffix in name:
            name = name.split(suffix)[0].strip()
    return name or symbol


def read_legacy_instruments(path: Path | None = None) -> pd.DataFrame:
    """Read the legacy screener dump into normalised instrument rows."""
    path = path or LEGACY_DB_PATH
    if not path.exists():
        log.info("legacy_db_absent", path=str(path))
        return pd.DataFrame()

    with sqlite3.connect(path) as conn:
        # The original code did `SELECT ... FROM {tables[0][0]}` -- the first
        # table in arbitrary order. Name the table explicitly instead.
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "stocks" not in tables:
            log.warning("legacy_table_missing", found=tables)
            return pd.DataFrame()
        raw = pd.read_sql_query("SELECT * FROM stocks", conn)

    log.info("legacy_db_read", rows=len(raw), columns=list(raw.columns))

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        symbol = str(row.get("Symbol") or "").strip().upper()
        # Screener files carry test tickers and blanks; also drop the handful of
        # symbols with characters yfinance/Stooq cannot address.
        if not symbol or len(symbol) > 10 or not re.fullmatch(r"[A-Z0-9.\-]+", symbol):
            continue

        market_cap = row.get("Market Cap")
        rows.append(
            {
                "symbol": symbol,
                "name": _clean_name(row.get("Name"), symbol),
                "asset_type": "equity",
                "sector": (str(row["Sector"]).strip() or None) if pd.notna(row.get("Sector")) else None,
                "industry": (str(row["Industry"]).strip() or None) if pd.notna(row.get("Industry")) else None,
                "country": (str(row["Country"]).strip() or None) if pd.notna(row.get("Country")) else None,
                "currency": "USD",
                "market_cap": float(market_cap) if pd.notna(market_cap) and market_cap else None,
                "ipo_year": int(row["IPO Year"]) if pd.notna(row.get("IPO Year")) else None,
                "exchange": None,
                "is_active": True,
            }
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="first")
    log.info("legacy_instruments_parsed", kept=len(frame), dropped=len(raw) - len(frame))
    return frame


async def seed_instruments_from_legacy(path: Path | None = None) -> int:
    frame = read_legacy_instruments(path)
    if frame.empty:
        return 0
    async with session_scope() as session:
        return await InstrumentRepository(session).upsert_many(frame.to_dict("records"))


#: Wikipedia blocks generic clients with 403; it wants a browser-shaped agent.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: Versioned CSV of S&P 500 members. Preferred over scraping: it is a stable
#: schema, needs no key, and does not break when Wikipedia restyles a table.
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def _http_get(url: str) -> httpx.Response:
    """Fetch with httpx, which ships certifi.

    ``pd.read_html(url)`` delegates to urllib and the OS certificate store,
    which fails outright on machines with a stale root bundle.
    """
    response = httpx.get(
        url,
        timeout=get_settings().http_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": _BROWSER_UA},
    )
    response.raise_for_status()
    return response


def _read_html_tables(url: str, match: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(_http_get(url).text), match=match)


def fetch_sp500_constituents() -> pd.DataFrame:
    """S&P 500 membership. Tries the CSV dataset, falls back to Wikipedia."""
    try:
        frame = pd.read_csv(io.StringIO(_http_get(SP500_CSV_URL).text))
        frame = frame.rename(
            columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector",
                     "GICS Sub-Industry": "industry"}
        )
        frame["symbol"] = (
            frame["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        )
        log.info("sp500_loaded", source="csv-dataset", count=len(frame))
        return frame[["symbol", "name", "sector", "industry"]]
    except Exception as exc:  # noqa: BLE001
        log.warning("sp500_csv_failed", error=str(exc))

    try:
        tables = _read_html_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", match="Symbol"
        )
    except Exception as exc:  # noqa: BLE001 -- offline is a valid state
        log.warning("sp500_fetch_failed", error=str(exc))
        return pd.DataFrame()

    table = tables[0].rename(
        columns={"Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "industry"}
    )
    table["symbol"] = table["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    log.info("sp500_loaded", source="wikipedia", count=len(table))
    return table[["symbol", "name", "sector", "industry"]]


#: Point-in-time NASDAQ-100 snapshot, used when Wikipedia is unreachable.
#: The index reconstitutes annually in December; refresh with `forecaster seed`
#: when the network path works, or edit this list.
NASDAQ100_FALLBACK = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR",
    "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]


def fetch_nasdaq100_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = _read_html_tables(url, match="Ticker")
    except Exception as exc:  # noqa: BLE001
        log.warning("nasdaq100_fetch_failed", error=str(exc), fallback="static list")
        return pd.DataFrame({"symbol": NASDAQ100_FALLBACK, "name": NASDAQ100_FALLBACK})

    table = tables[0]
    ticker_col = next((c for c in table.columns if "Ticker" in str(c)), None)
    name_col = next((c for c in table.columns if "Company" in str(c)), None)
    if ticker_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"symbol": table[ticker_col].astype(str).str.strip().str.upper()})
    out["name"] = table[name_col] if name_col else out["symbol"]
    return out


async def seed_universe(
    name: str, symbols: list[str], description: str = "", *, create_missing: bool = True
) -> int:
    """Create or refresh a named universe from a symbol list.

    Symbols absent from ``instruments`` are stubbed in rather than dropped --
    the legacy screener dump covers common equities only, so ETFs (SPY, QQQ)
    and recent listings would otherwise silently vanish from every universe.
    Ingestion enriches the stub with a real name and sector on first fetch.
    """
    from sqlalchemy import delete, select

    async with session_scope() as session:
        existing = (
            await session.execute(select(Universe).where(Universe.name == name))
        ).scalar_one_or_none()
        if existing is None:
            existing = Universe(name=name, description=description)
            session.add(existing)
            await session.flush()

        repo = InstrumentRepository(session)
        found = await repo.get_by_symbols(symbols)
        missing = sorted(set(s.upper() for s in symbols) - set(found))

        if missing and create_missing:
            await repo.upsert_many(
                [
                    {"symbol": s, "name": s, "asset_type": "unknown", "currency": "USD", "is_active": True}
                    for s in missing
                ]
            )
            await session.flush()
            found = await repo.get_by_symbols(symbols)
            log.info("universe_stubs_created", universe=name, count=len(missing), symbols=missing[:10])
        elif missing:
            log.warning("universe_symbols_missing", universe=name, count=len(missing), sample=missing[:10])

        await session.execute(
            delete(UniverseMember).where(UniverseMember.universe_id == existing.id)
        )
        today = pd.Timestamp.today().date()
        for instrument in found.values():
            session.add(
                UniverseMember(
                    universe_id=existing.id, instrument_id=instrument.id, added_at=today
                )
            )
        log.info("universe_seeded", universe=name, members=len(found))
        return len(found)


async def seed_all(*, include_index_universes: bool = True) -> dict[str, int]:
    """Full bootstrap: instruments from legacy, then the standard universes."""
    results: dict[str, int] = {}
    results["instruments"] = await seed_instruments_from_legacy()
    results["demo"] = await seed_universe("demo", DEMO_SYMBOLS, "Small liquid demo set")

    if include_index_universes:
        sp500 = fetch_sp500_constituents()
        if not sp500.empty:
            async with session_scope() as session:
                # Index members may not appear in the legacy screener dump.
                await InstrumentRepository(session).upsert_many(
                    sp500.assign(currency="USD", asset_type="equity", is_active=True).to_dict("records")
                )
            results["sp500"] = await seed_universe(
                "sp500", sp500["symbol"].tolist(), "S&P 500 constituents"
            )

        ndx = fetch_nasdaq100_constituents()
        if not ndx.empty:
            results["nasdaq100"] = await seed_universe(
                "nasdaq100", ndx["symbol"].tolist(), "NASDAQ-100 constituents"
            )

    return results
