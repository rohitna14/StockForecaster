"""In-memory fuzzy search over the instrument catalog.

Why in-memory rather than SQL: typo tolerance is the requirement, and SQL
``LIKE`` cannot do it. "aple" is not a substring of "Apple", so no combination
of ``LIKE`` patterns will ever match it. Postgres has ``pg_trgm``, SQLite has
nothing, and the project must run on both.

The catalog is ~7,000 rows of (symbol, name, sector, market cap) -- under a
megabyte. Loading it once and scoring in Python is both simpler and faster than
a round-trip, and it makes the ranking logic readable instead of buried in a
CASE expression.

Scoring combines four signals, highest wins:

===========================  =====  ==========================================
Signal                       Score  Example
===========================  =====  ==========================================
alias hit                     1000  "google"   -> GOOGL
exact symbol                   950  "AAPL"     -> AAPL
exact name (noise-stripped)    900  "apple"    -> Apple Inc.
symbol prefix                  800  "goog"     -> GOOG, GOOGL
name / name-token prefix       700  "mic"      -> Microsoft
substring in name              500  "cola"     -> Coca-Cola
fuzzy similarity              0-450 "aple"     -> Apple
===========================  =====  ==========================================

Ties break on market cap, so "app" surfaces Apple before Applied Signal.
"""

from __future__ import annotations

import difflib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from forecaster.logging import get_logger
from forecaster.search.aliases import (
    NAME_NOISE,
    fuzzy_alias,
    primary_symbols,
    resolve_alias,
)

log = get_logger(__name__)

#: Rebuild the index at most this often; the catalog changes rarely.
INDEX_TTL_SECONDS = 900

#: Below this similarity a fuzzy match is noise, not a typo.
FUZZY_FLOOR = 0.62

#: Primary listings, for breaking share-class ties (GOOGL over GOOG).
_PRIMARY = primary_symbols()


def prominence(entry: "Entry") -> float:
    """How likely this company is the one anyone means, 0-80.

    Match quality alone is not enough to rank a catalog containing both Apple
    and Apple Hospitality REIT. A $3.5T company and a $3B one can match a query
    equally well on text while being wildly different in how often they are the
    intended answer, so size and data availability are part of the score rather
    than a tiebreak applied after the fact.
    """
    import math

    cap = max(entry.market_cap, 0.0)
    # log10 of market cap: ~9 for a small cap, ~12.5 for a mega cap.
    size = min(math.log10(cap), 13.0) / 13.0 * 60.0 if cap > 1 else 0.0
    primary = 5.0 if entry.symbol in _PRIMARY else 0.0
    return size + primary + (20.0 if entry.has_data else 0.0)


def normalise(text: str) -> str:
    cleaned = text.lower().strip()
    for char in (".", ",", "'", '"', "(", ")", "/"):
        cleaned = cleaned.replace(char, "")
    for char in ("-", "_"):
        cleaned = cleaned.replace(char, " ")
    return " ".join(cleaned.split())


def strip_noise(name: str) -> str:
    """Remove legal-entity boilerplate so "apple" matches "Apple Inc.".

    Applied token-wise from the end, so a real word is never eaten from the
    middle of a name (e.g. "Co" in "Coca-Cola" survives).
    """
    tokens = normalise(name).split()

    # Share-class tails: "Alphabet Inc Class A" -> "alphabet". Dropped before
    # the generic noise pass so the single letter does not block it.
    for marker in ("class", "series"):
        if marker in tokens:
            position = tokens.index(marker)
            if position > 0 and len(tokens) - position <= 3:
                tokens = tokens[:position]
            break

    while tokens and tokens[-1] in NAME_NOISE:
        tokens.pop()

    text = " ".join(tokens)
    for phrase in ("common stock", "ordinary shares", "depositary shares"):
        text = text.replace(phrase, "").strip()
    return text or normalise(name)


@dataclass(slots=True)
class Entry:
    symbol: str
    name: str
    sector: str | None
    market_cap: float
    has_data: bool
    norm_name: str = field(default="", init=False)
    tokens: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        self.norm_name = strip_noise(self.name)
        self.tokens = tuple(self.norm_name.split())


@dataclass
class Match:
    entry: Entry
    score: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.entry.symbol,
            "name": self.entry.name,
            "sector": self.entry.sector,
            "market_cap": self.entry.market_cap or None,
            "has_data": self.entry.has_data,
            "score": round(self.score, 2),
            "match_reason": self.reason,
        }


class SearchIndex:
    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._by_symbol: dict[str, Entry] = {}
        self._built_at: float = 0.0
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────
    @property
    def is_stale(self) -> bool:
        return not self._entries or (time.monotonic() - self._built_at) > INDEX_TTL_SECONDS

    @property
    def size(self) -> int:
        return len(self._entries)

    def build(self, rows: list[dict[str, Any]]) -> None:
        entries = [
            Entry(
                symbol=str(row["symbol"]).upper(),
                name=str(row.get("name") or row["symbol"]),
                sector=row.get("sector"),
                market_cap=float(row.get("market_cap") or 0.0),
                has_data=bool(row.get("has_data")),
            )
            for row in rows
        ]
        with self._lock:
            self._entries = entries
            self._by_symbol = {e.symbol: e for e in entries}
            self._built_at = time.monotonic()
        log.info("search_index_built", entries=len(entries))

    def get(self, symbol: str) -> Entry | None:
        return self._by_symbol.get(symbol.upper())

    # ── querying ──────────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 10) -> list[Match]:
        raw = query.strip()
        if not raw:
            return []

        q = normalise(raw)
        q_upper = raw.strip().upper()

        # Both sides get the same treatment. Company names are stored with
        # legal boilerplate stripped ("Microsoft Corporation" -> "microsoft"),
        # so a query carrying that boilerplate must be stripped too -- otherwise
        # typing the *full correct legal name* scores worse than a typo, and
        # "microsoft corporation" loses to "Mint Incorporation Limited".
        q_stripped = strip_noise(raw)
        variants = {q, q_stripped} - {""}

        matches: dict[str, Match] = {}

        def offer(entry: Entry, score: float, reason: str) -> None:
            total = score + prominence(entry)
            existing = matches.get(entry.symbol)
            if existing is None or total > existing.score:
                matches[entry.symbol] = Match(entry, total, reason)

        # 1. Alias hits win outright -- "google" must never rank GOOGL second.
        for rank, symbol in enumerate(resolve_alias(q)):
            entry = self._by_symbol.get(symbol)
            if entry:
                offer(entry, 1000 - rank, "alias")

        # 1b. Misspelled common names. Scored just under an exact symbol match
        #     on raw text, but the prominence bonus lets a mega-cap alias beat
        #     an exact hit on an obscure ticker -- which is why "aple" resolves
        #     to Apple rather than Apple Hospitality REIT.
        if not matches:
            fuzzy_symbols, ratio = fuzzy_alias(q)
            for rank, symbol in enumerate(fuzzy_symbols):
                entry = self._by_symbol.get(symbol)
                if entry:
                    offer(entry, 1000 * ratio + 60 - rank, f"fuzzy alias ({ratio:.0%})")

        for entry in self._entries:
            symbol_lower = entry.symbol.lower()

            if entry.symbol == q_upper:
                offer(entry, 950, "exact symbol")
                continue

            matched = False
            for variant in variants:
                if entry.norm_name == variant:
                    offer(entry, 900, "exact name")
                elif symbol_lower.startswith(variant):
                    offer(entry, 800, "symbol prefix")
                elif entry.norm_name.startswith(variant):
                    offer(entry, 700, "name prefix")
                elif any(token.startswith(variant) for token in entry.tokens):
                    offer(entry, 690, "word prefix")
                elif variant in entry.norm_name:
                    offer(entry, 500, "name contains")
                elif variant in symbol_lower:
                    offer(entry, 460, "symbol contains")
                else:
                    continue
                matched = True
            if matched:
                continue

            # 2. Fuzzy fallback, for typos. Only worth computing when the
            #    lengths are comparable -- "aple" vs "Apple" yes, "aple" vs
            #    "Applied Digital Solutions Holdings" no.
            best_ratio = 0.0
            for variant in variants:
                if abs(len(entry.norm_name) - len(variant)) > max(4, len(variant) // 2):
                    continue
                best_ratio = max(
                    best_ratio,
                    difflib.SequenceMatcher(None, variant, entry.norm_name).ratio(),
                )
            if best_ratio >= FUZZY_FLOOR:
                offer(entry, 450 * best_ratio, "fuzzy name")
                continue

            if len(q) >= 3 and abs(len(symbol_lower) - len(q)) <= 2:
                ratio = difflib.SequenceMatcher(None, q, symbol_lower).ratio()
                if ratio >= FUZZY_FLOOR:
                    offer(entry, 440 * ratio, "fuzzy symbol")

        ranked = sorted(
            matches.values(),
            key=lambda m: (
                -m.score,
                # Prefer something the user can actually open, then size.
                not m.entry.has_data,
                -m.entry.market_cap,
                len(m.entry.symbol),
                m.entry.symbol,
            ),
        )
        return ranked[:limit]

    def best(self, query: str) -> Match | None:
        """Single best match -- what pressing Enter resolves to."""
        results = self.search(query, limit=1)
        return results[0] if results else None


_index = SearchIndex()


def get_index() -> SearchIndex:
    return _index
