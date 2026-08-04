"""Search relevance tests.

These encode the actual user-facing contract: someone types a nickname, a
partial, or a typo, and the right company comes back first.
"""

from __future__ import annotations

import pytest

from forecaster.search.index import SearchIndex, strip_noise

CATALOG = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "market_cap": 3.8e12, "has_data": True},
    {"symbol": "APLE", "name": "Apple Hospitality REIT Inc.", "sector": "Real Estate", "market_cap": 2.8e9, "has_data": False},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "market_cap": 3.9e12, "has_data": True},
    {"symbol": "MCHPP", "name": "Microchip Technology Incorporated", "sector": "Technology", "market_cap": 3.3e10, "has_data": False},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)", "sector": "Technology", "market_cap": 3.0e12, "has_data": True},
    {"symbol": "GOOG", "name": "Alphabet Inc. (Class C)", "sector": "Technology", "market_cap": 3.0e12, "has_data": True},
    {"symbol": "GOOS", "name": "Canada Goose Holdings Inc.", "sector": "Consumer", "market_cap": 1.4e9, "has_data": False},
    {"symbol": "META", "name": "Meta Platforms", "sector": "Technology", "market_cap": 1.5e12, "has_data": True},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "sector": "Consumer", "market_cap": 1.4e12, "has_data": True},
    {"symbol": "AEHR", "name": "Aehr Test Systems", "sector": "Technology", "market_cap": 9.3e8, "has_data": False},
    {"symbol": "NVDA", "name": "Nvidia", "sector": "Technology", "market_cap": 3.4e12, "has_data": True},
    {"symbol": "KO", "name": "Coca-Cola Company (The)", "sector": "Consumer", "market_cap": 3.7e11, "has_data": True},
]


@pytest.fixture(scope="module")
def index() -> SearchIndex:
    idx = SearchIndex()
    idx.build(CATALOG)
    return idx


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Tickers
        ("AAPL", "AAPL"), ("tsla", "TSLA"), ("nvda", "NVDA"),
        # Full names, with and without legal boilerplate
        ("Apple Inc", "AAPL"), ("Apple Inc.", "AAPL"),
        ("Microsoft Corporation", "MSFT"), ("Tesla Inc", "TSLA"),
        ("Meta Platforms", "META"), ("Alphabet Inc", "GOOGL"),
        # Nicknames / common names
        ("google", "GOOGL"), ("facebook", "META"), ("apple", "AAPL"),
        ("microsoft", "MSFT"), ("tesla", "TSLA"), ("coca cola", "KO"),
        # Partials
        ("mic", "MSFT"), ("tes", "TSLA"), ("goo", "GOOGL"),
        # Typos
        ("aple", "AAPL"), ("appl", "AAPL"), ("gogle", "GOOGL"),
        ("googel", "GOOGL"), ("teslla", "TSLA"), ("nvdia", "NVDA"),
        ("micrsoft", "MSFT"), ("fcebook", "META"),
    ],
)
def test_best_match(index: SearchIndex, query: str, expected: str) -> None:
    """Pressing Enter must land on the company the user meant."""
    best = index.best(query)
    assert best is not None, f"{query!r} matched nothing"
    assert best.entry.symbol == expected, (
        f"{query!r} -> {best.entry.symbol} ({best.reason}), expected {expected}"
    )


def test_prominence_beats_an_obscure_exact_ticker(index: SearchIndex) -> None:
    """"aple" is an exact hit on APLE, but nobody means Apple Hospitality REIT.

    Match quality alone cannot rank this; company prominence has to be part of
    the score.
    """
    results = index.search("aple", limit=3)
    assert results[0].entry.symbol == "AAPL"
    # The literal match is still offered, just not first.
    assert "APLE" in {r.entry.symbol for r in results}


def test_share_classes_both_returned(index: SearchIndex) -> None:
    symbols = {r.entry.symbol for r in index.search("google", limit=5)}
    assert {"GOOGL", "GOOG"} <= symbols


def test_results_are_ranked_descending(index: SearchIndex) -> None:
    results = index.search("apple", limit=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(index: SearchIndex) -> None:
    assert index.search("") == []
    assert index.search("   ") == []


def test_nonsense_query_does_not_crash(index: SearchIndex) -> None:
    assert index.search("qqqzzzxxx123") == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Apple Inc.", "apple"),
        ("Microsoft Corporation", "microsoft"),
        ("Alphabet Inc. (Class A)", "alphabet"),
        ("Tesla, Inc.", "tesla"),
        ("Coca-Cola Company (The)", "coca cola"),
    ],
)
def test_strip_noise(raw: str, expected: str) -> None:
    assert strip_noise(raw) == expected
