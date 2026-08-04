"""Company fundamentals, quotes and news.

Sourced from yfinance's quote-summary payload. Everything here is *best effort*
and heavily defended: the shape of that payload is undocumented, varies by
listing type (an ETF has no P/E, a foreign ADR often has no employee count), and
changes without notice. A missing field yields ``None`` rather than an
exception, because a partially populated company page is fine and a 500 is not.

**On "live" prices.** yfinance quotes are delayed, typically ~15 minutes for US
equities. Real-time tick data requires a paid feed. Every quote carries
``is_delayed`` and ``as_of`` so the UI can say so plainly rather than implying
something the data does not support.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from typing import Any

from forecaster.logging import get_logger

log = get_logger(__name__)

#: Quotes are cached briefly -- long enough to stop a page with several widgets
#: firing several upstream calls, short enough to still feel current.
QUOTE_TTL_SECONDS = 60
PROFILE_TTL_SECONDS = 3600

_quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _clean(value: Any) -> Any:
    """Normalise yfinance's mixed sentinels into None."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric == 0:
            # yfinance uses 0 for "unknown" on most numeric fields; a genuine
            # zero market cap or P/E is not meaningful anyway.
            return None
        return numeric
    return value


def _timestamp_to_date(value: Any) -> str | None:
    try:
        if not value:
            return None
        return dt.datetime.fromtimestamp(float(value), tz=dt.UTC).date().isoformat()
    except (ValueError, OSError, TypeError):
        return None


async def get_profile(symbol: str) -> dict[str, Any]:
    """Company fundamentals: valuation, size, sector, analyst view."""
    symbol = symbol.upper()
    cached = _profile_cache.get(symbol)
    now = asyncio.get_running_loop().time()
    if cached and now - cached[0] < PROFILE_TTL_SECONDS:
        return cached[1]

    payload = await asyncio.to_thread(_fetch_info, symbol)
    profile = _shape_profile(symbol, payload)
    _profile_cache[symbol] = (now, profile)
    return profile


def _fetch_info(symbol: str) -> dict[str, Any]:
    try:
        import yfinance as yf

        return yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001 -- profile is optional enrichment
        log.debug("profile_fetch_failed", symbol=symbol, error=str(exc))
        return {}


def _shape_profile(symbol: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": _clean(info.get("longName")) or _clean(info.get("shortName")) or symbol,
        "exchange": _clean(info.get("fullExchangeName")) or _clean(info.get("exchange")),
        "sector": _clean(info.get("sector")),
        "industry": _clean(info.get("industry")),
        "country": _clean(info.get("country")),
        "website": _clean(info.get("website")),
        "employees": _clean(info.get("fullTimeEmployees")),
        "summary": _clean(info.get("longBusinessSummary")),
        "currency": _clean(info.get("currency")) or "USD",
        # ── valuation ──
        "market_cap": _clean(info.get("marketCap")),
        "enterprise_value": _clean(info.get("enterpriseValue")),
        "trailing_pe": _clean(info.get("trailingPE")),
        "forward_pe": _clean(info.get("forwardPE")),
        "peg_ratio": _clean(info.get("pegRatio")),
        "price_to_book": _clean(info.get("priceToBook")),
        "eps_trailing": _clean(info.get("trailingEps")),
        "eps_forward": _clean(info.get("forwardEps")),
        "profit_margin": _clean(info.get("profitMargins")),
        "revenue": _clean(info.get("totalRevenue")),
        "revenue_growth": _clean(info.get("revenueGrowth")),
        # ── risk / income ──
        "beta": _clean(info.get("beta")),
        "dividend_yield": _clean(info.get("dividendYield")),
        "dividend_rate": _clean(info.get("dividendRate")),
        "payout_ratio": _clean(info.get("payoutRatio")),
        # ── trading ──
        "fifty_two_week_high": _clean(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _clean(info.get("fiftyTwoWeekLow")),
        "fifty_day_average": _clean(info.get("fiftyDayAverage")),
        "two_hundred_day_average": _clean(info.get("twoHundredDayAverage")),
        "average_volume": _clean(info.get("averageVolume")),
        "shares_outstanding": _clean(info.get("sharesOutstanding")),
        "float_shares": _clean(info.get("floatShares")),
        "short_ratio": _clean(info.get("shortRatio")),
        # ── analyst view ──
        "target_mean_price": _clean(info.get("targetMeanPrice")),
        "target_high_price": _clean(info.get("targetHighPrice")),
        "target_low_price": _clean(info.get("targetLowPrice")),
        "recommendation": _clean(info.get("recommendationKey")),
        "analyst_count": _clean(info.get("numberOfAnalystOpinions")),
        # ── calendar ──
        "earnings_date": _timestamp_to_date(info.get("earningsTimestamp")),
        "ex_dividend_date": _timestamp_to_date(info.get("exDividendDate")),
    }


async def get_quote(symbol: str) -> dict[str, Any]:
    """Most recent price available.

    Delayed, not real-time. ``is_delayed`` is always True on this data source
    and the UI is expected to surface it.
    """
    symbol = symbol.upper()
    cached = _quote_cache.get(symbol)
    now = asyncio.get_running_loop().time()
    if cached and now - cached[0] < QUOTE_TTL_SECONDS:
        return cached[1]

    quote = await asyncio.to_thread(_fetch_quote, symbol)
    _quote_cache[symbol] = (now, quote)
    return quote


def _fetch_quote(symbol: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "symbol": symbol,
        "price": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
        "market_state": None,
        "as_of": None,
        "is_delayed": True,
        "delay_note": "Prices are delayed ~15 minutes. Not a real-time feed.",
    }
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info

        price = _clean(getattr(fast, "last_price", None))
        previous = _clean(getattr(fast, "previous_close", None))

        result.update(
            {
                "price": price,
                "previous_close": previous,
                "day_high": _clean(getattr(fast, "day_high", None)),
                "day_low": _clean(getattr(fast, "day_low", None)),
                "volume": _clean(getattr(fast, "last_volume", None)),
                "as_of": dt.datetime.now(dt.UTC).isoformat(),
            }
        )
        if price is not None and previous:
            result["change"] = round(price - previous, 4)
            result["change_percent"] = round(price / previous - 1.0, 6)
    except Exception as exc:  # noqa: BLE001 -- fall back to stored close
        log.debug("quote_fetch_failed", symbol=symbol, error=str(exc))
    return result


async def get_news(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    """Recent headlines. Availability varies a lot by ticker."""
    return await asyncio.to_thread(_fetch_news, symbol, limit)


def _fetch_news(symbol: str, limit: int) -> list[dict[str, Any]]:
    try:
        import yfinance as yf

        raw = yf.Ticker(symbol).news or []
    except Exception as exc:  # noqa: BLE001
        log.debug("news_fetch_failed", symbol=symbol, error=str(exc))
        return []

    items: list[dict[str, Any]] = []
    for article in raw[:limit]:
        # yfinance has moved this payload around; accept both shapes.
        content = article.get("content", article)
        title = _clean(content.get("title"))
        if not title:
            continue

        provider = content.get("provider") or {}
        link = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or article.get("link")
        )
        published = content.get("pubDate") or _timestamp_to_date(
            article.get("providerPublishTime")
        )

        items.append(
            {
                "title": title,
                "publisher": _clean(provider.get("displayName"))
                or _clean(article.get("publisher")),
                "url": link,
                "published_at": published,
                "summary": _clean(content.get("summary")),
            }
        )
    return items
