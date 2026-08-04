"""Common-name aliases.

People search for what they *call* a company, not its legal name. Nobody types
"Alphabet Inc. Class C" -- they type "google". Nobody types "Meta Platforms" --
they type "facebook", and often mean the app rather than the ticker.

This is a curated map because there is no free API for it. It only needs to
cover the names people actually type; everything else is handled by fuzzy
matching against the real company name.

Keys are normalised (lowercase, no punctuation). Values are ticker symbols in
priority order -- the first is what pressing Enter resolves to.
"""

from __future__ import annotations

ALIASES: dict[str, list[str]] = {
    # ── Big tech, where the brand differs from the legal name ─────────────
    "google": ["GOOGL", "GOOG"],
    "google stock": ["GOOGL"],
    "alphabet": ["GOOGL", "GOOG"],
    "youtube": ["GOOGL"],
    "android": ["GOOGL"],
    "waymo": ["GOOGL"],
    "gmail": ["GOOGL"],
    "facebook": ["META"],
    "fb": ["META"],
    "instagram": ["META"],
    "whatsapp": ["META"],
    "oculus": ["META"],
    "meta": ["META"],
    "apple": ["AAPL"],
    "iphone": ["AAPL"],
    "ipad": ["AAPL"],
    "macbook": ["AAPL"],
    "mac": ["AAPL"],
    "microsoft": ["MSFT"],
    "windows": ["MSFT"],
    "xbox": ["MSFT"],
    "azure": ["MSFT"],
    "office": ["MSFT"],
    "linkedin": ["MSFT"],
    "amazon": ["AMZN"],
    "aws": ["AMZN"],
    "prime": ["AMZN"],
    "kindle": ["AMZN"],
    "twitch": ["AMZN"],
    "nvidia": ["NVDA"],
    "tesla": ["TSLA"],
    "spacex": ["TSLA"],  # not public; Tesla is the closest tradable proxy
    "netflix": ["NFLX"],
    "disney": ["DIS"],
    "marvel": ["DIS"],
    "pixar": ["DIS"],
    "espn": ["DIS"],
    "hulu": ["DIS"],

    # ── Consumer brands ───────────────────────────────────────────────────
    "coke": ["KO"],
    "coca cola": ["KO"],
    "cocacola": ["KO"],
    "pepsi": ["PEP"],
    "mcdonalds": ["MCD"],
    "mcdonald": ["MCD"],
    "starbucks": ["SBUX"],
    "nike": ["NKE"],
    "walmart": ["WMT"],
    "target": ["TGT"],
    "costco": ["COST"],
    "home depot": ["HD"],
    "lowes": ["LOW"],
    "chipotle": ["CMG"],
    "dominos": ["DPZ"],
    "kfc": ["YUM"],
    "taco bell": ["YUM"],
    "pizza hut": ["YUM"],
    "airbnb": ["ABNB"],
    "uber": ["UBER"],
    "lyft": ["LYFT"],
    "doordash": ["DASH"],
    "spotify": ["SPOT"],
    "shopify": ["SHOP"],
    "paypal": ["PYPL"],
    "venmo": ["PYPL"],
    "square": ["XYZ", "SQ"],
    "cashapp": ["XYZ", "SQ"],
    "robinhood": ["HOOD"],
    "coinbase": ["COIN"],
    "reddit": ["RDDT"],
    "pinterest": ["PINS"],
    "snapchat": ["SNAP"],
    "snap": ["SNAP"],
    "zoom": ["ZM"],
    "slack": ["CRM"],
    "salesforce": ["CRM"],
    "adobe": ["ADBE"],
    "photoshop": ["ADBE"],
    "oracle": ["ORCL"],
    "ibm": ["IBM"],
    "intel": ["INTC"],
    "amd": ["AMD"],
    "qualcomm": ["QCOM"],
    "broadcom": ["AVGO"],
    "cisco": ["CSCO"],
    "dell": ["DELL"],
    "hp": ["HPQ"],
    "sony": ["SONY"],
    "playstation": ["SONY"],
    "nintendo": ["NTDOY"],
    "samsung": ["SSNLF"],
    "palantir": ["PLTR"],
    "snowflake": ["SNOW"],
    "databricks": ["SNOW"],  # private; nearest public comparable
    "crowdstrike": ["CRWD"],
    "datadog": ["DDOG"],
    "servicenow": ["NOW"],
    "workday": ["WDAY"],
    "twilio": ["TWLO"],
    "roblox": ["RBLX"],
    "unity": ["U"],
    "ea": ["EA"],
    "electronic arts": ["EA"],
    "activision": ["MSFT"],  # acquired by Microsoft
    "rockstar": ["TTWO"],
    "gta": ["TTWO"],
    "take two": ["TTWO"],

    # ── Finance ───────────────────────────────────────────────────────────
    "berkshire": ["BRK-B", "BRK-A"],
    "buffett": ["BRK-B"],
    "jpmorgan": ["JPM"],
    "jp morgan": ["JPM"],
    "chase": ["JPM"],
    "goldman": ["GS"],
    "goldman sachs": ["GS"],
    "morgan stanley": ["MS"],
    "bank of america": ["BAC"],
    "bofa": ["BAC"],
    "wells fargo": ["WFC"],
    "citi": ["C"],
    "citibank": ["C"],
    "visa": ["V"],
    "mastercard": ["MA"],
    "amex": ["AXP"],
    "american express": ["AXP"],
    "blackrock": ["BLK"],
    "schwab": ["SCHW"],

    # ── Healthcare / pharma ───────────────────────────────────────────────
    "pfizer": ["PFE"],
    "moderna": ["MRNA"],
    "johnson": ["JNJ"],
    "johnson and johnson": ["JNJ"],
    "jnj": ["JNJ"],
    "merck": ["MRK"],
    "abbvie": ["ABBV"],
    "eli lilly": ["LLY"],
    "lilly": ["LLY"],
    "ozempic": ["NVO"],
    "novo nordisk": ["NVO"],
    "unitedhealth": ["UNH"],
    "cvs": ["CVS"],

    # ── Energy / industrial / auto ────────────────────────────────────────
    "exxon": ["XOM"],
    "chevron": ["CVX"],
    "shell": ["SHEL"],
    "bp": ["BP"],
    "boeing": ["BA"],
    "lockheed": ["LMT"],
    "caterpillar": ["CAT"],
    "ge": ["GE"],
    "general electric": ["GE"],
    "ford": ["F"],
    "gm": ["GM"],
    "general motors": ["GM"],
    "rivian": ["RIVN"],
    "lucid": ["LCID"],
    "toyota": ["TM"],
    "honda": ["HMC"],
    "ferrari": ["RACE"],

    # ── Telecom / media ───────────────────────────────────────────────────
    "verizon": ["VZ"],
    "att": ["T"],
    "at&t": ["T"],
    "tmobile": ["TMUS"],
    "t mobile": ["TMUS"],
    "comcast": ["CMCSA"],
    "warner": ["WBD"],
    "paramount": ["PARA"],

    # ── Index funds / ETFs people ask for by nickname ─────────────────────
    "sp500": ["SPY", "VOO"],
    "s&p 500": ["SPY", "VOO"],
    "s&p": ["SPY"],
    "spx": ["SPY"],
    "nasdaq": ["QQQ"],
    "nasdaq 100": ["QQQ"],
    "dow": ["DIA"],
    "dow jones": ["DIA"],
    "russell": ["IWM"],
    "russell 2000": ["IWM"],
    "total market": ["VTI"],
    "bitcoin etf": ["IBIT", "GBTC"],
    "gold": ["GLD"],
    "oil": ["USO"],
    "vix": ["VXX"],
}


#: Legal-name suffixes stripped before matching, so "apple" scores against
#: "Apple" rather than losing points to "Inc. Common Stock".
NAME_NOISE = (
    "common stock", "class a ordinary shares", "class b ordinary shares",
    "ordinary shares", "american depositary shares", "depositary shares",
    "class a common stock", "class b common stock", "class c common stock",
    "incorporated", "corporation", "company", "holdings", "group",
    "limited", "plc", "inc", "corp", "co", "ltd", "sa", "nv", "ag",
    "the", "&", ",", ".",
)


def resolve_alias(query: str) -> list[str]:
    """Return ticker symbols for a common-name query, best first."""
    return ALIASES.get(_normalise(query), [])


def fuzzy_alias(query: str, threshold: float = 0.8) -> tuple[list[str], float]:
    """Nearest alias for a misspelled common name.

    "aple" is one edit from "apple", "gogle" from "google", "teslla" from
    "tesla". Without this, a typo falls through to whatever obscure ticker
    happens to match exactly -- "aple" hits APLE (Apple Hospitality REIT),
    which is a real symbol but almost never what anyone meant.

    Returns ``(symbols, similarity)``; empty when nothing is close enough.
    """
    import difflib

    normalised = _normalise(query)
    if len(normalised) < 3:
        return [], 0.0

    best_key: str | None = None
    best_ratio = 0.0
    for key in ALIASES:
        # Only compare against comparable-length keys; "app" should not be
        # fuzzily dragged onto "amazon prime".
        if abs(len(key) - len(normalised)) > 3:
            continue
        ratio = difflib.SequenceMatcher(None, normalised, key).ratio()
        if ratio > best_ratio:
            best_key, best_ratio = key, ratio

    if best_key is None or best_ratio < threshold:
        return [], 0.0
    return ALIASES[best_key], best_ratio


def _normalise(text: str) -> str:
    cleaned = text.lower().strip()
    for char in (".", ",", "'", '"', "-", "_"):
        cleaned = cleaned.replace(char, " " if char in "-_" else "")
    return " ".join(cleaned.split())


def alias_targets() -> set[str]:
    """Every symbol referenced by an alias -- used to prioritise ingestion."""
    return {symbol for symbols in ALIASES.values() for symbol in symbols}


def primary_symbols() -> set[str]:
    """The first symbol of every alias entry.

    Several companies have multiple share classes (GOOGL/GOOG, BRK-B/BRK-A).
    They match a query equally well and often have identical market caps, so
    ranking would otherwise fall to an arbitrary tiebreak like symbol length.
    Being first in an alias list marks the listing people normally mean.
    """
    return {symbols[0] for symbols in ALIASES.values() if symbols}
