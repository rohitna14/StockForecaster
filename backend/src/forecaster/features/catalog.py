"""Feature catalog -- wires the indicator library into the registry.

Importing this module is what populates the registry, so it is imported by
``forecaster.features.__init__``.

**Design rule: every registered feature is scale-free.** No raw price levels,
no raw volume, no cumulative running totals. A model fed ``close`` learns that
NVDA trades near $200 and AAPL near $300, which is memorisation, not a pattern,
and it collapses the moment a fold crosses a split or a new ticker appears.
Everything here is a ratio, a z-score, a bounded oscillator or a return.
"""

from __future__ import annotations

import pandas as pd

from forecaster.features.indicators import momentum as mom
from forecaster.features.indicators import statistical as stat
from forecaster.features.indicators import trend
from forecaster.features.indicators import volatility as vol
from forecaster.features.indicators import volume as volm
from forecaster.features.registry import feature

# ═══════════════════════════════════ returns ═══════════════════════════════
for _p in (1, 2, 3, 5, 10, 20, 60):

    def _make_logret(period: int = _p):
        @feature(
            f"log_return_{period}",
            group="returns",
            min_history=period + 1,
            description=f"{period}-bar log return",
        )
        def _builder(df: pd.DataFrame, period: int = period) -> pd.Series:
            return stat.log_return(df["close"], period)

        return _builder

    _make_logret()


@feature("gap_open", group="returns", min_history=2, description="Overnight gap vs prior close")
def _gap_open(df: pd.DataFrame) -> pd.Series:
    return stat.gap_open(df["open"], df["close"])


@feature("intraday_range", group="returns", min_history=1, description="High-low range / close")
def _intraday_range(df: pd.DataFrame) -> pd.Series:
    return stat.intraday_range(df["high"], df["low"], df["close"])


@feature("close_location", group="returns", min_history=1, description="Close position within bar")
def _close_location(df: pd.DataFrame) -> pd.Series:
    return stat.close_location(df["high"], df["low"], df["close"])


# ═══════════════════════════════════ momentum ══════════════════════════════
for _p in (7, 14, 21):

    def _make_rsi(period: int = _p):
        @feature(
            f"rsi_{period}",
            group="momentum",
            min_history=period * 3,  # Wilder EMA needs warm-up beyond min_periods
            description=f"Wilder RSI({period})",
        )
        def _builder(df: pd.DataFrame, period: int = period) -> pd.Series:
            return mom.rsi(df["close"], period)

        return _builder

    _make_rsi()


for _p in (5, 10, 20, 60):

    def _make_mom(period: int = _p):
        @feature(
            f"momentum_{period}",
            group="momentum",
            min_history=period + 1,
            description=f"{period}-bar price momentum",
        )
        def _builder(df: pd.DataFrame, period: int = period) -> pd.Series:
            return mom.momentum(df["close"], period)

        return _builder

    _make_mom()


@feature("stochastic", group="momentum", min_history=20, description="Stochastic %K and %D")
def _stochastic(df: pd.DataFrame) -> pd.DataFrame:
    return mom.stochastic(df["high"], df["low"], df["close"])


@feature("williams_r", group="momentum", min_history=15, description="Williams %R(14)")
def _williams(df: pd.DataFrame) -> pd.Series:
    return mom.williams_r(df["high"], df["low"], df["close"])


@feature("cci", group="momentum", min_history=25, description="Commodity Channel Index(20)")
def _cci(df: pd.DataFrame) -> pd.Series:
    return mom.cci(df["high"], df["low"], df["close"])


@feature("tsi", group="momentum", min_history=60, description="True Strength Index")
def _tsi(df: pd.DataFrame) -> pd.Series:
    return mom.tsi(df["close"])


@feature(
    "awesome_oscillator",
    group="momentum",
    min_history=40,
    optional=True,
    description="Awesome Oscillator (SMA5-SMA34 of median price)",
)
def _ao(df: pd.DataFrame) -> pd.Series:
    median = (df["high"] + df["low"]) / 2.0
    return (mom.awesome_oscillator(df["high"], df["low"]) / median).rename("awesome_osc_norm")


# ═══════════════════════════════════ trend ════════════════════════════════
for _p in (10, 20, 50, 200):

    def _make_pma(period: int = _p):
        @feature(
            f"price_to_sma_{period}",
            group="trend",
            min_history=period + 1,
            description=f"Close relative to SMA({period})",
        )
        def _builder(df: pd.DataFrame, period: int = period) -> pd.Series:
            return trend.price_to_ma(df["close"], period)

        return _builder

    _make_pma()


@feature("ma_cross_20_50", group="trend", min_history=51, description="SMA20 vs SMA50 gap")
def _cross_20_50(df: pd.DataFrame) -> pd.Series:
    return trend.ma_crossover(df["close"], 20, 50)


@feature("ma_cross_50_200", group="trend", min_history=201, description="Golden/death cross gap")
def _cross_50_200(df: pd.DataFrame) -> pd.Series:
    return trend.ma_crossover(df["close"], 50, 200)


@feature("macd", group="trend", min_history=60, description="MACD line, signal, histogram")
def _macd(df: pd.DataFrame) -> pd.DataFrame:
    # Normalised by price so the feature is comparable across tickers.
    raw = trend.macd(df["close"])
    return raw.div(df["close"], axis=0).add_suffix("_norm")


@feature("adx", group="trend", min_history=45, description="ADX(14) with +DI/-DI")
def _adx(df: pd.DataFrame) -> pd.DataFrame:
    return trend.adx(df["high"], df["low"], df["close"])


@feature("aroon", group="trend", min_history=30, description="Aroon up/down/oscillator")
def _aroon(df: pd.DataFrame) -> pd.DataFrame:
    return trend.aroon(df["high"], df["low"])


@feature(
    "ichimoku",
    group="trend",
    min_history=30,
    optional=True,
    description="Ichimoku conversion/base lines (no forward-shifted cloud)",
)
def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    raw = trend.ichimoku(df["high"], df["low"])
    return raw.div(df["close"], axis=0).add_suffix("_norm")


# ═══════════════════════════════════ volatility ═══════════════════════════
for _p in (5, 10, 20, 60):

    def _make_rv(period: int = _p):
        @feature(
            f"realized_vol_{period}",
            group="volatility",
            min_history=period + 2,
            description=f"Annualised realised volatility ({period}d)",
        )
        def _builder(df: pd.DataFrame, period: int = period) -> pd.Series:
            return vol.realized_volatility(df["close"], period)

        return _builder

    _make_rv()


@feature("atr_pct", group="volatility", min_history=45, description="ATR(14) as fraction of price")
def _atr_pct(df: pd.DataFrame) -> pd.Series:
    return vol.atr_pct(df["high"], df["low"], df["close"])


@feature("bollinger", group="volatility", min_history=25, description="Bollinger %B and bandwidth")
def _bollinger(df: pd.DataFrame) -> pd.DataFrame:
    # Only the scale-free columns; the raw band levels are for charts.
    return vol.bollinger(df["close"])[["bb_pct", "bb_width"]]


@feature("keltner", group="volatility", min_history=45, description="Keltner channel position")
def _keltner(df: pd.DataFrame) -> pd.DataFrame:
    return vol.keltner(df["high"], df["low"], df["close"])[["keltner_pct"]]


@feature("donchian", group="volatility", min_history=25, description="Donchian position and width")
def _donchian(df: pd.DataFrame) -> pd.DataFrame:
    return vol.donchian(df["high"], df["low"], df["close"])


@feature(
    "parkinson_vol",
    group="volatility",
    min_history=25,
    description="Parkinson high-low volatility estimator",
)
def _parkinson(df: pd.DataFrame) -> pd.Series:
    return vol.parkinson_volatility(df["high"], df["low"])


@feature(
    "garman_klass_vol",
    group="volatility",
    min_history=25,
    optional=True,
    description="Garman-Klass OHLC volatility estimator",
)
def _gk(df: pd.DataFrame) -> pd.Series:
    return vol.garman_klass_volatility(df["open"], df["high"], df["low"], df["close"])


@feature(
    "vol_ratio_5_20",
    group="volatility",
    min_history=25,
    description="Short/long volatility ratio (regime signal)",
)
def _vol_ratio(df: pd.DataFrame) -> pd.Series:
    return vol.volatility_ratio(df["close"], 5, 20)


@feature(
    "ulcer_index",
    group="volatility",
    min_history=20,
    optional=True,
    description="Ulcer index (RMS drawdown)",
)
def _ulcer(df: pd.DataFrame) -> pd.Series:
    return vol.ulcer_index(df["close"])


# ═══════════════════════════════════ volume ═══════════════════════════════
@feature("volume_ratio_20", group="volume", min_history=25, description="Volume vs 20d average")
def _vol_ratio_20(df: pd.DataFrame) -> pd.Series:
    return volm.volume_ratio(df["volume"], 20)


@feature("volume_zscore_20", group="volume", min_history=25, description="Volume z-score (20d)")
def _vol_z(df: pd.DataFrame) -> pd.Series:
    return volm.volume_zscore(df["volume"], 20)


@feature("obv_slope_20", group="volume", min_history=25, description="Normalised OBV slope")
def _obv_slope(df: pd.DataFrame) -> pd.Series:
    return volm.obv_slope(df["close"], df["volume"], 20)


@feature("mfi", group="volume", min_history=20, description="Money Flow Index(14)")
def _mfi(df: pd.DataFrame) -> pd.Series:
    return volm.mfi(df["high"], df["low"], df["close"], df["volume"])


@feature("price_to_vwap_20", group="volume", min_history=25, description="Close vs 20d VWAP")
def _p2vwap(df: pd.DataFrame) -> pd.Series:
    return volm.price_to_vwap(df["high"], df["low"], df["close"], df["volume"], 20)


@feature(
    "chaikin_osc",
    group="volume",
    min_history=30,
    optional=True,
    description="Chaikin oscillator, volume-normalised",
)
def _chaikin(df: pd.DataFrame) -> pd.Series:
    return volm.chaikin_oscillator(df["high"], df["low"], df["close"], df["volume"])


@feature(
    "log_dollar_volume_20",
    group="volume",
    min_history=25,
    description="log10 average daily dollar volume (liquidity)",
)
def _dollar_vol(df: pd.DataFrame) -> pd.Series:
    return volm.dollar_volume(df["close"], df["volume"], 20)


# ═══════════════════════════════════ statistical ══════════════════════════
@feature("skew_20", group="statistical", min_history=25, description="Return skewness (20d)")
def _skew(df: pd.DataFrame) -> pd.Series:
    return stat.rolling_skew(df["close"], 20)


@feature("kurtosis_20", group="statistical", min_history=25, description="Excess kurtosis (20d)")
def _kurt(df: pd.DataFrame) -> pd.Series:
    return stat.rolling_kurtosis(df["close"], 20)


@feature(
    "autocorr_20",
    group="statistical",
    min_history=30,
    description="Lag-1 return autocorrelation (momentum vs mean-reversion regime)",
)
def _autocorr(df: pd.DataFrame) -> pd.Series:
    return stat.autocorrelation(df["close"], 20, 1)


@feature("autocorr_60", group="statistical", min_history=70, description="Lag-1 autocorr (60d)")
def _autocorr60(df: pd.DataFrame) -> pd.Series:
    return stat.autocorrelation(df["close"], 60, 1)


@feature(
    "drawdown_252", group="statistical", min_history=2, description="Drawdown from rolling 1y high"
)
def _dd(df: pd.DataFrame) -> pd.Series:
    return stat.drawdown_from_high(df["close"], 252)


@feature(
    "days_since_high_252",
    group="statistical",
    min_history=253,
    description="Normalised bars since 1y high",
)
def _dsh(df: pd.DataFrame) -> pd.Series:
    return stat.days_since_high(df["close"], 252)


@feature(
    "downside_dev_20",
    group="statistical",
    min_history=25,
    description="Downside deviation (Sortino denominator)",
)
def _dsd(df: pd.DataFrame) -> pd.Series:
    return stat.downside_deviation(df["close"], 20)


@feature(
    "hurst_100",
    group="statistical",
    min_history=110,
    optional=True,
    description="Rolling Hurst exponent (expensive)",
)
def _hurst(df: pd.DataFrame) -> pd.Series:
    return stat.hurst_exponent(df["close"], 100)


# ═══════════════════════════════════ calendar ═════════════════════════════
@feature(
    "calendar",
    group="calendar",
    min_history=1,
    description="Day-of-week / month / turn-of-month effects",
)
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    return pd.DataFrame(
        {
            "dow": idx.dayofweek.astype("float64"),
            "month": idx.month.astype("float64"),
            "day_of_month": idx.day.astype("float64"),
            # Turn-of-month effect is one of the more durable calendar anomalies.
            "is_month_end": idx.is_month_end.astype("float64"),
            "is_month_start": idx.is_month_start.astype("float64"),
            "is_quarter_end": idx.is_quarter_end.astype("float64"),
        },
        index=df.index,
    )


# ═══════════════════════════════ named feature sets ═══════════════════════
def _names(*groups: str, include_optional: bool = False) -> list[str]:
    from forecaster.features.registry import list_features

    return [s.name for s in list_features(include_optional=include_optional) if s.group in groups]


def feature_set(name: str) -> list[str]:
    """Resolve a named feature set to a concrete list.

    ``core``   -- fast, robust, the default for the harness
    ``full``   -- everything non-optional
    ``all``    -- everything including expensive/experimental features
    ``minimal``-- returns + a couple of oscillators; a deliberately weak set
                  used to show that more features is not automatically better
    """
    from forecaster.features.registry import list_features

    if name == "minimal":
        return ["log_return_1", "log_return_5", "rsi_14", "realized_vol_20"]
    if name == "core":
        return _names("returns", "momentum", "trend", "volatility", "volume")
    if name == "full":
        return [s.name for s in list_features(include_optional=False)]
    if name == "all":
        return [s.name for s in list_features(include_optional=True)]
    raise ValueError(f"Unknown feature set {name!r}; try minimal|core|full|all")
