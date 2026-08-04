"""Data quality gates.

Bad bars are worse than missing bars: a single mis-scaled close creates a fake
50% return that a momentum feature will happily learn from. Everything ingested
passes through here first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecaster.logging import get_logger

log = get_logger(__name__)

#: A single-day move larger than this is treated as a suspected unadjusted
#: split rather than a real return. 2:1 splits produce exactly -50%.
EXTREME_RETURN_THRESHOLD = 0.50

#: Gap (in calendar days) above which we flag missing history. Long weekends
#: plus holidays reach 4-5 days legitimately.
MAX_EXPECTED_GAP_DAYS = 7


@dataclass
class ValidationReport:
    symbol: str
    n_rows: int = 0
    n_dropped: int = 0
    ohlc_violations: int = 0
    zero_volume_days: int = 0
    extreme_moves: list[tuple[str, float]] = field(default_factory=list)
    gaps: list[tuple[str, str, int]] = field(default_factory=list)
    duplicate_dates: int = 0

    @property
    def is_clean(self) -> bool:
        return self.ohlc_violations == 0 and self.duplicate_dates == 0

    def summary(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "rows": self.n_rows,
            "dropped": self.n_dropped,
            "ohlc_violations": self.ohlc_violations,
            "zero_volume_days": self.zero_volume_days,
            "extreme_moves": len(self.extreme_moves),
            "gaps": len(self.gaps),
        }


def validate_bars(
    frame: pd.DataFrame, symbol: str, *, drop_invalid: bool = True
) -> tuple[pd.DataFrame, ValidationReport]:
    """Check OHLC coherence and flag suspicious history.

    Returns the (optionally cleaned) frame and a report. Rows violating OHLC
    ordering are dropped rather than repaired -- guessing which of the four
    prices is wrong would be fabrication.
    """
    report = ValidationReport(symbol=symbol, n_rows=len(frame))
    if frame.empty:
        return frame, report

    out = frame.copy()

    dupes = out.index.duplicated(keep="last")
    report.duplicate_dates = int(dupes.sum())
    if report.duplicate_dates:
        out = out[~dupes]

    # OHLC coherence: high must dominate, low must be dominated.
    valid = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out["close"] > 0)
        & (out["volume"] >= 0)
    )
    report.ohlc_violations = int((~valid).sum())
    if report.ohlc_violations:
        log.warning("ohlc_violations", symbol=symbol, count=report.ohlc_violations)
        if drop_invalid:
            out = out[valid]

    report.zero_volume_days = int((out["volume"] == 0).sum())

    # Suspected unadjusted corporate actions.
    if len(out) > 1:
        returns = out["adj_close"].pct_change()
        extreme = returns[returns.abs() > EXTREME_RETURN_THRESHOLD]
        report.extreme_moves = [
            (ts.strftime("%Y-%m-%d"), float(r)) for ts, r in extreme.items() if np.isfinite(r)
        ]
        if report.extreme_moves:
            log.warning(
                "extreme_moves_detected",
                symbol=symbol,
                count=len(report.extreme_moves),
                sample=report.extreme_moves[:3],
                hint="possible unadjusted split",
            )

    # Missing-history gaps.
    if len(out) > 1:
        deltas = out.index.to_series().diff().dt.days
        for ts, gap in deltas[deltas > MAX_EXPECTED_GAP_DAYS].items():
            pos = out.index.get_loc(ts)
            prev = out.index[pos - 1] if isinstance(pos, int) and pos > 0 else ts
            report.gaps.append((prev.strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%d"), int(gap)))

    report.n_dropped = report.n_rows - len(out)
    return out, report


def cross_check_providers(
    primary: pd.DataFrame, secondary: pd.DataFrame, *, tolerance: float = 0.005
) -> pd.DataFrame:
    """Compare overlapping closes between two providers.

    Returns a frame of disagreements beyond ``tolerance`` (default 0.5%).
    An empty result means the sources corroborate each other; a large result
    usually means one of them is serving unadjusted prices.
    """
    overlap = primary.index.intersection(secondary.index)
    if overlap.empty:
        return pd.DataFrame(columns=["primary", "secondary", "rel_diff"])

    a = primary.loc[overlap, "close"]
    b = secondary.loc[overlap, "close"]
    rel = ((a - b).abs() / b.replace(0, np.nan)).dropna()
    bad = rel[rel > tolerance]
    return pd.DataFrame(
        {"primary": a.loc[bad.index], "secondary": b.loc[bad.index], "rel_diff": bad}
    )
