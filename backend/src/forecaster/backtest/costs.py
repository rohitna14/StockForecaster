"""Transaction cost model.

A backtest without costs is a fantasy. Short-horizon signals trade often, and
turnover is exactly where paper edges die: a strategy rebalancing daily at 5bps
round-trip burns ~12% a year before it has predicted anything.

Three components, all charged on *traded notional*:

* **Commission** -- broker fee. Effectively zero at US retail brokers now, but
  parameterised because it is not zero everywhere.
* **Spread** -- half the bid-ask spread, paid on entry and on exit.
* **Slippage** -- market impact, scaled by volatility. Trading a 60%-vol name
  costs materially more than a 12%-vol one, so a flat assumption flatters
  exactly the volatile names a momentum signal tends to pick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Per-unit-notional trading costs, all expressed in basis points."""

    commission_bps: float = 0.0
    #: Half-spread paid per side. 2bps is realistic for large-cap US equities;
    #: small caps are far worse.
    half_spread_bps: float = 2.0
    #: Fixed slippage floor.
    slippage_bps: float = 1.0
    #: Additional slippage proportional to annualised volatility.
    #: 0.05 charges 1bp extra per 20 vol points.
    slippage_vol_coefficient: float = 0.05
    #: Annual borrow cost for short positions, charged pro rata.
    short_borrow_bps_annual: float = 30.0

    def trade_cost_bps(self, volatility: float | np.ndarray | None = None) -> Any:
        """Cost in bps of trading one unit of notional (one side)."""
        base = self.commission_bps + self.half_spread_bps + self.slippage_bps
        if volatility is None:
            return base
        vol = np.nan_to_num(np.asarray(volatility, dtype="float64"), nan=0.0)
        return base + self.slippage_vol_coefficient * vol * 100.0

    def apply(
        self,
        positions: np.ndarray,
        *,
        volatility: np.ndarray | None = None,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Per-bar cost as a fraction of equity.

        Costs are charged on the **change** in position, not on the position
        itself. Holding a view for ten bars costs one trade, not ten -- charging
        per bar is the most common way a backtest understates a stable model and
        overstates a jumpy one.
        """
        positions = np.asarray(positions, dtype="float64")
        previous = np.concatenate([[0.0], positions[:-1]])
        turnover = np.abs(positions - previous)

        cost_bps = self.trade_cost_bps(volatility)
        trade_costs = turnover * np.asarray(cost_bps, dtype="float64") / 10_000.0

        # Borrow is charged on short exposure for every bar it is held.
        short_exposure = np.clip(-positions, 0.0, None)
        borrow = short_exposure * (self.short_borrow_bps_annual / 10_000.0) / periods_per_year

        return trade_costs + borrow

    def as_dict(self) -> dict[str, Any]:
        return {
            "commission_bps": self.commission_bps,
            "half_spread_bps": self.half_spread_bps,
            "slippage_bps": self.slippage_bps,
            "slippage_vol_coefficient": self.slippage_vol_coefficient,
            "short_borrow_bps_annual": self.short_borrow_bps_annual,
            "round_trip_bps_estimate": 2 * self.trade_cost_bps(),
        }


#: Presets, so the UI can offer a cost-sensitivity slider with defensible stops.
COST_PRESETS: dict[str, CostModel] = {
    "zero": CostModel(0.0, 0.0, 0.0, 0.0, 0.0),
    "optimistic": CostModel(0.0, 1.0, 0.5, 0.02, 20.0),
    "realistic": CostModel(0.0, 2.0, 1.0, 0.05, 30.0),
    "conservative": CostModel(1.0, 5.0, 3.0, 0.10, 50.0),
}


@dataclass
class CostReport:
    total_costs: float = 0.0
    total_turnover: float = 0.0
    n_trades: int = 0
    cost_drag_annual: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_costs": self.total_costs,
            "total_turnover": self.total_turnover,
            "n_trades": self.n_trades,
            "cost_drag_annual": self.cost_drag_annual,
            **self.breakdown,
        }
