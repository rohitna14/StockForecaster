"""Event-driven backtester.

Design rules, each of which exists because violating it is a standard way to
manufacture a fake edge:

**1. Next-bar execution.** A signal computed from bar *t*'s close is filled at
bar *t+1*'s open, never at bar *t*'s close. Filling on the same bar that
generated the signal is the single most common backtest error -- it assumes you
could trade at a price you only knew after the close.

**2. Positions shift forward.** The return earned on bar *t+1* is the position
established *at* bar *t+1*'s open, based on information through bar *t*.

**3. Costs on turnover.** Charged when the position changes, scaled by
volatility. See :mod:`forecaster.backtest.costs`.

**4. Buy-and-hold is always computed.** Over the identical window, with the same
starting capital. A strategy that returns 40% while the underlying returned 80%
has destroyed value, and reporting only the 40% would hide that.

**5. Volatility targeting.** Raw sign-of-forecast sizing loads up on exactly the
most dangerous names. Sizing inversely to trailing volatility is what makes a
Sharpe ratio comparable across tickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from forecaster.backtest.costs import COST_PRESETS, CostModel, CostReport
from forecaster.logging import get_logger
from forecaster.validation import metrics as M

log = get_logger(__name__)

TRADING_DAYS = 252


class SizingMode(StrEnum):
    #: Full long/short on the sign of the forecast.
    SIGN = "sign"
    #: Position proportional to forecast magnitude, capped.
    PROPORTIONAL = "proportional"
    #: Sign, scaled to hit a target annualised volatility.
    VOL_TARGET = "vol_target"
    #: Long-only: sign, floored at zero.
    LONG_ONLY = "long_only"


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    sizing: SizingMode = SizingMode.VOL_TARGET
    #: Annualised volatility target for VOL_TARGET sizing.
    target_volatility: float = 0.15
    #: Maximum absolute position as a multiple of equity.
    max_leverage: float = 1.0
    #: Forecast magnitude below which no position is taken. Filters the
    #: near-zero predictions that generate churn and pure cost.
    signal_threshold: float = 0.0
    costs: CostModel = field(default_factory=lambda: COST_PRESETS["realistic"])
    #: Trailing window for the volatility estimate used in sizing.
    vol_window: int = 20
    periods_per_year: int = TRADING_DAYS

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "sizing": str(self.sizing),
            "target_volatility": self.target_volatility,
            "max_leverage": self.max_leverage,
            "signal_threshold": self.signal_threshold,
            "vol_window": self.vol_window,
            "costs": self.costs.as_dict(),
        }


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    positions: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    stats: dict[str, Any]
    cost_report: CostReport
    config: BacktestConfig

    def summary(self) -> str:
        s = self.stats
        return (
            f"Return {s['total_return']:+.2%} vs buy-and-hold "
            f"{s['benchmark_return']:+.2%} | Sharpe {s['sharpe']:.2f} "
            f"(bench {s['benchmark_sharpe']:.2f}) | MaxDD {s['max_drawdown']:.2%} | "
            f"{s['n_trades']} trades | costs {s['total_costs']:.2%} of capital"
        )


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: pd.Series,
        predictions: pd.Series,
        *,
        opens: pd.Series | None = None,
    ) -> BacktestResult:
        """Backtest a prediction series against a price series.

        Args:
            prices: close prices, date-indexed.
            predictions: forecast of the *next* bar's return, indexed by the
                date the forecast was made (its ``as_of_date``).
            opens: open prices for next-bar fills. Falls back to closes when
                absent, which is slightly optimistic and is flagged in stats.

        The two series are aligned on their intersection, so a prediction
        without a matching price is dropped rather than silently misaligned.
        """
        cfg = self.config

        frame = pd.DataFrame({"close": prices}).join(
            predictions.rename("prediction"), how="inner"
        )
        if opens is not None:
            frame = frame.join(opens.rename("open"), how="left")
        else:
            frame["open"] = frame["close"]
        frame = frame.dropna(subset=["close", "prediction"]).sort_index()

        if len(frame) < 20:
            raise ValueError(f"Not enough aligned bars to backtest: {len(frame)}")

        frame["open"] = frame["open"].fillna(frame["close"])

        # ── realised returns ──────────────────────────────────────────────
        # Bar t's return is close_t / close_{t-1} - 1.
        bar_returns = frame["close"].pct_change().fillna(0.0)

        # ── position sizing from the forecast ─────────────────────────────
        raw_positions = self._size_positions(frame["prediction"], frame["close"])

        # ── next-bar execution ────────────────────────────────────────────
        # The forecast made at bar t is acted on from bar t+1 onward. Shifting
        # by one is what enforces that; without it, bar t's position would earn
        # bar t's return, which was already known when the signal was computed.
        positions = raw_positions.shift(1).fillna(0.0)

        gross_returns = positions * bar_returns

        trailing_vol = (
            bar_returns.rolling(cfg.vol_window, min_periods=5).std(ddof=1)
            * np.sqrt(cfg.periods_per_year)
        ).fillna(0.0)
        cost_series = pd.Series(
            cfg.costs.apply(
                positions.to_numpy(),
                volatility=trailing_vol.to_numpy(),
                periods_per_year=cfg.periods_per_year,
            ),
            index=frame.index,
        )

        net_returns = gross_returns - cost_series

        equity = cfg.initial_capital * (1.0 + net_returns).cumprod()
        benchmark_equity = cfg.initial_capital * (1.0 + bar_returns).cumprod()

        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1.0

        trades = self._extract_trades(frame, positions, equity)
        cost_report = self._cost_report(positions, cost_series, net_returns)
        stats = self._compute_stats(
            net_returns, gross_returns, bar_returns, equity, benchmark_equity,
            drawdown, cost_series, positions, trades, used_opens=opens is not None,
        )

        log.info("backtest_complete", **{k: v for k, v in stats.items()
                                         if isinstance(v, (int, float))})

        return BacktestResult(
            equity=equity,
            benchmark_equity=benchmark_equity,
            positions=positions,
            returns=net_returns,
            gross_returns=gross_returns,
            costs=cost_series,
            drawdown=drawdown,
            trades=trades,
            stats=stats,
            cost_report=cost_report,
            config=cfg,
        )

    # ── sizing ────────────────────────────────────────────────────────────
    def _size_positions(self, predictions: pd.Series, close: pd.Series) -> pd.Series:
        cfg = self.config
        pred = predictions.fillna(0.0)

        # Dead-band: ignore forecasts too small to be worth the spread.
        if cfg.signal_threshold > 0:
            pred = pred.where(pred.abs() >= cfg.signal_threshold, 0.0)

        if cfg.sizing is SizingMode.SIGN:
            positions = np.sign(pred)
        elif cfg.sizing is SizingMode.LONG_ONLY:
            positions = np.clip(np.sign(pred), 0.0, 1.0)
        elif cfg.sizing is SizingMode.PROPORTIONAL:
            scale = pred.abs().rolling(60, min_periods=20).quantile(0.9)
            positions = (pred / scale.replace(0.0, np.nan)).clip(-1.0, 1.0).fillna(0.0)
        else:  # VOL_TARGET
            returns = close.pct_change()
            trailing_vol = returns.rolling(cfg.vol_window, min_periods=5).std(ddof=1) * np.sqrt(
                cfg.periods_per_year
            )
            # Inverse-volatility scaling to hit the target. Clipped so a calm
            # period cannot produce absurd leverage.
            scale = (cfg.target_volatility / trailing_vol.replace(0.0, np.nan)).clip(0.0, 3.0)
            positions = np.sign(pred) * scale.fillna(0.0)

        return pd.Series(positions, index=predictions.index).clip(
            -cfg.max_leverage, cfg.max_leverage
        ).fillna(0.0)

    # ── reporting ─────────────────────────────────────────────────────────
    def _extract_trades(
        self, frame: pd.DataFrame, positions: pd.Series, equity: pd.Series
    ) -> pd.DataFrame:
        """Collapse the position path into discrete round-trip trades."""
        rows: list[dict[str, Any]] = []
        current_side = 0.0
        entry_idx: Any = None
        entry_price = 0.0
        entry_equity = 0.0

        for ts, position in positions.items():
            side = float(np.sign(position))
            if side == current_side:
                continue

            if current_side != 0.0 and entry_idx is not None:
                exit_price = float(frame.loc[ts, "open"])
                pnl_pct = (exit_price / entry_price - 1.0) * current_side
                rows.append(
                    {
                        "entry_date": entry_idx,
                        "exit_date": ts,
                        "side": "long" if current_side > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "pnl": float(equity.loc[ts] - entry_equity),
                        "bars_held": int(
                            positions.index.get_loc(ts) - positions.index.get_loc(entry_idx)
                        ),
                    }
                )

            if side != 0.0:
                entry_idx = ts
                entry_price = float(frame.loc[ts, "open"])
                entry_equity = float(equity.loc[ts])
            current_side = side

        return pd.DataFrame(
            rows,
            columns=["entry_date", "exit_date", "side", "entry_price", "exit_price",
                     "pnl_pct", "pnl", "bars_held"],
        )

    def _cost_report(
        self, positions: pd.Series, costs: pd.Series, net_returns: pd.Series
    ) -> CostReport:
        turnover = positions.diff().abs().fillna(positions.abs())
        n_bars = max(1, len(net_returns))
        years = n_bars / self.config.periods_per_year
        return CostReport(
            total_costs=float(costs.sum()),
            total_turnover=float(turnover.sum()),
            n_trades=int((turnover > 1e-9).sum()),
            cost_drag_annual=float(costs.sum() / years) if years > 0 else 0.0,
            breakdown={"mean_cost_per_bar_bps": float(costs.mean() * 10_000)},
        )

    def _compute_stats(
        self,
        net_returns: pd.Series,
        gross_returns: pd.Series,
        bar_returns: pd.Series,
        equity: pd.Series,
        benchmark_equity: pd.Series,
        drawdown: pd.Series,
        costs: pd.Series,
        positions: pd.Series,
        trades: pd.DataFrame,
        *,
        used_opens: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        net = net_returns.to_numpy()
        bench = bar_returns.to_numpy()
        years = max(len(net) / cfg.periods_per_year, 1e-9)

        total_return = float(equity.iloc[-1] / cfg.initial_capital - 1.0)
        benchmark_return = float(benchmark_equity.iloc[-1] / cfg.initial_capital - 1.0)

        # Beta and alpha against buy-and-hold.
        beta = float("nan")
        if np.std(bench) > 1e-12:
            beta = float(np.cov(net, bench)[0, 1] / np.var(bench))
        alpha = float(np.mean(net) - beta * np.mean(bench)) * cfg.periods_per_year if np.isfinite(beta) else float("nan")

        wins = trades["pnl_pct"] > 0 if not trades.empty else pd.Series(dtype=bool)

        return {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return,
            "cagr": float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else float("nan"),
            "sharpe": M.sharpe_ratio(net, cfg.periods_per_year),
            "benchmark_sharpe": M.sharpe_ratio(bench, cfg.periods_per_year),
            "sortino": M.sortino_ratio(net, cfg.periods_per_year),
            "calmar": M.calmar_ratio(net, cfg.periods_per_year),
            "max_drawdown": float(drawdown.min()),
            "volatility": float(np.std(net, ddof=1) * np.sqrt(cfg.periods_per_year)) if len(net) > 1 else float("nan"),
            "profit_factor": M.profit_factor(net),
            "hit_rate": M.hit_rate(net),
            "win_rate": float(wins.mean()) if len(wins) else float("nan"),
            "n_trades": int(len(trades)),
            "avg_bars_held": float(trades["bars_held"].mean()) if not trades.empty else float("nan"),
            "turnover": float(positions.diff().abs().sum()),
            "total_costs": float(costs.sum()),
            "gross_sharpe": M.sharpe_ratio(gross_returns.to_numpy(), cfg.periods_per_year),
            "cost_drag_on_sharpe": float(
                M.sharpe_ratio(gross_returns.to_numpy(), cfg.periods_per_year)
                - M.sharpe_ratio(net, cfg.periods_per_year)
            ),
            "alpha": alpha,
            "beta": beta,
            "time_in_market": float((positions != 0).mean()),
            "start_date": str(equity.index[0].date()),
            "end_date": str(equity.index[-1].date()),
            "n_bars": int(len(net)),
            "next_bar_open_fills": used_opens,
        }


def sweep_costs(
    prices: pd.Series, predictions: pd.Series, opens: pd.Series | None = None
) -> pd.DataFrame:
    """Run the same strategy under every cost preset.

    The output answers the question that decides whether a signal is real:
    *at what cost level does this edge disappear?* An edge that survives only at
    zero cost is not an edge.
    """
    rows = []
    for preset_name, cost_model in COST_PRESETS.items():
        config = BacktestConfig(costs=cost_model)
        try:
            result = Backtester(config).run(prices, predictions, opens=opens)
        except Exception as exc:  # noqa: BLE001
            log.warning("cost_sweep_failed", preset=preset_name, error=str(exc))
            continue
        rows.append(
            {
                "preset": preset_name,
                "round_trip_bps": 2 * cost_model.trade_cost_bps(),
                "total_return": result.stats["total_return"],
                "sharpe": result.stats["sharpe"],
                "max_drawdown": result.stats["max_drawdown"],
                "n_trades": result.stats["n_trades"],
                "benchmark_return": result.stats["benchmark_return"],
                "benchmark_sharpe": result.stats["benchmark_sharpe"],
            }
        )
    return pd.DataFrame(rows)
