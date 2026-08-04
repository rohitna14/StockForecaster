"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { Badge, Callout, EmptyState, Skeleton, Stat } from "@/components/ui/primitives";
import { SymbolTabs } from "@/components/SymbolTabs";
import { cn, fmtNumber, fmtPercent, fmtPrice } from "@/lib/format";
import type { BacktestResponse } from "@/lib/types";

const COST_PRESETS = [
  { id: "zero", label: "Zero", note: "Signal quality only — never a claim." },
  { id: "optimistic", label: "Optimistic", note: "Large caps, tight spreads, small size." },
  { id: "realistic", label: "Realistic", note: "Default retail assumptions." },
  { id: "conservative", label: "Conservative", note: "Stress test: does the edge survive?" },
];

export default function BacktestPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol: raw } = use(params);
  const symbol = raw.toUpperCase();

  const [preset, setPreset] = useState("realistic");
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await api.backtest({
          symbol,
          model: "lightgbm",
          horizon: 1,
          cost_preset: preset,
          sizing: "vol_target",
        }),
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Backtest failed.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [symbol, preset]);

  useEffect(() => {
    void run();
  }, [run]);

  const stats = data?.stats ?? {};
  const num = (key: string): number | null => {
    const value = stats[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };

  const beatsBenchmark =
    num("total_return") !== null &&
    num("benchmark_return") !== null &&
    num("total_return")! > num("benchmark_return")!;

  return (
    <div className="space-y-6">
      <div>
        <span className="label">Backtest</span>
        <h1 className="mt-1 text-3xl font-black tracking-tight">
          <span className="tnum font-mono grad-text">{symbol}</span> strategy test
        </h1>
        <p className="mt-2 text-ink-muted">
          Out-of-sample predictions only. Next-bar execution, costs charged on turnover.
        </p>
      </div>

      <SymbolTabs symbol={symbol} active="backtest" />

      <div className="glass p-5">
        <div className="label mb-2">Transaction costs</div>
        <div className="flex flex-wrap gap-2">
          {COST_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setPreset(p.id)}
              className={cn("btn-chip", preset === p.id && "btn-chip-active")}
            >
              {p.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-ink-faint">
          {COST_PRESETS.find((p) => p.id === preset)?.note}
        </p>
      </div>

      {loading && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      )}

      {error && !loading && (
        <EmptyState
          title="Backtest failed"
          description={error}
          action={
            <button
              type="button"
              onClick={() => void run()}
              className="btn-ghost"
            >
              Retry
            </button>
          }
        />
      )}

      {data && !loading && (
        <>
          <Callout tone={beatsBenchmark ? "gain" : "loss"}>
            <span className="font-medium text-ink">
              Strategy {fmtPercent(num("total_return"))} vs buy-and-hold{" "}
              {fmtPercent(num("benchmark_return"))}.
            </span>{" "}
            {beatsBenchmark
              ? "The strategy outperformed simply holding the stock over this window."
              : "The strategy underperformed simply holding the stock. Directional models on daily returns generally do."}
          </Callout>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Strategy return"
              value={fmtPercent(num("total_return"))}
              emphasis
              hint="Total compounded return after transaction costs."
            />
            <Stat
              label="Buy & hold"
              value={fmtPercent(num("benchmark_return"))}
              emphasis
              hint="Holding the stock over the identical window. A strategy is only interesting relative to this."
            />
            <Stat
              label="Sharpe"
              value={fmtNumber(num("sharpe"))}
              hint="Return per unit of risk, after costs. Compare against the benchmark's own Sharpe."
            />
            <Stat
              label="Benchmark Sharpe"
              value={fmtNumber(num("benchmark_sharpe"))}
              hint="Buy-and-hold's risk-adjusted return."
            />
            <Stat
              label="Max drawdown"
              value={fmtPercent(num("max_drawdown"), 1)}
              hint="Worst peak-to-trough loss. Decides whether a strategy is actually holdable."
            />
            <Stat
              label="Trades"
              value={String(num("n_trades") ?? "—")}
              hint="Round trips. More trades means more cost drag."
            />
            <Stat
              label="Costs paid"
              value={fmtPercent(num("total_costs"), 2, { signed: false })}
              hint="Total transaction costs as a fraction of starting capital."
            />
            <Stat
              label="Cost drag on Sharpe"
              value={fmtNumber(num("cost_drag_on_sharpe"))}
              hint="How much Sharpe the costs consumed. Gross Sharpe minus net Sharpe."
            />
          </div>

          {data.cost_sensitivity.length > 0 && (
            <section className="glass p-5">
              <h2 className="text-sm font-semibold">Cost sensitivity</h2>
              <p className="mt-1 text-xs text-ink-muted">
                The question that decides whether an edge is real: at what cost
                level does it disappear?
              </p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead>
                    <tr className="border-b border-line text-left">
                      {["Preset", "Round trip", "Return", "Sharpe", "Max DD", "Trades"].map(
                        (h) => (
                          <th
                            key={h}
                            className="pb-2 text-2xs font-medium uppercase tracking-wider text-ink-faint"
                          >
                            {h}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {data.cost_sensitivity.map((row, i) => (
                      <tr key={i} className="border-b border-line/60">
                        <td className="py-2 text-ink-muted">{String(row.preset)}</td>
                        <td className="tnum py-2 font-mono text-xs text-ink-muted">
                          {fmtNumber(row.round_trip_bps as number, 1)} bps
                        </td>
                        <td
                          className={cn(
                            "tnum py-2 font-mono text-xs",
                            (row.total_return as number) > 0 ? "text-gain" : "text-loss",
                          )}
                        >
                          {fmtPercent(row.total_return as number)}
                        </td>
                        <td className="tnum py-2 font-mono text-xs text-ink-muted">
                          {fmtNumber(row.sharpe as number)}
                        </td>
                        <td className="tnum py-2 font-mono text-xs text-ink-muted">
                          {fmtPercent(row.max_drawdown as number, 1)}
                        </td>
                        <td className="tnum py-2 font-mono text-xs text-ink-muted">
                          {String(row.n_trades)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <div className="flex flex-wrap gap-2 text-xs text-ink-faint">
            <Badge tone="neutral">
              next-bar fills: {String(stats.next_bar_open_fills ?? "—")}
            </Badge>
            <Badge tone="neutral">
              time in market: {fmtPercent(num("time_in_market"), 0, { signed: false })}
            </Badge>
            <Badge tone="neutral">
              {String(stats.start_date)} → {String(stats.end_date)}
            </Badge>
            <Badge tone="neutral">
              initial {fmtPrice(100000)}
            </Badge>
          </div>

          <Callout title="What this backtest does and does not assume">
            A signal computed from bar <em>t</em>&apos;s close fills at bar{" "}
            <em>t+1</em>&apos;s open — never the same bar that generated it. Costs
            are charged when the position changes, not every bar. The benchmark is
            buy-and-hold over the identical window. What it cannot model: your
            actual fills, market impact at size, or borrow availability.
          </Callout>
        </>
      )}
    </div>
  );
}
