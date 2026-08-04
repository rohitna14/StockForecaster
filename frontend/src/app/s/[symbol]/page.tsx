import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import {
  fmtCompact,
  fmtDate,
  fmtNumber,
  fmtPercent,
  fmtPrice,
  sectorColor,
  trendGlyph,
} from "@/lib/format";
import { PriceChart } from "@/components/charts/PriceChart";
import { SymbolTabs } from "@/components/SymbolTabs";
import { StatRing } from "@/components/ui/StatRing";
import { SearchBar } from "@/components/SearchBar";
import type {
  InstrumentSummary,
  InstrumentSummaryStats,
  OHLCVResponse,
  RiskResponse,
} from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function SymbolPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol: raw } = await params;
  const symbol = raw.toUpperCase();

  let summary: InstrumentSummaryStats;
  let ohlcv: OHLCVResponse;
  let risk: RiskResponse | null = null;
  let info: InstrumentSummary | null = null;

  try {
    [summary, ohlcv] = await Promise.all([
      api.getSummary(symbol),
      api.getOHLCV(symbol, 500),
    ]);
  } catch (error) {
    return <ErrorState symbol={symbol} error={error} />;
  }

  [risk, info] = await Promise.all([
    api.getRisk(symbol).catch(() => null),
    api.getInstrument(symbol).catch(() => null),
  ]);

  const accent = sectorColor(info?.sector);
  const up = summary.change_1d >= 0;
  const rangePos =
    ((summary.last_close - summary.range_52w_low) /
      Math.max(summary.range_52w_high - summary.range_52w_low, 1e-9)) *
    100;

  return (
    <div className="space-y-7">
      {/* ═══════ HEADER ═══════ */}
      <div className="glass relative overflow-hidden p-6">
        <span
          className="absolute -right-24 -top-24 h-64 w-64 rounded-full opacity-20 blur-3xl"
          style={{ background: accent }}
        />
        <span className="absolute inset-x-0 top-0 h-[2px]" style={{ background: accent }} />

        <div className="relative flex flex-wrap items-start justify-between gap-6">
          <div className="flex items-start gap-4">
            <span
              className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl text-lg font-black text-white shadow-lg"
              style={{ background: accent }}
            >
              {symbol.slice(0, 2)}
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="tnum font-mono text-3xl font-black tracking-tight">{symbol}</h1>
                {info?.sector && (
                  <span
                    className="rounded-pill px-2.5 py-1 text-xs font-medium"
                    style={{ background: `${accent}22`, color: accent }}
                  >
                    {info.sector}
                  </span>
                )}
                <span className="rounded-pill border border-gain/40 bg-gain/10 px-2.5 py-1 text-xs font-medium text-gain">
                  {summary.source_tier} tier
                </span>
              </div>
              {info?.name && <p className="mt-1 text-ink-muted">{info.name}</p>}
              <p className="mt-1 text-xs text-ink-faint">
                {summary.n_bars.toLocaleString()} bars · {fmtDate(summary.first_date)} →{" "}
                {fmtDate(summary.as_of)}
                {info?.market_cap ? ` · $${fmtCompact(info.market_cap)} cap` : ""}
              </p>
            </div>
          </div>

          <div className="text-right">
            <div className="tnum font-mono text-4xl font-black tracking-tight">
              {fmtPrice(summary.last_close)}
            </div>
            <div
              className={`tnum mt-1 inline-flex items-center gap-1.5 rounded-pill px-3 py-1 font-mono text-sm font-bold ${
                up ? "bg-gain/15 text-gain" : "bg-loss/15 text-loss"
              }`}
            >
              {trendGlyph(summary.change_1d)} {fmtPercent(summary.change_1d)}
              <span className="font-normal opacity-60">today</span>
            </div>
          </div>
        </div>

        {/* 52-week range bar */}
        <div className="relative mt-6">
          <div className="mb-1.5 flex justify-between text-2xs text-ink-faint">
            <span className="tnum font-mono">{fmtPrice(summary.range_52w_low)}</span>
            <span className="label">52-week range</span>
            <span className="tnum font-mono">{fmtPrice(summary.range_52w_high)}</span>
          </div>
          <div className="relative h-2 overflow-hidden rounded-pill bg-line">
            <div
              className="absolute inset-y-0 left-0 rounded-pill bg-grad-primary opacity-40"
              style={{ width: `${Math.max(0, Math.min(100, rangePos))}%` }}
            />
            <div
              className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded-full bg-ink shadow-glow"
              style={{ left: `calc(${Math.max(0, Math.min(100, rangePos))}% - 2px)` }}
            />
          </div>
        </div>
      </div>

      <SymbolTabs symbol={symbol} active="overview" />

      {/* ═══════ PERFORMANCE ═══════ */}
      <div className="grid gap-3 stagger sm:grid-cols-2 lg:grid-cols-4">
        <PerfCard label="1 week" value={summary.change_1w} />
        <PerfCard label="1 month" value={summary.change_1m} />
        <PerfCard label="1 year" value={summary.change_1y} />
        <div className="glass glass-hover p-4">
          <div className="label">Volume</div>
          <div className="tnum mt-2 font-mono text-2xl font-bold">
            {fmtCompact(summary.volume)}
          </div>
          <div className="tnum mt-1 font-mono text-xs text-ink-faint">
            20d avg {fmtCompact(summary.avg_volume_20d)}
          </div>
        </div>
      </div>

      {/* ═══════ CHART ═══════ */}
      <section className="glass overflow-hidden p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="label">Price history</span>
            <h2 className="mt-0.5 text-lg font-bold">Two years of daily bars</h2>
          </div>
          <div className="flex gap-1.5">
            <span className="btn-chip pointer-events-none">
              <span className="h-1.5 w-1.5 rounded-full bg-gain" /> up
            </span>
            <span className="btn-chip pointer-events-none">
              <span className="h-1.5 w-1.5 rounded-full bg-loss" /> down
            </span>
          </div>
        </div>
        <PriceChart bars={ohlcv.bars} height={440} />
        <p className="mt-3 text-2xs text-ink-faint">
          Split- and dividend-adjusted. The whole OHLC bar is back-adjusted and volume
          inverted, so gap and range features stay correct across splits.
        </p>
      </section>

      {/* ═══════ RISK ═══════ */}
      {risk && (
        <section>
          <div className="mb-4">
            <span className="label">Risk profile</span>
            <h2 className="mt-0.5 text-lg font-bold">How bumpy has this been?</h2>
          </div>

          <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
            <div className="glass flex items-center gap-5 p-5">
              <StatRing
                value={Math.min((risk.annual_volatility ?? 0) / 0.8, 1)}
                label="vol"
              />
              <div>
                <div className="tnum font-mono text-2xl font-bold">
                  {fmtPercent(risk.annual_volatility, 1, { signed: false })}
                </div>
                <div className="label mt-1">Annual volatility</div>
                <p className="mt-2 max-w-[15rem] text-xs leading-relaxed text-ink-muted">
                  Typical size of yearly price swings. Higher means bigger moves in{" "}
                  <em>both</em> directions — it says nothing about which way.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <RiskCard
                label="Sharpe"
                value={fmtNumber(risk.sharpe)}
                hint="Return per unit of risk. This is the stock's own, not a strategy's."
                tone={(risk.sharpe ?? 0) > 1 ? "gain" : "neutral"}
              />
              <RiskCard
                label="Sortino"
                value={fmtNumber(risk.sortino)}
                hint="Like Sharpe but only penalises downside. Upside volatility isn't risk."
                tone={(risk.sortino ?? 0) > 1 ? "gain" : "neutral"}
              />
              <RiskCard
                label="Max drawdown"
                value={fmtPercent(risk.max_drawdown, 1)}
                hint="Worst peak-to-trough fall. Decides whether a position is holdable."
                tone="loss"
              />
              <RiskCard
                label="VaR 95%"
                value={fmtPercent(risk.var_95)}
                hint="On the worst 5% of days, losses exceeded this."
                tone="loss"
              />
            </div>
          </div>
        </section>
      )}

      {/* ═══════ CTA STRIP ═══════ */}
      <section className="grid gap-4 lg:grid-cols-2">
        <Link
          href={`/s/${symbol}/models`}
          className="ring-grad glass glass-hover group relative overflow-hidden p-6"
        >
          <span className="absolute -left-12 -top-12 h-40 w-40 rounded-full bg-violet/20 blur-3xl" />
          <div className="relative">
            <span className="label">Next step</span>
            <h3 className="mt-1 text-xl font-bold">
              Run the models on {symbol}
              <span className="ml-2 inline-block transition-transform duration-300 ease-spring group-hover:translate-x-1">
                →
              </span>
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              Walk-forward evaluation across 6 models and 7 baselines, computed live.
              Switch between volatility and direction to see the difference.
            </p>
          </div>
        </Link>

        <div className="glass relative overflow-hidden p-6">
          <span className="absolute -right-12 -bottom-12 h-40 w-40 rounded-full bg-warn/15 blur-3xl" />
          <div className="relative">
            <span className="rounded-pill border border-warn/40 bg-warn/10 px-2.5 py-1 text-xs font-semibold text-warn">
              Read this first
            </span>
            <h3 className="mt-3 text-lg font-bold">No model predicts direction</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              On our own testing, nothing beat a naive baseline at calling up or down —
              for any symbol, at any horizon. The measurable edge is in forecasting{" "}
              <strong className="text-ink">volatility</strong>. Both are shown, with the
              baseline always in the table.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

/* ── pieces ──────────────────────────────────────────────────────────── */

function PerfCard({ label, value }: { label: string; value: number | null }) {
  const up = (value ?? 0) >= 0;
  const magnitude = Math.min(Math.abs((value ?? 0) * 100) / 40, 1) * 100;

  return (
    <div className="glass glass-hover relative overflow-hidden p-4">
      <div className="label">{label}</div>
      <div
        className={`tnum mt-2 font-mono text-2xl font-bold ${
          value === null ? "text-ink-faint" : up ? "text-gain" : "text-loss"
        }`}
      >
        {value === null ? "—" : `${trendGlyph(value)} ${fmtPercent(value)}`}
      </div>
      <div className="mt-2.5 h-1 overflow-hidden rounded-pill bg-line">
        <div
          className={`h-full rounded-pill ${up ? "bg-grad-gain" : "bg-grad-loss"}`}
          style={{ width: `${magnitude}%` }}
        />
      </div>
    </div>
  );
}

function RiskCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint: string;
  tone: "gain" | "loss" | "neutral";
}) {
  const color =
    tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : "text-ink";
  return (
    <div className="glass glass-hover group p-4">
      <div className="label">{label}</div>
      <div className={`tnum mt-1.5 font-mono text-xl font-bold ${color}`}>{value}</div>
      <p className="mt-2 text-2xs leading-relaxed text-ink-faint opacity-0 transition-opacity duration-300 group-hover:opacity-100">
        {hint}
      </p>
    </div>
  );
}

function ErrorState({ symbol, error }: { symbol: string; error: unknown }) {
  const isApiDown = error instanceof ApiError && error.status === 0;
  const isMissing = error instanceof ApiError && error.isMissingData;

  return (
    <div className="mx-auto max-w-xl space-y-6 py-16 text-center">
      <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-line-strong bg-canvas-raised text-2xl">
        {isApiDown ? "🔌" : "🔍"}
      </div>

      <div>
        <h1 className="text-2xl font-bold">
          {isApiDown ? "The API isn't running" : `No data for ${symbol} yet`}
        </h1>
        <p className="mt-2 text-ink-muted">
          {isApiDown
            ? "Start the backend and reload this page."
            : isMissing
              ? "This ticker is in the catalog but its price history hasn't been ingested."
              : "Something went wrong loading this symbol."}
        </p>
      </div>

      <div className="glass p-4 text-left">
        <div className="label mb-2">Run this</div>
        <code className="block rounded-lg bg-canvas px-3 py-2 font-mono text-xs text-cyan">
          {isApiDown
            ? "uvicorn forecaster.api.main:app --reload"
            : `forecaster ingest ${symbol} --hot --years 5`}
        </code>
      </div>

      <div className="mx-auto max-w-md">
        <SearchBar placeholder="Try another company…" />
      </div>

      <div className="flex flex-wrap justify-center gap-2">
        <span className="text-xs text-ink-faint">Ready to explore:</span>
        {["AAPL", "NVDA", "TSLA", "MSFT"].map((s) => (
          <Link key={s} href={`/s/${s}`} className="btn-chip tnum font-mono">
            {s}
          </Link>
        ))}
      </div>
    </div>
  );
}
