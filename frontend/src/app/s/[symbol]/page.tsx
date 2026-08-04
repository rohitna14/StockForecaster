import Link from "next/link";
import { notFound } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import {
  fmtCompact,
  fmtDate,
  fmtNumber,
  fmtPercent,
  fmtPrice,
  trendClass,
  trendGlyph,
} from "@/lib/format";
import { Badge, Callout, EmptyState, Stat } from "@/components/ui/primitives";
import { PriceChart } from "@/components/charts/PriceChart";
import type { InstrumentSummaryStats, OHLCVResponse, RiskResponse } from "@/lib/types";

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

  try {
    [summary, ohlcv] = await Promise.all([api.getSummary(symbol), api.getOHLCV(symbol, 500)]);
  } catch (error) {
    if (error instanceof ApiError && error.status === 0) {
      return (
        <EmptyState
          title="The API isn't reachable"
          description="Start the backend with `uvicorn forecaster.api.main:app --reload` from the backend directory, then reload this page."
        />
      );
    }
    if (error instanceof ApiError && error.isMissingData) {
      return (
        <EmptyState
          title={`No data for ${symbol} yet`}
          description={`This ticker hasn't been ingested. Run \`forecaster ingest ${symbol} --hot\` to fetch five years of history.`}
          action={
            <Link
              href="/s/AAPL"
              className="rounded-lg border border-line-strong px-3 py-1.5 text-sm text-ink-muted hover:text-ink"
            >
              Try AAPL instead
            </Link>
          }
        />
      );
    }
    notFound();
  }

  try {
    risk = await api.getRisk(symbol);
  } catch {
    // Risk needs 30+ observations; a short history is not a page failure.
  }

  return (
    <div className="space-y-8">
      {/* ── header ──────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="tnum font-mono text-3xl font-semibold tracking-tight">{symbol}</h1>
            <Badge tone={summary.source_tier === "hot" ? "accent" : "neutral"}>
              {summary.source_tier} tier
            </Badge>
          </div>
          <p className="mt-1 text-sm text-ink-muted">
            {summary.n_bars.toLocaleString()} bars · {fmtDate(summary.first_date)} →{" "}
            {fmtDate(summary.as_of)}
          </p>
        </div>

        <div className="text-right">
          <div className="tnum font-mono text-3xl tracking-tight">
            {fmtPrice(summary.last_close)}
          </div>
          <div className={`tnum font-mono text-sm ${trendClass(summary.change_1d)}`}>
            {trendGlyph(summary.change_1d)} {fmtPercent(summary.change_1d)}
          </div>
        </div>
      </div>

      <nav className="flex gap-1 border-b border-line">
        <TabLink href={`/s/${symbol}`} active>
          Overview
        </TabLink>
        <TabLink href={`/s/${symbol}/models`}>Models</TabLink>
        <TabLink href={`/s/${symbol}/backtest`}>Backtest</TabLink>
      </nav>

      {/* ── stats ───────────────────────────────────────────────────────── */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="1 week" value={fmtPercent(summary.change_1w)} />
        <Stat label="1 month" value={fmtPercent(summary.change_1m)} />
        <Stat label="1 year" value={fmtPercent(summary.change_1y)} />
        <Stat
          label="Volume"
          value={fmtCompact(summary.volume)}
          hint={`20-day average: ${fmtCompact(summary.avg_volume_20d)}`}
        />
      </div>

      {/* ── chart ───────────────────────────────────────────────────────── */}
      <section className="card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Price history</h2>
          <span className="tnum font-mono text-2xs text-ink-faint">
            52w {fmtPrice(summary.range_52w_low)} – {fmtPrice(summary.range_52w_high)}
          </span>
        </div>
        <PriceChart bars={ohlcv.bars} height={440} />
        <p className="mt-3 text-2xs text-ink-faint">
          Split- and dividend-adjusted. The whole OHLC bar is back-adjusted and
          volume inverted, so gap and range features stay correct across splits.
        </p>
      </section>

      {/* ── risk ────────────────────────────────────────────────────────── */}
      {risk && (
        <section>
          <h2 className="mb-3 text-sm font-semibold">Realised risk</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Stat
              label="Volatility"
              value={fmtPercent(risk.annual_volatility, 1, { signed: false })}
              hint="Annualised size of typical price moves. A measure of uncertainty, not of direction."
            />
            <Stat
              label="Sharpe"
              value={fmtNumber(risk.sharpe)}
              hint="Return per unit of risk. Above 1.0 is decent — but this is the stock's own Sharpe, not a strategy's."
            />
            <Stat
              label="Sortino"
              value={fmtNumber(risk.sortino)}
              hint="Like Sharpe, but only penalises downside moves. Upside volatility isn't risk."
            />
            <Stat
              label="Max drawdown"
              value={fmtPercent(risk.max_drawdown, 1)}
              hint="Worst peak-to-trough fall. This is the number that decides whether a position is actually holdable."
            />
            <Stat
              label="VaR 95%"
              value={fmtPercent(risk.var_95)}
              hint="On the worst 5% of days, losses exceeded this."
            />
          </div>
        </section>
      )}

      <Callout tone="warn" title="Before you read anything into a forecast">
        On this project&apos;s own testing, no model beat a naive baseline at
        predicting <em>direction</em> for any symbol or horizon. The measurable
        edge is in forecasting <em>volatility</em>. The models tab shows both,
        with the baseline always in the table.
      </Callout>
    </div>
  );
}

function TabLink({
  href,
  children,
  active = false,
}: {
  href: string;
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <Link
      href={href}
      className={`-mb-px border-b-2 px-3.5 py-2 text-sm transition-colors ${
        active
          ? "border-accent text-ink"
          : "border-transparent text-ink-muted hover:border-line-strong hover:text-ink"
      }`}
    >
      {children}
    </Link>
  );
}
