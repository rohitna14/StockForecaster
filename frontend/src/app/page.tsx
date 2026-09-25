import Link from "next/link";
import { api } from "@/lib/api";
import { SearchBar } from "@/components/SearchBar";
import { TickerCard, type TickerCardData } from "@/components/TickerCard";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import { StatRing } from "@/components/ui/StatRing";
import type { InstrumentSummary, SparklineResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const FEATURED = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM", "SPY", "XOM", "WMT", "JNJ"];

const SECTORS = [
  { name: "Technology", color: "#7C5CFF", symbols: "AAPL · MSFT · NVDA" },
  { name: "Health Care", color: "#22D3EE", symbols: "JNJ" },
  { name: "Finance", color: "#2DD97B", symbols: "JPM" },
  { name: "Energy", color: "#FB923C", symbols: "XOM" },
  { name: "Consumer", color: "#E056C1", symbols: "AMZN · TSLA · WMT" },
  { name: "Index", color: "#FBBF24", symbols: "SPY" },
];

export default async function HomePage() {
  let cards: TickerCardData[] = [];

  try {
    const [spark, instruments] = await Promise.all([
      api.getSparklines(FEATURED, 90),
      api.searchInstruments(undefined, 40),
    ]);
    cards = buildCards(spark, instruments.items);
  } catch {
    cards = FEATURED.map((symbol) => ({ symbol }));
  }

  const movers = [...cards].sort(
    (a, b) => Math.abs(b.changePct ?? 0) - Math.abs(a.changePct ?? 0),
  );

  return (
    <div className="space-y-20 pb-10">
      {/* ═══════════════ HERO ═══════════════ */}
      <section className="relative pt-8">
        <div className="pointer-events-none absolute -top-24 left-1/2 h-72 w-[42rem] -translate-x-1/2 rounded-full bg-violet/20 blur-[120px]" />

        <div className="relative mx-auto max-w-3xl text-center">
          <span className="inline-flex animate-fade-in items-center gap-2 rounded-pill border border-violet/40 bg-violet/10 px-4 py-1.5 text-xs font-medium text-violet-bright">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet" />
            Walk-forward validated · 200 tests · null results published
          </span>

          <h1 className="mt-6 animate-fade-up text-4xl font-black leading-[1.05] tracking-tight sm:text-6xl">
            Stock forecasting
            <br />
            that <span className="grad-text">shows its work.</span>
          </h1>

          <p className="mx-auto mt-5 max-w-xl animate-fade-up text-base leading-relaxed text-ink-muted sm:text-lg">
            Search any company. Get volatility forecasts benchmarked against
            honest baselines — plus the results that <em>didn&apos;t</em> work,
            because those matter too.
          </p>

          <div className="mx-auto mt-8 max-w-xl animate-fade-up">
            <SearchBar size="lg" placeholder="Try “Apple”, “Tesla”, or “NVDA”…" />
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
            <span className="text-xs text-ink-faint">Popular:</span>
            {["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"].map((symbol) => (
              <Link key={symbol} href={`/s/${symbol}`} className="btn-chip tnum font-mono">
                {symbol}
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════ HEADLINE STATS ═══════════════ */}
      <section>
        <div className="grid gap-4 stagger sm:grid-cols-2 lg:grid-cols-4">
          <HeroStat
            label="Better than random walk"
            value={14.17}
            decimals={2}
            suffix="%"
            showPlus
            hint="RMSE reduction on 5-day volatility"
            gradient="bg-grad-gain"
            glow="shadow-glow-gain"
          />
          <HeroStat
            label="Tickers with positive skill"
            value={12}
            decimals={0}
            suffix=" / 12"
            hint="Consistency, not cherry-picking"
            gradient="bg-grad-violet"
            glow="shadow-glow"
          />
          <HeroStat
            label="Model runs logged"
            value={864}
            decimals={0}
            separator
            hint="436k out-of-sample predictions"
            gradient="bg-grad-cyan"
            glow="shadow-glow"
          />
          <HeroStat
            label="Interval accuracy"
            value={81}
            decimals={0}
            suffix="%"
            hint="vs 80% target — calibrated"
            gradient="bg-grad-violet"
            glow="shadow-glow"
          />
        </div>
      </section>

      {/* ═══════════════ TICKER GRID ═══════════════ */}
      <section>
        <SectionHead
          eyebrow="Live data"
          title="Explore the universe"
          subtitle="12 tickers with five years of history. Click any card to dive in."
          action={
            <Link href="/explore" className="btn-ghost">
              See all 6,600+ →
            </Link>
          }
        />
        <div className="grid gap-4 stagger sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {cards.map((card, i) => (
            <TickerCard key={card.symbol} data={card} rank={i + 1} />
          ))}
        </div>
      </section>

      {/* ═══════════════ MOVERS ═══════════════ */}
      {movers.length > 2 && (
        <section>
          <SectionHead
            eyebrow="90-day"
            title="Biggest movers"
            subtitle="Largest absolute moves across the tracked universe."
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {movers.slice(0, 6).map((card, i) => (
              <MoverRow key={card.symbol} data={card} rank={i + 1} />
            ))}
          </div>
        </section>
      )}

      {/* ═══════════════ SECTORS ═══════════════ */}
      <section>
        <SectionHead
          eyebrow="Browse"
          title="By sector"
          subtitle="Colour is consistent everywhere — you'll learn to read it."
        />
        <div className="grid gap-3 stagger sm:grid-cols-2 lg:grid-cols-3">
          {SECTORS.map((sector) => (
            <Link
              key={sector.name}
              href={`/explore?sector=${encodeURIComponent(sector.name)}`}
              className="ring-grad glass glass-hover group relative overflow-hidden p-5"
            >
              <span
                className="absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-20 blur-2xl transition-opacity group-hover:opacity-40"
                style={{ background: sector.color }}
              />
              <div className="relative flex items-center gap-3">
                <span
                  className="h-10 w-1.5 rounded-full"
                  style={{ background: sector.color }}
                />
                <div>
                  <div className="font-semibold">{sector.name}</div>
                  <div className="tnum mt-0.5 font-mono text-xs text-ink-faint">
                    {sector.symbols}
                  </div>
                </div>
                <span className="ml-auto text-ink-faint transition-transform duration-300 ease-spring group-hover:translate-x-1">
                  →
                </span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ═══════════════ THE HONEST BIT ═══════════════ */}
      <section className="grid gap-5 lg:grid-cols-[1.25fr_1fr]">
        <div className="glass relative overflow-hidden p-7">
          <span className="absolute -left-16 -top-16 h-52 w-52 rounded-full bg-loss/15 blur-3xl" />
          <div className="relative">
            <span className="rounded-pill border border-loss/40 bg-loss/10 px-3 py-1 text-xs font-semibold text-loss">
              Published negative result
            </span>
            <h3 className="mt-4 text-2xl font-bold tracking-tight">
              Predicting <em>direction</em> doesn&apos;t work.
            </h3>
            <p className="mt-3 leading-relaxed text-ink-muted">
              Across 12 symbols, 4 horizons and every model tested — 336
              evaluations — nothing beat a naive baseline. The best model scored{" "}
              <span className="tnum font-mono font-semibold text-loss">-0.30%</span>,
              worse than predicting zero. Hit rates sat <em>below</em> the base
              rate, meaning the models were less accurate than a rule that says
              &ldquo;up&rdquo; every day.
            </p>
            <p className="mt-3 leading-relaxed text-ink-muted">
              That&apos;s what market efficiency looks like in data. Most projects
              bury this. We lead with it — and it&apos;s why the volatility number
              above is worth believing.
            </p>
            <Link href="/methodology" className="btn-primary mt-6">
              How we validate →
            </Link>
          </div>
        </div>

        <div className="grid gap-4">
          <div className="glass flex items-center gap-5 p-6">
            <StatRing value={0.81} target={0.8} label="coverage" />
            <div>
              <div className="font-semibold">Calibrated uncertainty</div>
              <p className="mt-1 text-sm leading-relaxed text-ink-muted">
                When the model says 80% confident, it&apos;s right 81% of the
                time. Most forecasting tools never check.
              </p>
            </div>
          </div>

          <div className="glass p-6">
            <div className="font-semibold">7 baselines, always visible</div>
            <p className="mt-1 text-sm leading-relaxed text-ink-muted">
              Naive, drift, EWMA, seasonal, mean, always-long, coin flip. They
              run through the same code path and can&apos;t be hidden from a
              leaderboard.
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {["naive", "drift", "ewma", "seasonal", "mean", "always-long", "coin-flip"].map((b) => (
                <span key={b} className="rounded-pill bg-line px-2 py-0.5 text-[10px] text-ink-muted">
                  {b}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════ CTA ═══════════════ */}
      <section className="glass relative overflow-hidden p-10 text-center">
        <span className="absolute inset-0 bg-grad-primary opacity-[0.07]" />
        <div className="relative mx-auto max-w-xl">
          <h3 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Pick a company. See the evidence.
          </h3>
          <p className="mt-3 text-ink-muted">
            Every forecast comes with its measured skill against a baseline — and
            says so plainly when that skill is negative.
          </p>
          <div className="mx-auto mt-6 max-w-md">
            <SearchBar size="lg" placeholder="Search any company…" />
          </div>
        </div>
      </section>
    </div>
  );
}

/* ── helpers ─────────────────────────────────────────────────────────── */

function buildCards(
  spark: SparklineResponse,
  instruments: InstrumentSummary[],
): TickerCardData[] {
  const meta = new Map(instruments.map((i) => [i.symbol, i]));
  return FEATURED.map((symbol) => {
    const series = spark.series[symbol];
    const info = meta.get(symbol);
    return {
      symbol,
      name: info?.name ?? null,
      sector: info?.sector ?? null,
      last: series?.last ?? null,
      changePct: series?.change_pct ?? null,
      points: series?.points,
    };
  }).filter((c) => c.points || c.last);
}

function SectionHead({
  eyebrow,
  title,
  subtitle,
  action,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <span className="label">{eyebrow}</span>
        <h2 className="mt-1 text-2xl font-bold tracking-tight">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

function HeroStat({
  label,
  value,
  decimals = 0,
  suffix,
  showPlus = false,
  separator = false,
  hint,
  gradient,
  glow,
}: {
  label: string;
  value: number;
  decimals?: number;
  suffix?: string;
  showPlus?: boolean;
  separator?: boolean;
  hint: string;
  gradient: string;
  glow: string;
}) {
  return (
    <div className={`glass glass-hover relative overflow-hidden p-5 ${glow}`}>
      <span className={`absolute inset-x-0 top-0 h-[2px] ${gradient}`} />
      <div className="label">{label}</div>
      <div className="mt-2 font-mono text-3xl font-black tracking-tight">
        <AnimatedNumber
          value={value}
          decimals={decimals}
          suffix={suffix}
          showPlus={showPlus}
          separator={separator}
        />
      </div>
      <div className="mt-1.5 text-xs text-ink-faint">{hint}</div>
    </div>
  );
}

function MoverRow({ data, rank }: { data: TickerCardData; rank: number }) {
  const rising = (data.changePct ?? 0) >= 0;
  return (
    <Link
      href={`/s/${data.symbol}`}
      className="glass glass-hover flex items-center gap-4 p-4"
    >
      <span className="tnum w-5 font-mono text-xs text-ink-faint">{rank}</span>
      <span className="tnum flex-1 font-mono text-sm font-bold">{data.symbol}</span>
      <span
        className={`tnum rounded-pill px-2.5 py-1 font-mono text-xs font-semibold ${
          rising ? "bg-gain/15 text-gain" : "bg-loss/15 text-loss"
        }`}
      >
        {rising ? "▲" : "▼"} {Math.abs((data.changePct ?? 0) * 100).toFixed(1)}%
      </span>
    </Link>
  );
}
