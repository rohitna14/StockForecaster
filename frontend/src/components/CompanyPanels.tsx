import { cn, fmtCompact, fmtDate, fmtNumber, fmtPercent, fmtPrice } from "@/lib/format";
import type { CompanyProfile, NewsItem, Quote } from "@/lib/types";

/* ── Fundamentals grid ─────────────────────────────────────────────────── */
export function Fundamentals({ profile }: { profile: CompanyProfile }) {
  const groups: { title: string; rows: [string, string, string?][] }[] = [
    {
      title: "Valuation",
      rows: [
        ["Market cap", profile.market_cap ? `$${fmtCompact(profile.market_cap)}` : "—"],
        ["Enterprise value", profile.enterprise_value ? `$${fmtCompact(profile.enterprise_value)}` : "—"],
        ["P/E (trailing)", fmtNumber(profile.trailing_pe), "Price divided by last year's earnings. Lower can mean cheaper — or a struggling business."],
        ["P/E (forward)", fmtNumber(profile.forward_pe), "Same, but against expected earnings. Depends on analyst forecasts being right."],
        ["Price / book", fmtNumber(profile.price_to_book)],
        ["PEG ratio", fmtNumber(profile.peg_ratio), "P/E adjusted for growth. Around 1 is often considered fair."],
      ],
    },
    {
      title: "Earnings",
      rows: [
        ["EPS (trailing)", profile.eps_trailing ? fmtPrice(profile.eps_trailing) : "—", "Profit per share over the last year."],
        ["EPS (forward)", profile.eps_forward ? fmtPrice(profile.eps_forward) : "—"],
        ["Revenue", profile.revenue ? `$${fmtCompact(profile.revenue)}` : "—"],
        ["Revenue growth", fmtPercent(profile.revenue_growth, 1)],
        ["Profit margin", fmtPercent(profile.profit_margin, 1)],
        ["Next earnings", fmtDate(profile.earnings_date)],
      ],
    },
    {
      title: "Trading",
      rows: [
        ["52w high", fmtPrice(profile.fifty_two_week_high)],
        ["52w low", fmtPrice(profile.fifty_two_week_low)],
        ["50d average", fmtPrice(profile.fifty_day_average)],
        ["200d average", fmtPrice(profile.two_hundred_day_average)],
        ["Avg volume", profile.average_volume ? fmtCompact(profile.average_volume) : "—"],
        ["Beta", fmtNumber(profile.beta), "How much it moves relative to the market. Above 1 is more volatile than the index."],
      ],
    },
    {
      title: "Income & shares",
      rows: [
        ["Dividend yield", profile.dividend_yield ? `${fmtNumber(profile.dividend_yield, 2)}%` : "—"],
        ["Dividend rate", profile.dividend_rate ? fmtPrice(profile.dividend_rate) : "—"],
        ["Payout ratio", fmtPercent(profile.payout_ratio, 0)],
        ["Shares out", profile.shares_outstanding ? fmtCompact(profile.shares_outstanding) : "—"],
        ["Float", profile.float_shares ? fmtCompact(profile.float_shares) : "—"],
        ["Short ratio", fmtNumber(profile.short_ratio)],
      ],
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {groups.map((group) => (
        <div key={group.title} className="glass p-4">
          <div className="label mb-3">{group.title}</div>
          <dl className="space-y-2">
            {group.rows.map(([label, value, hint]) => (
              <div key={label} className="group flex items-baseline justify-between gap-3">
                <dt className="text-xs text-ink-muted" title={hint}>
                  {label}
                  {hint && <span className="ml-1 text-ink-faint">ⓘ</span>}
                </dt>
                <dd className="tnum font-mono text-xs font-medium">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}

/* ── Analyst consensus ─────────────────────────────────────────────────── */
export function AnalystPanel({
  profile,
  currentPrice,
}: {
  profile: CompanyProfile;
  currentPrice: number | null;
}) {
  const target = profile.target_mean_price;
  if (!target && !profile.recommendation) return null;

  const upside =
    target && currentPrice ? target / currentPrice - 1 : null;

  const tone: Record<string, string> = {
    strong_buy: "text-gain",
    buy: "text-gain",
    hold: "text-warn",
    underperform: "text-loss",
    sell: "text-loss",
  };

  return (
    <div className="glass p-5">
      <div className="label mb-3">Analyst consensus</div>

      <div className="flex flex-wrap items-end gap-6">
        {profile.recommendation && (
          <div>
            <div
              className={cn(
                "text-2xl font-bold capitalize",
                tone[profile.recommendation] ?? "text-ink",
              )}
            >
              {profile.recommendation.replace("_", " ")}
            </div>
            <div className="mt-0.5 text-xs text-ink-faint">
              {profile.analyst_count ? `${profile.analyst_count} analysts` : "consensus"}
            </div>
          </div>
        )}

        {target && (
          <div>
            <div className="tnum font-mono text-2xl font-bold">{fmtPrice(target)}</div>
            <div className="mt-0.5 text-xs text-ink-faint">mean target</div>
          </div>
        )}

        {upside !== null && (
          <div>
            <div
              className={cn(
                "tnum font-mono text-2xl font-bold",
                upside >= 0 ? "text-gain" : "text-loss",
              )}
            >
              {fmtPercent(upside, 1)}
            </div>
            <div className="mt-0.5 text-xs text-ink-faint">implied upside</div>
          </div>
        )}
      </div>

      {profile.target_low_price && profile.target_high_price && (
        <div className="mt-4">
          <div className="mb-1.5 flex justify-between text-2xs text-ink-faint">
            <span className="tnum font-mono">{fmtPrice(profile.target_low_price)}</span>
            <span>analyst target range</span>
            <span className="tnum font-mono">{fmtPrice(profile.target_high_price)}</span>
          </div>
          <div className="relative h-1.5 rounded-pill bg-line">
            <div className="absolute inset-0 rounded-pill bg-grad-primary opacity-30" />
            {currentPrice && (
              <div
                className="absolute top-1/2 h-3.5 w-1 -translate-y-1/2 rounded-full bg-ink shadow-glow"
                style={{
                  left: `${Math.max(0, Math.min(100, ((currentPrice - profile.target_low_price) / Math.max(profile.target_high_price - profile.target_low_price, 1e-9)) * 100))}%`,
                }}
                title={`Current: ${fmtPrice(currentPrice)}`}
              />
            )}
          </div>
        </div>
      )}

      <p className="mt-3 text-2xs leading-relaxed text-ink-faint">
        Analyst targets are opinions, not forecasts this project validated. They
        are shown for context and are not part of any model here.
      </p>
    </div>
  );
}

/* ── About ─────────────────────────────────────────────────────────────── */
export function AboutPanel({ profile }: { profile: CompanyProfile }) {
  if (!profile.summary) return null;
  return (
    <div className="glass p-5">
      <div className="label mb-2">About {profile.name}</div>
      <p className="text-sm leading-relaxed text-ink-muted">{profile.summary}</p>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-ink-faint">
        {profile.industry && <span>Industry: <span className="text-ink-muted">{profile.industry}</span></span>}
        {profile.country && <span>HQ: <span className="text-ink-muted">{profile.country}</span></span>}
        {profile.employees && (
          <span>Employees: <span className="tnum text-ink-muted">{profile.employees.toLocaleString()}</span></span>
        )}
        {profile.exchange && <span>Exchange: <span className="text-ink-muted">{profile.exchange}</span></span>}
        {profile.website && (
          <a href={profile.website} target="_blank" rel="noopener noreferrer" className="text-violet hover:underline">
            Website ↗
          </a>
        )}
      </div>
    </div>
  );
}

/* ── News ──────────────────────────────────────────────────────────────── */
export function NewsPanel({ items }: { items: NewsItem[] }) {
  if (!items.length) {
    return (
      <div className="glass p-5">
        <div className="label mb-2">Recent news</div>
        <p className="text-sm text-ink-faint">
          No headlines available for this ticker right now.
        </p>
      </div>
    );
  }

  return (
    <div className="glass p-5">
      <div className="label mb-3">Recent news</div>
      <ul className="divide-y divide-line">
        {items.map((item, i) => (
          <li key={i} className="py-2.5 first:pt-0 last:pb-0">
            <a
              href={item.url ?? "#"}
              target="_blank"
              rel="noopener noreferrer"
              className="group block"
            >
              <div className="text-sm font-medium leading-snug transition-colors group-hover:text-violet-bright">
                {item.title}
              </div>
              <div className="mt-1 flex items-center gap-2 text-2xs text-ink-faint">
                {item.publisher && <span>{item.publisher}</span>}
                {item.published_at && <span>· {fmtDate(item.published_at)}</span>}
              </div>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ── Live quote strip ──────────────────────────────────────────────────── */
export function QuoteBadge({ quote }: { quote: Quote }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-2xs text-ink-faint">
      <span className="flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn" />
        {quote.is_delayed ? "Delayed ~15 min" : "Live"}
      </span>
      {quote.as_of && <span>· as of {quote.as_of.slice(0, 16).replace("T", " ")}</span>}
      {quote.source && <span>· {quote.source}</span>}
    </div>
  );
}
