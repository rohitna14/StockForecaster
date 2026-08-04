"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { SearchBar } from "@/components/SearchBar";
import { TickerCardSkeleton } from "@/components/TickerCard";
import { Sparkline } from "@/components/ui/Sparkline";
import { cn, fmtCompact, fmtPercent, fmtPrice, sectorColor } from "@/lib/format";
import type { InstrumentSummary, SparklineSeries } from "@/lib/types";

type SortKey = "relevance" | "marketcap" | "change" | "symbol";
type Filter = "all" | "tradable";

const SORTS: { id: SortKey; label: string }[] = [
  { id: "relevance", label: "Relevance" },
  { id: "marketcap", label: "Market cap" },
  { id: "change", label: "90d change" },
  { id: "symbol", label: "A–Z" },
];

/**
 * useSearchParams() opts a route into client-side rendering, so Next requires a
 * Suspense boundary around it or the build fails while prerendering. The
 * fallback is a skeleton grid rather than a spinner — it holds layout, so the
 * page doesn't jump when results land.
 */
export default function ExplorePage() {
  return (
    <Suspense fallback={<ExploreFallback />}>
      <ExploreContent />
    </Suspense>
  );
}

function ExploreFallback() {
  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div className="skeleton h-3 w-20" />
        <div className="skeleton h-9 w-64" />
        <div className="skeleton h-4 w-96" />
      </div>
      <div className="mx-auto h-14 max-w-2xl skeleton rounded-pill" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <TickerCardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

function ExploreContent() {
  const params = useSearchParams();
  const sectorParam = params.get("sector");

  const [items, setItems] = useState<InstrumentSummary[]>([]);
  const [series, setSeries] = useState<Record<string, SparklineSeries>>({});
  const [sectors, setSectors] = useState<string[]>([]);
  const [sector, setSector] = useState<string | null>(sectorParam);
  const [sort, setSort] = useState<SortKey>("marketcap");
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"grid" | "list">("grid");

  useEffect(() => {
    api.getSectors().then(setSectors).catch(() => setSectors([]));
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const page = await api.searchInstruments(undefined, 60);
      let rows = page.items;
      if (sector) rows = rows.filter((r) => r.sector === sector);
      setItems(rows);

      // Only fetch mini-charts for symbols that actually have history.
      const tradable = rows.filter((r) => r.has_data).map((r) => r.symbol).slice(0, 40);
      if (tradable.length) {
        const spark = await api.getSparklines(tradable, 90);
        setSeries(spark.series);
      } else {
        setSeries({});
      }
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [sector]);

  useEffect(() => {
    void load();
  }, [load]);

  const visible = useMemo(() => {
    let rows = filter === "tradable" ? items.filter((r) => r.has_data) : items;
    rows = [...rows];

    if (sort === "marketcap") {
      rows.sort((a, b) => (b.market_cap ?? 0) - (a.market_cap ?? 0));
    } else if (sort === "symbol") {
      rows.sort((a, b) => a.symbol.localeCompare(b.symbol));
    } else if (sort === "change") {
      rows.sort(
        (a, b) =>
          (series[b.symbol]?.change_pct ?? -Infinity) -
          (series[a.symbol]?.change_pct ?? -Infinity),
      );
    }
    return rows;
  }, [items, filter, sort, series]);

  const tradableCount = items.filter((r) => r.has_data).length;

  return (
    <div className="space-y-8">
      {/* header */}
      <div>
        <span className="label">Explore</span>
        <h1 className="mt-1 text-3xl font-black tracking-tight">
          The <span className="grad-text">universe</span>
        </h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          6,600+ instruments in the catalog.{" "}
          <span className="font-medium text-gain">{tradableCount} have price history</span>{" "}
          loaded and are ready to analyse — the rest need one ingest command.
        </p>
      </div>

      <div className="mx-auto max-w-2xl">
        <SearchBar size="lg" placeholder="Search any company or ticker…" />
      </div>

      {/* controls */}
      <div className="glass space-y-4 p-4">
        <div>
          <div className="label mb-2">Sector</div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setSector(null)}
              className={cn("btn-chip", !sector && "btn-chip-active")}
            >
              All sectors
            </button>
            {sectors.slice(0, 12).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSector(s === sector ? null : s)}
                className={cn("btn-chip", sector === s && "btn-chip-active")}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: sectorColor(s) }}
                />
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="label mb-2">Sort by</div>
            <div className="flex flex-wrap gap-2">
              {SORTS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => setSort(s.id)}
                  className={cn("btn-chip", sort === s.id && "btn-chip-active")}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-end gap-4">
            <div>
              <div className="label mb-2">Show</div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setFilter("all")}
                  className={cn("btn-chip", filter === "all" && "btn-chip-active")}
                >
                  Everything
                </button>
                <button
                  type="button"
                  onClick={() => setFilter("tradable")}
                  className={cn("btn-chip", filter === "tradable" && "btn-chip-active")}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-gain" />
                  With data
                </button>
              </div>
            </div>

            <div>
              <div className="label mb-2">View</div>
              <div className="flex gap-1 rounded-pill border border-line-strong p-1">
                {(["grid", "list"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setView(v)}
                    className={cn(
                      "rounded-pill px-3 py-1 text-xs font-medium transition-all",
                      view === v ? "bg-violet text-white" : "text-ink-faint hover:text-ink",
                    )}
                  >
                    {v === "grid" ? "▦" : "☰"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* results */}
      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <TickerCardSkeleton key={i} />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <div className="glass py-16 text-center">
          <p className="text-ink-muted">Nothing matches those filters.</p>
          <button type="button" onClick={() => { setSector(null); setFilter("all"); }} className="btn-ghost mt-4">
            Reset filters
          </button>
        </div>
      ) : view === "grid" ? (
        <div className="grid gap-4 stagger sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {visible.slice(0, 48).map((item) => (
            <ExploreCard key={item.symbol} item={item} series={series[item.symbol]} />
          ))}
        </div>
      ) : (
        <div className="glass divide-y divide-line overflow-hidden">
          {visible.slice(0, 60).map((item) => (
            <ExploreRow key={item.symbol} item={item} series={series[item.symbol]} />
          ))}
        </div>
      )}

      <p className="text-center text-xs text-ink-faint">
        Showing {Math.min(visible.length, view === "grid" ? 48 : 60)} of {visible.length}
      </p>
    </div>
  );
}

function ExploreCard({
  item,
  series,
}: {
  item: InstrumentSummary;
  series?: SparklineSeries;
}) {
  const accent = sectorColor(item.sector);
  const body = (
    <>
      <span className="absolute inset-x-0 top-0 h-[2px]" style={{ background: accent }} />
      <div className="flex items-start gap-3">
        <span
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl text-xs font-bold text-white transition-transform duration-300 ease-spring group-hover:scale-110"
          style={{ background: accent }}
        >
          {item.symbol.slice(0, 2)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="tnum font-mono text-sm font-bold">{item.symbol}</span>
            {!item.has_data && (
              <span className="rounded-pill bg-line px-1.5 py-0.5 text-[9px] text-ink-faint">
                no data
              </span>
            )}
          </div>
          <div className="truncate text-xs text-ink-muted">{item.name}</div>
        </div>
      </div>

      <div className="mt-3 flex items-end justify-between gap-2">
        <div>
          {series ? (
            <>
              <div className="tnum font-mono text-base font-semibold">{fmtPrice(series.last)}</div>
              <div
                className={cn(
                  "tnum font-mono text-xs font-medium",
                  series.change_pct >= 0 ? "text-gain" : "text-loss",
                )}
              >
                {series.change_pct >= 0 ? "▲" : "▼"} {fmtPercent(series.change_pct)}
              </div>
            </>
          ) : (
            <div className="tnum font-mono text-xs text-ink-faint">
              {item.market_cap ? `$${fmtCompact(item.market_cap)} cap` : "—"}
            </div>
          )}
        </div>
        {series && <Sparkline points={series.points} width={88} height={34} animate={false} />}
      </div>
    </>
  );

  if (!item.has_data) {
    return (
      <div className="glass relative overflow-hidden p-4 opacity-60" title="Not ingested yet">
        {body}
      </div>
    );
  }

  return (
    <Link
      href={`/s/${item.symbol}`}
      className="ring-grad glass glass-hover group relative overflow-hidden p-4"
    >
      {body}
    </Link>
  );
}

function ExploreRow({
  item,
  series,
}: {
  item: InstrumentSummary;
  series?: SparklineSeries;
}) {
  const accent = sectorColor(item.sector);
  const inner = (
    <>
      <span
        className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-[10px] font-bold text-white"
        style={{ background: accent }}
      >
        {item.symbol.slice(0, 2)}
      </span>
      <span className="tnum w-16 shrink-0 font-mono text-sm font-bold">{item.symbol}</span>
      <span className="min-w-0 flex-1 truncate text-sm text-ink-muted">{item.name}</span>
      {item.sector && (
        <span
          className="hidden shrink-0 rounded-pill px-2 py-0.5 text-[10px] lg:block"
          style={{ background: `${accent}22`, color: accent }}
        >
          {item.sector}
        </span>
      )}
      <span className="tnum hidden w-24 shrink-0 text-right font-mono text-xs text-ink-muted sm:block">
        {item.market_cap ? `$${fmtCompact(item.market_cap)}` : "—"}
      </span>
      {series ? (
        <>
          <Sparkline points={series.points} width={64} height={24} animate={false} />
          <span
            className={cn(
              "tnum w-20 shrink-0 text-right font-mono text-xs font-medium",
              series.change_pct >= 0 ? "text-gain" : "text-loss",
            )}
          >
            {fmtPercent(series.change_pct)}
          </span>
        </>
      ) : (
        <span className="w-20 shrink-0 text-right text-[10px] text-ink-faint">no data</span>
      )}
    </>
  );

  if (!item.has_data) {
    return <div className="flex items-center gap-3 px-4 py-2.5 opacity-50">{inner}</div>;
  }
  return (
    <Link
      href={`/s/${item.symbol}`}
      className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-canvas-hover"
    >
      {inner}
    </Link>
  );
}
