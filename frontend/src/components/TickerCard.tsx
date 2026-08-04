"use client";

import Link from "next/link";
import { Sparkline } from "@/components/ui/Sparkline";
import { cn, fmtPercent, fmtPrice, sectorColor, trendGlyph } from "@/lib/format";

export interface TickerCardData {
  symbol: string;
  name?: string | null;
  sector?: string | null;
  last?: number | null;
  changePct?: number | null;
  points?: number[];
}

export function TickerCard({ data, rank }: { data: TickerCardData; rank?: number }) {
  const rising = (data.changePct ?? 0) >= 0;
  const accent = sectorColor(data.sector);

  return (
    <Link
      href={`/s/${data.symbol}`}
      className="ring-grad glass glass-hover group relative block overflow-hidden p-4"
    >
      {/* Sector accent bar */}
      <span
        className="absolute inset-x-0 top-0 h-[2px] opacity-70 transition-opacity group-hover:opacity-100"
        style={{ background: accent }}
      />

      {rank !== undefined && (
        <span className="tnum absolute right-3 top-3 font-mono text-[10px] text-ink-faint">
          #{rank}
        </span>
      )}

      <div className="flex items-start gap-3">
        <span
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl text-xs font-bold text-white shadow-lg transition-transform duration-300 ease-spring group-hover:scale-110"
          style={{ background: accent }}
        >
          {data.symbol.slice(0, 2)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="tnum font-mono text-sm font-bold tracking-tight">{data.symbol}</div>
          {data.name && (
            <div className="truncate text-xs text-ink-muted">{data.name}</div>
          )}
        </div>
      </div>

      <div className="mt-3 flex items-end justify-between gap-3">
        <div>
          <div className="tnum font-mono text-lg font-semibold tracking-tight">
            {fmtPrice(data.last)}
          </div>
          <div
            className={cn(
              "tnum flex items-center gap-1 font-mono text-xs font-medium",
              rising ? "text-gain" : "text-loss",
            )}
          >
            <span>{trendGlyph(data.changePct)}</span>
            {fmtPercent(data.changePct)}
            <span className="text-ink-faint">90d</span>
          </div>
        </div>

        {data.points && data.points.length > 1 && (
          <Sparkline points={data.points} width={96} height={38} />
        )}
      </div>

      {data.sector && (
        <div className="mt-3 border-t border-line pt-2">
          <span
            className="rounded-pill px-2 py-0.5 text-[10px] font-medium"
            style={{ background: `${accent}22`, color: accent }}
          >
            {data.sector}
          </span>
        </div>
      )}
    </Link>
  );
}

export function TickerCardSkeleton() {
  return (
    <div className="glass p-4">
      <div className="flex items-start gap-3">
        <div className="skeleton h-10 w-10 rounded-xl" />
        <div className="flex-1 space-y-2">
          <div className="skeleton h-3 w-14" />
          <div className="skeleton h-2.5 w-24" />
        </div>
      </div>
      <div className="mt-4 flex items-end justify-between">
        <div className="space-y-2">
          <div className="skeleton h-5 w-20" />
          <div className="skeleton h-3 w-16" />
        </div>
        <div className="skeleton h-9 w-24" />
      </div>
    </div>
  );
}
