"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";
import type { InstrumentSummary } from "@/lib/types";

const QUICK_PICKS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "AMZN"];

/**
 * ⌘K ticker search.
 *
 * Debounced at 180ms — fast enough to feel instant, slow enough that typing
 * "GOOGL" fires one request rather than five.
 */
export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<InstrumentSummary[]>([]);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen((v) => !v);
      }
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 30);
    else {
      setQuery("");
      setResults([]);
      setActive(0);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const term = query.trim();
    if (!term) {
      setResults([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const page = await api.searchInstruments(term, 8);
        if (!cancelled) {
          setResults(page.items);
          setActive(0);
        }
      } catch {
        if (!cancelled) setResults([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 180);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query, open]);

  const items = useMemo(
    () =>
      results.length > 0
        ? results
        : QUICK_PICKS.map(
            (symbol) =>
              ({ symbol, name: "", sector: null, industry: null, exchange: null,
                 market_cap: null, tier: "cold", first_date: null,
                 last_date: null }) as InstrumentSummary,
          ),
    [results],
  );

  function go(symbol: string) {
    setOpen(false);
    router.push(`/s/${symbol.toUpperCase()}`);
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-lg border border-line-strong bg-canvas-raised px-3 py-1.5 text-xs text-ink-muted transition-colors hover:border-ink-faint hover:text-ink"
      >
        <span>Search ticker</span>
        <kbd className="rounded border border-line-strong px-1 py-0.5 font-mono text-[10px] text-ink-faint">
          ⌘K
        </kbd>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[100] flex items-start justify-center bg-black/70 p-4 pt-[12vh] backdrop-blur-sm"
          onClick={() => setOpen(false)}
          role="presentation"
        >
          <div
            className="w-full max-w-lg animate-fade-in overflow-hidden rounded-xl border border-line-strong bg-canvas-overlay shadow-2xl"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="Ticker search"
          >
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setActive((i) => Math.min(i + 1, items.length - 1));
                } else if (event.key === "ArrowUp") {
                  event.preventDefault();
                  setActive((i) => Math.max(i - 1, 0));
                } else if (event.key === "Enter") {
                  event.preventDefault();
                  const chosen = items[active];
                  if (chosen) go(chosen.symbol);
                }
              }}
              placeholder="Search by symbol or company name…"
              className="w-full border-b border-line bg-transparent px-4 py-3.5 text-sm outline-none placeholder:text-ink-faint"
            />

            <div className="max-h-80 overflow-y-auto p-1.5">
              {!query && (
                <div className="px-2.5 py-1.5 text-2xs uppercase tracking-wider text-ink-faint">
                  Popular
                </div>
              )}
              {loading && (
                <div className="px-3 py-4 text-sm text-ink-faint">Searching…</div>
              )}
              {!loading && query && items.length === 0 && (
                <div className="px-3 py-4 text-sm text-ink-muted">
                  Nothing matched &ldquo;{query}&rdquo;.
                </div>
              )}
              {!loading &&
                items.map((item, index) => (
                  <button
                    key={item.symbol}
                    type="button"
                    onMouseEnter={() => setActive(index)}
                    onClick={() => go(item.symbol)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                      index === active ? "bg-accent/15 text-ink" : "text-ink-muted",
                    )}
                  >
                    <span className="flex items-center gap-2.5">
                      <span className="tnum font-mono font-medium">{item.symbol}</span>
                      {item.name && (
                        <span className="truncate text-xs text-ink-faint">{item.name}</span>
                      )}
                    </span>
                    {item.sector && (
                      <span className="shrink-0 text-2xs text-ink-faint">{item.sector}</span>
                    )}
                  </button>
                ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
