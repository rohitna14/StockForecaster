"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { cn, fmtCompact, sectorColor } from "@/lib/format";
import type { InstrumentSummary } from "@/lib/types";

const POPULAR = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY"];

/**
 * Always-visible search with a live dropdown.
 *
 * Previously this lived behind ⌘K, which meant most people never found it.
 * Search by company name works ("apple", "micro", "amazon") because the backend
 * ranks name matches by market cap and pushes symbols that actually have price
 * history to the top — so the first result is the one you meant.
 */
export function SearchBar({
  size = "md",
  autoFocus = false,
  placeholder = "Search Apple, Tesla, NVDA…",
}: {
  size?: "md" | "lg";
  autoFocus?: boolean;
  placeholder?: string;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<InstrumentSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);

  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      setRecent(JSON.parse(localStorage.getItem("sf.recent") ?? "[]").slice(0, 5));
    } catch {
      /* storage unavailable */
    }
  }, []);

  // ⌘K / Ctrl+K still focuses it, for people who expect that.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
      if (e.key === "Escape") {
        setOpen(false);
        inputRef.current?.blur();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // Debounced so typing "GOOGL" fires one request, not five.
  useEffect(() => {
    const term = query.trim();
    if (!term) {
      setResults([]);
      setLoading(false);
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
    }, 160);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  const quickPicks = useMemo(
    () => (recent.length ? recent : POPULAR.slice(0, 5)),
    [recent],
  );

  function go(symbol: string) {
    const upper = symbol.toUpperCase();
    try {
      const next = [upper, ...recent.filter((s) => s !== upper)].slice(0, 5);
      localStorage.setItem("sf.recent", JSON.stringify(next));
      setRecent(next);
    } catch {
      /* ignore */
    }
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
    router.push(`/s/${upper}`);
  }

  const showDropdown = open && (query.trim().length > 0 || quickPicks.length > 0);
  const big = size === "lg";

  return (
    <div ref={wrapRef} className="relative w-full">
      <div
        className={cn(
          "group relative flex items-center gap-3 rounded-pill border bg-canvas-raised/80 backdrop-blur-xl transition-all duration-300 ease-spring",
          big ? "px-5 py-4" : "px-4 py-2.5",
          open
            ? "border-violet shadow-glow"
            : "border-line-strong hover:border-violet/50",
        )}
      >
        <SearchIcon className={cn("shrink-0 text-ink-faint transition-colors group-focus-within:text-violet", big ? "h-5 w-5" : "h-4 w-4")} />
        <input
          ref={inputRef}
          value={query}
          autoFocus={autoFocus}
          onFocus={() => setOpen(true)}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            const list = query.trim() ? results : [];
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((i) => Math.min(i + 1, Math.max(list.length - 1, 0)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (e.key === "Enter") {
              e.preventDefault();
              const chosen = list[active];
              if (chosen) go(chosen.symbol);
              else if (query.trim()) go(query.trim());
            }
          }}
          placeholder={placeholder}
          aria-label="Search stocks by name or symbol"
          className={cn(
            "w-full bg-transparent outline-none placeholder:text-ink-faint",
            big ? "text-base" : "text-sm",
          )}
        />
        {loading ? (
          <div className="h-4 w-4 shrink-0 animate-spin-slow rounded-full border-2 border-line-strong border-t-violet" />
        ) : query ? (
          <button
            type="button"
            onClick={() => { setQuery(""); inputRef.current?.focus(); }}
            aria-label="Clear search"
            className="shrink-0 rounded-full p-1 text-ink-faint transition-colors hover:bg-canvas-hover hover:text-ink"
          >
            <CloseIcon className="h-3.5 w-3.5" />
          </button>
        ) : (
          <kbd className="hidden shrink-0 rounded border border-line-strong px-1.5 py-0.5 font-mono text-[10px] text-ink-faint sm:block">
            ⌘K
          </kbd>
        )}
      </div>

      {showDropdown && (
        <div className="absolute left-0 right-0 top-full z-50 mt-2 animate-slide-down overflow-hidden rounded-2xl border border-line-strong bg-canvas-overlay/95 shadow-lift backdrop-blur-2xl">
          {!query.trim() && (
            <>
              <div className="px-4 pb-1.5 pt-3">
                <span className="label">{recent.length ? "Recently viewed" : "Popular"}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 px-3 pb-3">
                {quickPicks.map((symbol) => (
                  <button
                    key={symbol}
                    type="button"
                    onClick={() => go(symbol)}
                    className="btn-chip tnum font-mono"
                  >
                    {symbol}
                  </button>
                ))}
              </div>
            </>
          )}

          {query.trim() && results.length === 0 && !loading && (
            <div className="px-4 py-6 text-center">
              <p className="text-sm text-ink-muted">
                Nothing matched &ldquo;{query}&rdquo;
              </p>
              <p className="mt-1 text-xs text-ink-faint">
                Try a company name like &ldquo;Apple&rdquo; or a ticker like &ldquo;AAPL&rdquo;
              </p>
            </div>
          )}

          {query.trim() && results.length > 0 && (
            <ul className="max-h-[22rem] overflow-y-auto p-1.5" role="listbox">
              {results.map((item, index) => (
                <li key={item.symbol}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={index === active}
                    onMouseEnter={() => setActive(index)}
                    onClick={() => go(item.symbol)}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors",
                      index === active ? "bg-violet/15" : "hover:bg-canvas-hover",
                    )}
                  >
                    <span
                      className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-xs font-bold text-white"
                      style={{ background: sectorColor(item.sector) }}
                    >
                      {item.symbol.slice(0, 2)}
                    </span>

                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="tnum font-mono text-sm font-semibold">{item.symbol}</span>
                        {item.has_data ? (
                          <span className="rounded-pill bg-gain/15 px-1.5 py-0.5 text-[10px] font-medium text-gain">
                            live
                          </span>
                        ) : (
                          <span className="rounded-pill bg-line px-1.5 py-0.5 text-[10px] text-ink-faint">
                            no data
                          </span>
                        )}
                      </span>
                      <span className="block truncate text-xs text-ink-muted">{item.name}</span>
                    </span>

                    <span className="hidden shrink-0 text-right sm:block">
                      {item.market_cap ? (
                        <span className="tnum block font-mono text-xs text-ink-muted">
                          ${fmtCompact(item.market_cap)}
                        </span>
                      ) : null}
                      {item.sector && (
                        <span className="block text-[10px] text-ink-faint">{item.sector}</span>
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="flex items-center justify-between border-t border-line px-4 py-2 text-[10px] text-ink-faint">
            <span>↑↓ navigate · ↵ open · esc close</span>
            <span>{results.length > 0 ? `${results.length} results` : "type to search"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </svg>
  );
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
