"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { cn, fmtCompact, sectorColor } from "@/lib/format";
import type { SearchMatch } from "@/lib/types";

const POPULAR = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY"];

/**
 * Smart company search.
 *
 * Backed by a fuzzy index rather than SQL LIKE, so it handles the four ways
 * people actually search:
 *
 *   ticker    AAPL          nickname   google -> GOOGL, facebook -> META
 *   name      Apple Inc     partial    mic -> Microsoft, tes -> Tesla
 *   typo      aple, teslla, micrsoft, nvdia
 *
 * Pressing Enter without picking anything navigates to the top-ranked match,
 * so you never need to know the exact ticker.
 */
export function SearchBar({
  size = "md",
  autoFocus = false,
  placeholder = "Search Apple, Tesla, google…",
}: {
  size?: "md" | "lg";
  autoFocus?: boolean;
  placeholder?: string;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchMatch[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const [navigating, setNavigating] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);

  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      setRecent(JSON.parse(localStorage.getItem("sf.recent") ?? "[]").slice(0, 6));
    } catch {
      /* storage unavailable */
    }
  }, []);

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

  // Debounced so typing "microsoft" fires one request, not nine.
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
        const response = await api.search(term, 8);
        if (!cancelled) {
          setResults(response.results);
          setActive(0);
        }
      } catch {
        if (!cancelled) setResults([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  function go(symbol: string) {
    const upper = symbol.toUpperCase();
    try {
      const next = [upper, ...recent.filter((s) => s !== upper)].slice(0, 6);
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

  /**
   * Enter with nothing highlighted resolves server-side, so "google" lands on
   * Alphabet even if the dropdown hasn't returned yet.
   */
  async function submit() {
    const term = query.trim();
    if (!term) return;

    const highlighted = results[active];
    if (highlighted) {
      go(highlighted.symbol);
      return;
    }

    setNavigating(true);
    try {
      const match = await api.resolve(term);
      go(match.symbol);
    } catch {
      go(term); // let the symbol page's on-demand ingest have a go
    } finally {
      setNavigating(false);
    }
  }

  const big = size === "lg";
  const showDropdown = open;

  return (
    <div ref={wrapRef} className="relative w-full">
      <div
        className={cn(
          "group relative flex items-center gap-3 rounded-pill border bg-canvas-raised/80 backdrop-blur-xl transition-all duration-300 ease-spring",
          big ? "px-5 py-4" : "px-4 py-2.5",
          open ? "border-violet shadow-glow" : "border-line-strong hover:border-violet/50",
        )}
      >
        <SearchIcon
          className={cn(
            "shrink-0 text-ink-faint transition-colors group-focus-within:text-violet",
            big ? "h-5 w-5" : "h-4 w-4",
          )}
        />
        <input
          ref={inputRef}
          value={query}
          autoFocus={autoFocus}
          onFocus={() => setOpen(true)}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((i) => Math.min(i + 1, Math.max(results.length - 1, 0)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (e.key === "Enter") {
              e.preventDefault();
              void submit();
            }
          }}
          placeholder={placeholder}
          aria-label="Search stocks by name, ticker or nickname"
          className={cn(
            "w-full bg-transparent outline-none placeholder:text-ink-faint",
            big ? "text-base" : "text-sm",
          )}
        />
        {loading || navigating ? (
          <div className="h-4 w-4 shrink-0 animate-spin-slow rounded-full border-2 border-line-strong border-t-violet" />
        ) : query ? (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              inputRef.current?.focus();
            }}
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
                {(recent.length ? recent : POPULAR).map((symbol) => (
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
              <div className="border-t border-line px-4 py-2 text-[10px] leading-relaxed text-ink-faint">
                Try a nickname (<span className="text-violet">google</span>,{" "}
                <span className="text-violet">facebook</span>), a partial (
                <span className="text-violet">mic</span>), or even a typo (
                <span className="text-violet">teslla</span>).
              </div>
            </>
          )}

          {query.trim() && results.length === 0 && !loading && (
            <div className="px-4 py-6 text-center">
              <p className="text-sm text-ink-muted">
                Nothing matched &ldquo;{query}&rdquo;
              </p>
              <p className="mt-1 text-xs text-ink-faint">
                Press Enter anyway — we&apos;ll try to fetch it.
              </p>
            </div>
          )}

          {query.trim() && results.length > 0 && (
            <ul className="max-h-[24rem] overflow-y-auto p-1.5" role="listbox">
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
                      {/* "Apple Inc. (AAPL)" — name leads, because that is what
                          people searched for. */}
                      <span className="block truncate text-sm font-medium">
                        {item.name}{" "}
                        <span className="tnum font-mono text-ink-muted">
                          ({item.symbol})
                        </span>
                      </span>
                      <span className="mt-0.5 flex items-center gap-1.5">
                        {item.sector && (
                          <span className="text-[10px] text-ink-faint">{item.sector}</span>
                        )}
                        {index === 0 && (
                          <span className="rounded-pill bg-violet/20 px-1.5 py-0.5 text-[9px] font-medium text-violet-bright">
                            ↵ best match
                          </span>
                        )}
                        {!item.has_data && (
                          <span className="text-[9px] text-ink-faint">will fetch on open</span>
                        )}
                      </span>
                    </span>

                    {item.market_cap ? (
                      <span className="tnum hidden shrink-0 font-mono text-xs text-ink-muted sm:block">
                        ${fmtCompact(item.market_cap)}
                      </span>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="flex items-center justify-between border-t border-line px-4 py-2 text-[10px] text-ink-faint">
            <span>↑↓ navigate · ↵ open best match · esc close</span>
            {results.length > 0 && <span>{results.length} results</span>}
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
