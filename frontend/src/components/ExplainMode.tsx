"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { cn } from "@/lib/format";

/**
 * Global "explain like I'm five" mode.
 *
 * When on, every metric label is replaced by plain English and the extended
 * explanation is shown inline rather than hidden behind a tooltip. This is the
 * feature that makes the product usable by someone who does not already know
 * what a Sortino ratio is — which is almost everyone.
 *
 * The preference persists, because having to re-enable it on every page is the
 * kind of small friction that makes people stop using a thing.
 */

interface ExplainContextValue {
  enabled: boolean;
  toggle: () => void;
}

const ExplainContext = createContext<ExplainContextValue>({
  enabled: false,
  toggle: () => {},
});

const STORAGE_KEY = "sf.explainMode";

export function ExplainProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    try {
      setEnabled(window.localStorage.getItem(STORAGE_KEY) === "1");
    } catch {
      // Private browsing / storage disabled — default off, no crash.
    }
  }, []);

  const toggle = useCallback(() => {
    setEnabled((previous) => {
      const next = !previous;
      try {
        window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const value = useMemo(() => ({ enabled, toggle }), [enabled, toggle]);

  return <ExplainContext.Provider value={value}>{children}</ExplainContext.Provider>;
}

export function useExplain() {
  return useContext(ExplainContext);
}

export function ExplainToggle() {
  const { enabled, toggle } = useExplain();

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={enabled}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
        enabled
          ? "border-accent/50 bg-accent/10 text-accent"
          : "border-line-strong text-ink-muted hover:border-ink-faint hover:text-ink",
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full transition-colors",
          enabled ? "bg-accent" : "bg-ink-faint",
        )}
      />
      Explain mode
    </button>
  );
}

/**
 * Renders the technical label normally, and swaps in plain English when explain
 * mode is on.
 */
export function Explainable({
  technical,
  plain,
  className,
}: {
  technical: ReactNode;
  plain: ReactNode;
  className?: string;
}) {
  const { enabled } = useExplain();
  return <span className={className}>{enabled ? plain : technical}</span>;
}
