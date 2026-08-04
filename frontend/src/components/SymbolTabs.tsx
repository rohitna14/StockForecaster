"use client";

import Link from "next/link";
import { cn } from "@/lib/format";

const TABS = [
  { id: "overview", label: "Overview", icon: "◉", href: "" },
  { id: "models", label: "Models", icon: "◈", href: "/models" },
  { id: "backtest", label: "Backtest", icon: "◧", href: "/backtest" },
] as const;

export function SymbolTabs({
  symbol,
  active,
}: {
  symbol: string;
  active: "overview" | "models" | "backtest";
}) {
  return (
    <nav className="flex gap-1.5 overflow-x-auto pb-1">
      {TABS.map((tab) => {
        const isActive = tab.id === active;
        return (
          <Link
            key={tab.id}
            href={`/s/${symbol}${tab.href}`}
            className={cn(
              "group relative flex shrink-0 items-center gap-2 rounded-pill px-4 py-2 text-sm font-medium transition-all duration-300 ease-spring",
              isActive
                ? "bg-grad-violet text-white shadow-glow"
                : "border border-line-strong text-ink-muted hover:-translate-y-0.5 hover:border-violet/60 hover:text-ink",
            )}
          >
            <span className={cn("text-xs", isActive && "animate-pulse")}>{tab.icon}</span>
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
