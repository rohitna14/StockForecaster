"use client";

import { useState, type ReactNode } from "react";
import { cn, fmtPercentPoints, trendClass, trendGlyph } from "@/lib/format";

/* ── Info tooltip ─────────────────────────────────────────────────────────
 * Every metric in this UI carries one. A dashboard that shows "Sortino 1.42"
 * with no explanation is decoration, not information — and the tooltips are the
 * main defence against a user reading a 53% hit rate as a money printer.
 */
export function InfoTip({ children, label }: { children: ReactNode; label: string }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={`What is ${label}?`}
        className="ml-1 grid h-3.5 w-3.5 place-items-center rounded-full border border-line-strong text-[9px] leading-none text-ink-faint transition-colors hover:border-violet hover:text-violet"
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onBlur={() => setOpen(false)}
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 animate-fade-in rounded-lg border border-line-strong bg-canvas-overlay p-3 text-xs font-normal leading-relaxed text-ink-muted shadow-2xl"
        >
          {children}
        </span>
      )}
    </span>
  );
}

/* ── Stat tile ─────────────────────────────────────────────────────────── */
export function Stat({
  label,
  value,
  delta,
  hint,
  emphasis = false,
  className,
}: {
  label: string;
  value: ReactNode;
  delta?: number | null;
  hint?: ReactNode;
  emphasis?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("glass glass-hover p-4", className)}>
      <div className="flex items-center">
        <span className="label">{label}</span>
        {hint && <InfoTip label={label}>{hint}</InfoTip>}
      </div>
      <div
        className={cn(
          "tnum mt-2 font-mono tracking-tight",
          emphasis ? "text-2xl" : "text-xl",
        )}
      >
        {value}
      </div>
      {delta !== undefined && delta !== null && (
        <div className={cn("tnum mt-1 font-mono text-xs", trendClass(delta))}>
          {trendGlyph(delta)} {fmtPercentPoints(delta * 100, 2)}
        </div>
      )}
    </div>
  );
}

/* ── Badge ─────────────────────────────────────────────────────────────── */
export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "gain" | "loss" | "warn";
  className?: string;
}) {
  const tones = {
    neutral: "border-line-strong bg-canvas-overlay text-ink-muted",
    accent: "border-violet/40 bg-violet/10 text-violet",
    gain: "border-gain/40 bg-gain/10 text-gain",
    loss: "border-loss/40 bg-loss/10 text-loss",
    warn: "border-warn/40 bg-warn/10 text-warn",
  } as const;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-2xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ── Callout ───────────────────────────────────────────────────────────── */
export function Callout({
  title,
  tone = "neutral",
  children,
}: {
  title?: string;
  tone?: "neutral" | "warn" | "gain" | "loss";
  children: ReactNode;
}) {
  const tones = {
    neutral: "border-line-strong bg-canvas-raised",
    warn: "border-warn/30 bg-warn/5",
    gain: "border-gain/30 bg-gain/5",
    loss: "border-loss/30 bg-loss/5",
  } as const;

  return (
    <div className={cn("rounded-card border p-4", tones[tone])}>
      {title && <div className="mb-1.5 text-sm font-semibold">{title}</div>}
      <div className="text-sm leading-relaxed text-ink-muted">{children}</div>
    </div>
  );
}

/* ── Loading skeletons ─────────────────────────────────────────────────── */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} />;
}

export function StatSkeleton() {
  return (
    <div className="glass p-4">
      <Skeleton className="h-2.5 w-16" />
      <Skeleton className="mt-3 h-6 w-24" />
    </div>
  );
}

/* ── Empty state ───────────────────────────────────────────────────────── */
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="glass flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="grid h-10 w-10 place-items-center rounded-full border border-line-strong text-ink-faint">
        ∅
      </div>
      <div className="text-sm font-medium">{title}</div>
      <p className="max-w-sm text-sm leading-relaxed text-ink-muted">{description}</p>
      {action}
    </div>
  );
}

/* ── Section header ────────────────────────────────────────────────────── */
export function SectionHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-end justify-between gap-4">
      <div>
        <h2 className="text-base font-semibold tracking-tight">{title}</h2>
        {subtitle && <p className="mt-0.5 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}
