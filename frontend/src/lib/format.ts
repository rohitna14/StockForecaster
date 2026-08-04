import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Formatting helpers.
 *
 * Every numeric display in the app goes through one of these. The shared rule:
 * a value that is null, NaN or infinite renders as an em-dash, never as "NaN",
 * "null" or a silently coerced 0 — the last of which would read as a real
 * measurement.
 */

const DASH = "—";

function isNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function fmtPrice(value: number | null | undefined, currency = "USD"): string {
  if (!isNum(value)) return DASH;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function fmtPercent(
  value: number | null | undefined,
  digits = 2,
  { signed = true }: { signed?: boolean } = {},
): string {
  if (!isNum(value)) return DASH;
  const formatted = (value * 100).toFixed(digits);
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${formatted}%`;
}

/** For values already expressed in percentage points (e.g. skill scores). */
export function fmtPercentPoints(
  value: number | null | undefined,
  digits = 2,
  { signed = true }: { signed?: boolean } = {},
): string {
  if (!isNum(value)) return DASH;
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}%`;
}

export function fmtNumber(value: number | null | undefined, digits = 2): string {
  if (!isNum(value)) return DASH;
  return value.toFixed(digits);
}

export function fmtCompact(value: number | null | undefined): string {
  if (!isNum(value)) return DASH;
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function fmtDate(value: string | null | undefined): string {
  if (!value) return DASH;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return DASH;
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function fmtPValue(value: number | null | undefined): string {
  if (!isNum(value)) return DASH;
  if (value < 0.001) return "<0.001";
  return value.toFixed(3);
}

/** Direction glyph. Paired with colour so meaning survives colour-blindness. */
export function trendGlyph(value: number | null | undefined): string {
  if (!isNum(value) || value === 0) return "";
  return value > 0 ? "▲" : "▼";
}

export function trendClass(value: number | null | undefined): string {
  if (!isNum(value) || value === 0) return "text-ink-muted";
  return value > 0 ? "text-gain" : "text-loss";
}

/**
 * Colour for a skill score.
 *
 * Deliberately does NOT treat "slightly negative" as neutral. A model that
 * loses to the naive baseline is a failure and is coloured as one, because the
 * entire point of this project is that such results usually get quietly rounded
 * up to "promising".
 */
export function skillClass(value: number | null | undefined): string {
  if (!isNum(value)) return "text-ink-faint";
  if (value > 5) return "text-gain";
  if (value > 0) return "text-gain/80";
  return "text-loss";
}

export function significanceLabel(p: number | null | undefined): {
  label: string;
  className: string;
} {
  if (!isNum(p)) return { label: "not tested", className: "text-ink-faint" };
  if (p < 0.01) return { label: "significant (p<0.01)", className: "text-gain" };
  if (p < 0.05) return { label: "significant (p<0.05)", className: "text-gain/80" };
  return { label: "not significant", className: "text-warn" };
}
