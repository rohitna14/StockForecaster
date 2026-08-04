"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/format";

/**
 * Count-up animation for headline figures.
 *
 * Eases out over ~900ms, starting only when the element scrolls into view so
 * numbers below the fold animate when you reach them. Respects
 * prefers-reduced-motion by snapping straight to the final value.
 *
 * Formatting is expressed as *data* (prefix/suffix/decimals) rather than a
 * callback, because this is a Client Component and React Server Components
 * cannot serialise a function across that boundary — passing one throws
 * "Functions cannot be passed directly to Client Components" and takes the
 * whole page down with it.
 */
export function AnimatedNumber({
  value,
  decimals = 2,
  prefix = "",
  suffix = "",
  showPlus = false,
  separator = false,
  className,
  duration = 900,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  /** Prefix positive values with "+" (for deltas and skill scores). */
  showPlus?: boolean;
  /** Thousands separators. */
  separator?: boolean;
  className?: string;
  duration?: number;
}) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || !Number.isFinite(value)) {
      setDisplay(value);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting || started.current) return;
        started.current = true;

        const start = performance.now();
        const tick = (now: number) => {
          const t = Math.min((now - start) / duration, 1);
          // easeOutExpo — fast start, gentle settle
          const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
          setDisplay(value * eased);
          if (t < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.25 },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [value, duration]);

  const body = separator
    ? display.toLocaleString("en-US", {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    : display.toFixed(decimals);

  const sign = showPlus && value > 0 ? "+" : "";

  return (
    <span ref={ref} className={cn("tnum", className)}>
      {sign}
      {prefix}
      {body}
      {suffix}
    </span>
  );
}
