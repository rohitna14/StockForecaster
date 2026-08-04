"use client";

import { useId } from "react";
import { cn } from "@/lib/format";

/**
 * Inline SVG sparkline with a gradient fill.
 *
 * Hand-rolled rather than pulled from a chart library: a card grid renders a
 * dozen of these, and a full charting runtime per card would cost more than the
 * rest of the page combined. This is ~40 lines and renders server-side.
 */
export function Sparkline({
  points,
  className,
  height = 40,
  width = 120,
  strokeWidth = 2,
  animate = true,
}: {
  points: number[];
  className?: string;
  height?: number;
  width?: number;
  strokeWidth?: number;
  animate?: boolean;
}) {
  const id = useId().replace(/:/g, "");

  if (!points || points.length < 2) {
    return (
      <div
        className={cn("rounded bg-line/40", className)}
        style={{ height, width }}
        aria-hidden
      />
    );
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const pad = strokeWidth;

  const coords = points.map((value, i) => {
    const x = (i / (points.length - 1)) * (width - pad * 2) + pad;
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y] as const;
  });

  const line = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = `${line} L${width - pad},${height} L${pad},${height} Z`;

  const rising = points[points.length - 1]! >= points[0]!;
  const stroke = rising ? "#2DD97B" : "#FF5A6E";

  return (
    <svg
      className={cn("overflow-visible", className)}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      fill="none"
      aria-hidden
    >
      <defs>
        <linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={stroke} stopOpacity="0.32" />
          <stop offset="100%" stopColor={stroke} stopOpacity="0" />
        </linearGradient>
      </defs>

      <path d={area} fill={`url(#fill-${id})`} />
      <path
        d={line}
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={animate ? "animate-draw-line" : undefined}
        style={animate ? { strokeDasharray: 1000 } : undefined}
      />
      <circle
        cx={coords[coords.length - 1]![0]}
        cy={coords[coords.length - 1]![1]}
        r={strokeWidth + 0.6}
        fill={stroke}
      />
    </svg>
  );
}
