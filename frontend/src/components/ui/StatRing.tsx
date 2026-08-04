"use client";

import { useId } from "react";

/**
 * Circular progress ring with a target marker.
 *
 * Used for calibration coverage, where the interesting question isn't "how
 * high?" but "how close to target?" — so the ring draws the actual value and a
 * tick shows where it *should* be. A plain number can't express that.
 */
export function StatRing({
  value,
  target,
  label,
  size = 88,
}: {
  value: number;
  target?: number;
  label?: string;
  size?: number;
}) {
  const id = useId().replace(/:/g, "");
  const stroke = 7;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, value));
  const offset = circumference * (1 - clamped);

  const onTarget = target === undefined || Math.abs(value - target) <= 0.05;
  const color = onTarget ? "#2DD97B" : "#FBBF24";

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id={`ring-${id}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={color} />
            <stop offset="100%" stopColor="#7C5CFF" />
          </linearGradient>
        </defs>

        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#22222E"
          strokeWidth={stroke}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={`url(#ring-${id})`}
          strokeWidth={stroke}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.22,1,0.36,1)" }}
        />

        {target !== undefined && (
          <circle
            cx={size / 2 + radius * Math.cos(2 * Math.PI * target)}
            cy={size / 2 + radius * Math.sin(2 * Math.PI * target)}
            r={2.5}
            fill="#F2F3F7"
          />
        )}
      </svg>

      <div className="absolute inset-0 grid place-items-center">
        <div className="text-center">
          <div className="tnum font-mono text-lg font-bold leading-none">
            {(clamped * 100).toFixed(0)}%
          </div>
          {label && (
            <div className="mt-0.5 text-[9px] uppercase tracking-wider text-ink-faint">
              {label}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
