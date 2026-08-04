"use client";

import {
  cn,
  fmtNumber,
  fmtPValue,
  fmtPercentPoints,
  significanceLabel,
  skillClass,
} from "@/lib/format";
import { Badge, InfoTip } from "@/components/ui/primitives";
import { useExplain } from "@/components/ExplainMode";
import type { LeaderboardRow } from "@/lib/types";

/**
 * Model leaderboard.
 *
 * Two non-negotiable behaviours:
 *
 * 1. **Baselines are always rendered**, visually marked, and cannot be filtered
 *    out. If a tree model loses to `naive_last_price`, the user sees it.
 * 2. **Losing models are coloured as losses**, not as neutral. Rounding a
 *    negative skill score up to "grey" is exactly the presentational dishonesty
 *    this project exists to avoid.
 */
export function Leaderboard({
  rows,
  target,
}: {
  rows: LeaderboardRow[];
  target: string;
}) {
  const { enabled: explain } = useExplain();
  const best = rows.find((r) => !r.is_baseline && (r.rmse_skill_pct ?? -1) > 0);

  return (
    <div className="space-y-3">
      {!best && (
        <div className="rounded-card border border-warn/30 bg-warn/5 p-3 text-sm text-ink-muted">
          <span className="font-medium text-warn">No model beat the baseline.</span>{" "}
          Every model below performed worse than assuming no change. For
          directional price targets this is the expected result, and it is shown
          rather than hidden.
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <Th>Model</Th>
              <Th align="right">
                {explain ? "Beat doing nothing by" : "Skill vs naive"}
                <InfoTip label="skill">
                  How much smaller this model&apos;s error is than the naive
                  baseline&apos;s. Negative means the naive forecast was better.
                </InfoTip>
              </Th>
              <Th align="right">
                {explain ? "Error vs naive" : "MASE"}
                <InfoTip label="MASE">
                  Below 1.0 beats the naive forecast; above 1.0 loses to it.
                </InfoTip>
              </Th>
              <Th align="right">R²</Th>
              {target !== "vol_ratio" && (
                <Th align="right">
                  {explain ? "Right calls" : "Hit rate"}
                  <InfoTip label="hit rate">
                    Share of up/down calls that were correct. Compare to the base
                    rate, not to 50%.
                  </InfoTip>
                </Th>
              )}
              <Th align="right">
                {explain ? "Real or luck?" : "DM p-value"}
                <InfoTip label="significance">
                  Probability of seeing this gap if the model and baseline were
                  genuinely equal. Below 0.05 means it is unlikely to be noise.
                </InfoTip>
              </Th>
              <Th align="right">
                {explain ? "Band accuracy" : "Coverage"}
                <InfoTip label="coverage">
                  Share of outcomes that landed inside the 80% prediction band.
                  Close to 0.80 means the uncertainty estimate is honest.
                </InfoTip>
              </Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isBest = best?.model === row.model;
              const sig = significanceLabel(row.dm_pvalue);

              return (
                <tr
                  key={row.model}
                  className={cn(
                    "border-b border-line/60 transition-colors hover:bg-canvas-raised/60",
                    row.is_baseline && "bg-canvas-raised/30",
                  )}
                >
                  <td className="py-2.5 pr-4">
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "font-medium",
                          row.is_baseline ? "text-ink-muted" : "text-ink",
                        )}
                      >
                        {row.display_name ?? row.model}
                      </span>
                      {row.is_baseline && <Badge tone="neutral">baseline</Badge>}
                      {isBest && <Badge tone="gain">best</Badge>}
                    </div>
                  </td>
                  <Td className={skillClass(row.rmse_skill_pct)}>
                    {fmtPercentPoints(row.rmse_skill_pct, 2)}
                  </Td>
                  <Td
                    className={
                      row.mase == null
                        ? "text-ink-faint"
                        : row.mase < 1
                          ? "text-gain"
                          : "text-loss"
                    }
                  >
                    {fmtNumber(row.mase, 3)}
                  </Td>
                  <Td className="text-ink-muted">{fmtNumber(row.r2, 3)}</Td>
                  {target !== "vol_ratio" && (
                    <Td className="text-ink-muted">
                      {row.hit_rate == null ? (
                        <span className="text-ink-faint" title="Makes no directional call">
                          n/a
                        </span>
                      ) : (
                        <>
                          {fmtNumber(row.hit_rate * 100, 1)}%
                          {row.base_rate != null && (
                            <span className="ml-1 text-2xs text-ink-faint">
                              vs {fmtNumber(row.base_rate * 100, 1)}%
                            </span>
                          )}
                        </>
                      )}
                    </Td>
                  )}
                  <Td className={sig.className}>{fmtPValue(row.dm_pvalue)}</Td>
                  <Td
                    className={
                      row.interval_coverage != null &&
                      Math.abs(row.interval_coverage - 0.8) < 0.05
                        ? "text-gain"
                        : "text-ink-muted"
                    }
                  >
                    {fmtNumber(row.interval_coverage, 3)}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs leading-relaxed text-ink-faint">
        Baselines run through the same code path as every other model and cannot
        be removed from this table. A leaderboard without one is not a result.
      </p>
    </div>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={cn(
        "pb-2 text-2xs font-medium uppercase tracking-wider text-ink-faint",
        align === "right" ? "pl-4 text-right" : "pr-4",
      )}
    >
      {children}
    </th>
  );
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <td className={cn("tnum py-2.5 pl-4 text-right font-mono text-xs", className)}>
      {children}
    </td>
  );
}
