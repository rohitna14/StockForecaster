"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { Leaderboard } from "@/components/charts/Leaderboard";
import { Badge, Callout, EmptyState, Skeleton } from "@/components/ui/primitives";
import { cn, fmtNumber } from "@/lib/format";
import type { EvaluationResponse } from "@/lib/types";

const TARGETS = [
  {
    id: "vol_ratio",
    label: "Volatility",
    blurb: "Forecast whether the next period is calmer or more turbulent than the last. This is where the measurable edge is.",
  },
  {
    id: "return",
    label: "Direction",
    blurb: "Forecast the next period's return. Included so the null result is visible, not hidden.",
  },
] as const;

const HORIZONS = [1, 5, 10, 21];
const MODELS = ["ridge", "elastic_net", "random_forest", "gradient_boosting", "lightgbm"];

export default function ModelsPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol: raw } = use(params);
  const symbol = raw.toUpperCase();

  const [target, setTarget] = useState<string>("vol_ratio");
  const [horizon, setHorizon] = useState(5);
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await api.evaluate({
          symbol,
          models: MODELS,
          horizon,
          target_type: target,
          train_size: 504,
          test_size: 63,
        }),
      );
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Evaluation failed. Check that the backend is running.",
      );
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [symbol, horizon, target]);

  useEffect(() => {
    void run();
  }, [run]);

  const selected = TARGETS.find((t) => t.id === target);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            <span className="tnum font-mono">{symbol}</span> · model comparison
          </h1>
          <p className="mt-1 text-sm text-ink-muted">
            Walk-forward with purged splits. Every run is computed live.
          </p>
        </div>
        <Link
          href={`/s/${symbol}`}
          className="text-sm text-ink-muted transition-colors hover:text-ink"
        >
          ← Overview
        </Link>
      </div>

      {/* ── controls ────────────────────────────────────────────────────── */}
      <div className="card space-y-4 p-4">
        <div>
          <div className="label mb-2">Target</div>
          <div className="flex flex-wrap gap-2">
            {TARGETS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTarget(t.id)}
                className={cn(
                  "rounded-lg border px-3 py-1.5 text-sm transition-colors",
                  target === t.id
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-line-strong text-ink-muted hover:border-ink-faint hover:text-ink",
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          {selected && (
            <p className="mt-2 text-xs leading-relaxed text-ink-faint">{selected.blurb}</p>
          )}
        </div>

        <div>
          <div className="label mb-2">Horizon (trading days)</div>
          <div className="flex flex-wrap gap-2">
            {HORIZONS.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setHorizon(h)}
                className={cn(
                  "tnum rounded-lg border px-3 py-1.5 font-mono text-sm transition-colors",
                  horizon === h
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-line-strong text-ink-muted hover:border-ink-faint hover:text-ink",
                )}
              >
                {h}d
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── results ─────────────────────────────────────────────────────── */}
      {loading && (
        <div className="card space-y-3 p-4">
          <div className="flex items-center gap-2 text-sm text-ink-muted">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            Running walk-forward evaluation — fitting {MODELS.length + 1} models across
            every fold…
          </div>
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      )}

      {error && !loading && (
        <EmptyState
          title="Couldn't run the evaluation"
          description={error}
          action={
            <button
              type="button"
              onClick={() => void run()}
              className="rounded-lg border border-line-strong px-3 py-1.5 text-sm text-ink-muted hover:text-ink"
            >
              Retry
            </button>
          }
        />
      )}

      {data && !loading && (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
            <Badge tone="neutral">{data.n_folds} folds</Badge>
            <Badge tone="neutral">{data.n_samples.toLocaleString()} samples</Badge>
            <Badge tone="neutral">{data.n_features} features</Badge>
            <Badge tone="accent">selected: {data.selected_model}</Badge>
            <span className="tnum font-mono text-ink-faint">
              {fmtNumber(data.duration_seconds, 1)}s
            </span>
          </div>

          <Leaderboard rows={data.leaderboard} target={target} />

          <Callout title="How the selected model was chosen">
            Selection runs on every fold <em>except the last</em>, and the held-out
            final fold gives an unbiased estimate of its performance. Picking the
            top row of this table and then reporting that row&apos;s score would be
            selecting and reporting on the same data — with six models and a
            handful of folds, the winner would mostly be whichever got luckiest.
          </Callout>
        </>
      )}
    </div>
  );
}
