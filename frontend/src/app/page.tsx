import Link from "next/link";
import { Badge, Callout } from "@/components/ui/primitives";

const DEMO = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPY"];

const HEADLINE = {
  skill: "+14.2%",
  r2: "0.229",
  symbols: "12 / 12",
  pvalue: "p < 0.01",
};

export default function HomePage() {
  return (
    <div className="space-y-16 py-6">
      {/* ── hero ────────────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <Badge tone="accent">Walk-forward validated · purged splits · published null results</Badge>

        <h1 className="max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
          Most stock-prediction projects are measuring their own bugs.
        </h1>

        <p className="max-w-2xl text-lg leading-relaxed text-ink-muted">
          This one measures a baseline. It forecasts realised volatility{" "}
          <span className="tnum font-mono text-gain">{HEADLINE.skill}</span> better
          than a random walk — and reports, with equal prominence, that it finds{" "}
          <span className="text-ink">no directional price edge at all</span>.
        </p>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <Link
            href="/s/AAPL"
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Explore a ticker
          </Link>
          <Link
            href="/methodology"
            className="rounded-lg border border-line-strong px-4 py-2 text-sm font-medium text-ink-muted transition-colors hover:border-ink-faint hover:text-ink"
          >
            How it&apos;s validated
          </Link>
        </div>

        <div className="flex flex-wrap gap-2 pt-1">
          {DEMO.map((symbol) => (
            <Link
              key={symbol}
              href={`/s/${symbol}`}
              className="tnum rounded-md border border-line bg-canvas-raised px-2.5 py-1 font-mono text-xs text-ink-muted transition-colors hover:border-accent/50 hover:text-accent"
            >
              {symbol}
            </Link>
          ))}
        </div>
      </section>

      {/* ── headline result ─────────────────────────────────────────────── */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ResultTile
          label="RMSE skill vs naive"
          value={HEADLINE.skill}
          note="5-day realised volatility, LightGBM"
          tone="gain"
        />
        <ResultTile label="R²" value={HEADLINE.r2} note="pooled across the universe" />
        <ResultTile
          label="Symbols with positive skill"
          value={HEADLINE.symbols}
          note="consistency, not cherry-picking"
          tone="gain"
        />
        <ResultTile
          label="Significance"
          value={HEADLINE.pvalue}
          note="Diebold-Mariano, 11 of 12 symbols"
        />
      </section>

      {/* ── the honest bit ──────────────────────────────────────────────── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">
          The result this project is most proud of is a negative one
        </h2>

        <Callout tone="warn" title="Directional price prediction does not work">
          Across 12 symbols, 4 horizons and every model tested, nothing beat the
          naive baseline. The best non-baseline model scored{" "}
          <span className="tnum font-mono text-loss">-0.30%</span> RMSE skill —
          worse than predicting zero — and directional hit rates sat consistently{" "}
          <em>below</em> the base rate, meaning the models were less accurate than
          a rule that says &ldquo;up&rdquo; every day.
          <br />
          <br />
          That is what weak-form market efficiency looks like in data. Any project
          claiming otherwise should be asked to show its baseline.
        </Callout>

        <div className="grid gap-4 md:grid-cols-3">
          <Feature
            title="Purged walk-forward"
            body="With an h-day target, the last h training labels are computed from bars inside the test window. They get dropped. Every fold self-checks and raises rather than warns."
          />
          <Feature
            title="Seven baselines, always visible"
            body="Naive, drift, EWMA, seasonal, historical mean, always-long, coin flip. They run through the same code path and cannot be filtered out of a leaderboard."
          />
          <Feature
            title="Calibrated uncertainty"
            body="Split-conformal prediction intervals. Measured coverage is 0.810 against a 0.800 target — the bands mean what they say."
          />
        </div>
      </section>

      {/* ── the test ────────────────────────────────────────────────────── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">
          The test that keeps it honest
        </h2>
        <div className="card overflow-hidden">
          <div className="border-b border-line px-4 py-2 font-mono text-2xs text-ink-faint">
            backend/tests/unit/test_leakage.py
          </div>
          <pre className="overflow-x-auto p-4 font-mono text-xs leading-relaxed text-ink-muted">
{`def test_shuffled_target_yields_no_skill() -> None:
    """Permute the labels, then run the whole pipeline.

    This does not check one specific mistake. It checks the
    conclusion. If any stage leaks -- a feature peeking ahead,
    a scaler fitted on everything, an off-by-one, an index
    misalignment -- the model finds the target through the leak
    and posts positive skill on pure noise.
    """
    skill = _run_walk_forward(frame, shuffle_target=True)
    assert skill < 0.02`}
          </pre>
        </div>
        <p className="text-sm text-ink-muted">
          Paired with a positive control on planted-signal data, so the suite
          cannot pass by the pipeline simply being broken.
        </p>
      </section>
    </div>
  );
}

function ResultTile({
  label,
  value,
  note,
  tone = "neutral",
}: {
  label: string;
  value: string;
  note: string;
  tone?: "neutral" | "gain";
}) {
  return (
    <div className="card p-4">
      <div className="label">{label}</div>
      <div
        className={`tnum mt-2 font-mono text-2xl tracking-tight ${
          tone === "gain" ? "text-gain" : "text-ink"
        }`}
      >
        {value}
      </div>
      <div className="mt-1 text-xs text-ink-faint">{note}</div>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="card p-4">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">{body}</p>
    </div>
  );
}
