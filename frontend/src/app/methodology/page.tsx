import type { Metadata } from "next";
import { Callout } from "@/components/ui/primitives";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How forecasts here are validated, and why each choice is made.",
};

export default function MethodologyPage() {
  return (
    <article className="mx-auto max-w-3xl space-y-12 py-4">
      <header className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Methodology</h1>
        <p className="text-lg leading-relaxed text-ink-muted">
          The modelling here is ordinary. The validation is the part worth
          defending.
        </p>
      </header>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">
          The problem with the obvious approach
        </h2>
        <p className="leading-relaxed text-ink-muted">
          Download history, compute indicators, split 80/20, train a dozen
          models, report the best R² on the test set. Every step after the
          indicators is wrong, and the errors all push the same way — they make
          results look better than they are.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-line text-left">
                <th className="pb-2 text-2xs font-medium uppercase tracking-wider text-ink-faint">
                  Step
                </th>
                <th className="pb-2 text-2xs font-medium uppercase tracking-wider text-ink-faint">
                  What goes wrong
                </th>
              </tr>
            </thead>
            <tbody className="text-ink-muted">
              {[
                ["Single 80/20 split", "One number from one arbitrary cut point. Move the cut and the winner changes."],
                ["Train through T, test from T+1", "With an h-day target, the last h training labels were built from bars inside the test window."],
                ["Report best-of-N on the test set", "With 15 models and ~100 test days, the winner is whichever got luckiest."],
                ["No baseline", "“R² = 0.62” is meaningless without knowing what doing nothing scores."],
                ["R² as the headline", "Next-day return R² is ~0 for everyone. A rubric calling 0.7 “good” is unreachable."],
              ].map(([step, problem]) => (
                <tr key={step} className="border-b border-line/60">
                  <td className="py-2.5 pr-4 font-medium text-ink">{step}</td>
                  <td className="py-2.5 leading-relaxed">{problem}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Walk-forward with purging</h2>
        <p className="leading-relaxed text-ink-muted">
          Train on the past, test on the immediate future, roll forward, repeat.
          The output is a distribution of out-of-sample scores across many market
          regimes, not a single number that depends on where you happened to cut.
        </p>

        <pre className="card overflow-x-auto p-4 font-mono text-xs leading-relaxed text-ink-muted">
{`fold 0  |=== train ===|·gap·|test|
fold 1        |=== train ===|·gap·|test|
fold 2              |=== train ===|·gap·|test|
                                            ────▶ time`}
        </pre>

        <h3 className="pt-2 text-base font-semibold">The gap is not decoration</h3>
        <p className="leading-relaxed text-ink-muted">
          With an <em>h</em>-bar forward target, the label at bar <em>t</em> is
          computed from the close at <em>t + h</em>. Train through bar{" "}
          <em>T</em> and the labels at <em>T−h+1 … T</em> were all built from bars
          inside the test window. The model has seen the test period&apos;s prices
          through its own labels.
        </p>
        <p className="leading-relaxed text-ink-muted">
          So the last <em>h</em> training labels are dropped. For horizon 5 with a
          test window opening at bar 1000, training stops at bar 994 — not 999.
          Every fold re-checks this at runtime and raises rather than warns.
        </p>

        <Callout tone="neutral" title="Embargo">
          Returns are serially correlated, so bars immediately after a test window
          still carry information about it. An embargo widens the gap further.
          Total gap = horizon + embargo.
        </Callout>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Baselines are models</h2>
        <p className="leading-relaxed text-ink-muted">
          Seven of them — naive last price, historical mean, drift, EWMA,
          seasonal naive, always-long, coin flip. They are fitted per fold and
          ranked on the same leaderboard as everything else, and they cannot be
          filtered out of the UI.
        </p>
        <p className="leading-relaxed text-ink-muted">
          The reference baseline predicts exactly zero, because for a{" "}
          <em>price</em> target the naive forecast is &ldquo;tomorrow equals
          today&rdquo; — which as a return is 0. Random-walk theory says this is
          close to optimal at short horizons, and empirically it is very hard to
          beat.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">
          What &ldquo;15% better&rdquo; can honestly mean
        </h2>
        <Callout tone="warn">
          Beating naive last-price by 15% RMSE on <em>next-day close</em> is not
          realistic. Naive RMSE is roughly one day&apos;s volatility (~1.5–2%),
          and a 15% reduction on that would be a world-class result. If a harness
          reports it, look for a leak before celebrating.
        </Callout>
        <p className="leading-relaxed text-ink-muted">
          Where an edge of that size can legitimately appear: relative directional
          hit rate (50.0% → 57.5% <em>is</em> +15% relative, though only +7.5
          points — conflating the two is misleading), longer horizons, and{" "}
          <strong className="text-ink">volatility rather than direction</strong>.
          That last one is where this project&apos;s measurable result lives.
        </p>
        <p className="leading-relaxed text-ink-muted">
          Directional skill is measured against the <em>base rate</em>, not a coin
          flip. Equities rise on ~54% of days, so a 54% hit rate has added nothing;
          measuring against 50% would credit a model for market drift it never
          predicted.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">The leakage suite</h2>
        <p className="leading-relaxed text-ink-muted">
          Four independent guards, run as their own CI job so a failure is visible
          rather than buried.
        </p>
        <ol className="space-y-3 text-ink-muted">
          {[
            ["Feature causality", "Corrupt all data after a cut point; assert every value before it is byte-identical. Catches centred windows, negative shifts, whole-series statistics."],
            ["Purge correctness", "For every fold and horizon, assert max(train) + horizon < min(test)."],
            ["Shuffled target", "Permute the labels and run the whole pipeline. Skill must collapse to ~0. This does not check one mistake — it checks the conclusion."],
            ["Preprocessing isolation", "Assert scalers are fitted per fold, on training rows only."],
          ].map(([title, body], i) => (
            <li key={title} className="flex gap-3">
              <span className="tnum mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border border-line-strong font-mono text-2xs text-ink-faint">
                {i + 1}
              </span>
              <span className="leading-relaxed">
                <strong className="text-ink">{title}.</strong> {body}
              </span>
            </li>
          ))}
        </ol>
        <p className="leading-relaxed text-ink-muted">
          The shuffled-target test is paired with a positive control on
          planted-signal data, so the suite cannot pass by the pipeline simply
          being broken and predicting nothing.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Known limitations</h2>
        <p className="leading-relaxed text-ink-muted">
          Listed because a methodology section that claims none is not credible.
        </p>
        <ul className="space-y-2 text-ink-muted">
          {[
            "The feature set was chosen once, up front, not re-selected per fold — a mild selection effect at that level cannot be ruled out.",
            "Universes are seeded from present-day index membership, so backtests inherit some survivorship bias.",
            "Single-name only; no cross-sectional or pooled training, which is how the effect is usually exploited in practice.",
            "Volatility is not directly tradable without options. Forecasting a quantity well is not the same as making money from it, and no trading claim is made on the headline result.",
            "Costs are assumptions, not fills. Real slippage depends on size, venue and time of day.",
          ].map((item) => (
            <li key={item} className="flex gap-2.5 leading-relaxed">
              <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-ink-faint" />
              {item}
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}
