# StockForecaster

**A forecasting platform that measures a baseline before it claims anything.**

It forecasts 5-day realised volatility **14.2% better than a random walk**
(R² 0.229, Diebold–Mariano *p* < 0.01, positive on 12 of 12 tickers) — and
reports, with equal prominence, that it finds **no directional price edge at
any horizon**.

That second sentence is the more important one.

---

## The headline numbers

| | |
|---|---|
| **RMSE skill vs naive** | **+14.17%** (5-day realised volatility, LightGBM) |
| R² | 0.229 |
| Symbols with positive skill | **12 / 12** (+9.33% WMT → +22.85% TSLA) |
| Statistical significance | DM *p* < 0.01 on 11 of 12 |
| Conformal P10–P90 coverage | 0.810 against a 0.800 target |
| **Directional price prediction** | **-0.30% skill — worse than predicting zero** |

All of it reproducible: `forecaster evaluate --write-results` regenerates
[`docs/RESULTS.md`](docs/RESULTS.md) from scratch. Nothing in that file is
transcribed by hand.

---

## Why the negative result is the point

The obvious way to build this project is: download history, compute indicators,
split 80/20, train a dozen models, report the best R². Every step after the
indicators is wrong, and the errors all push the same direction — they make
results look better than they are.

This repository's own first commit did all of it. What replaced it:

- **Walk-forward validation** with rolling and anchored windows, so results are
  a distribution across market regimes rather than one number from one arbitrary
  cut point.
- **Purging.** With an *h*-day target, the last *h* training labels are computed
  from bars inside the test window. They get dropped. Every fold re-checks this
  at runtime and raises rather than warns.
- **Seven baselines** that run through the same code path as every other model
  and cannot be filtered out of a leaderboard.
- **Honest model selection** on all folds except the last, so the reported score
  is not the score that was selected on.
- **Significance testing** — Diebold–Mariano, block-bootstrap CIs, PBO, deflated
  Sharpe — because +8% across 20 folds might be ±3% (real) or ±25% (noise).

When you do all of that, technical indicators stop predicting next-day
direction. That is what weak-form market efficiency looks like in data, and it
is [published in full](docs/RESULTS.md#the-negative-result-stated-plainly)
rather than quietly dropped.

---

## The test that keeps it honest

```python
def test_shuffled_target_yields_no_skill() -> None:
    """Permute the labels, then run the entire pipeline.

    This does not check one specific mistake. It checks the *conclusion*.
    If any stage leaks -- a feature peeking ahead, a scaler fitted on
    everything, an off-by-one in the splitter, an index misalignment --
    the model finds the target through the leak and posts positive skill
    on pure noise.
    """
    skill = _run_walk_forward(frame, shuffle_target=True)
    assert skill < 0.02
```

Paired with a positive control on planted-signal data, so the suite cannot pass
by the pipeline simply being broken and predicting nothing.

It runs as its own CI job, so a failure shows on the badge instead of being
buried among a hundred other tests.

---

## Architecture

```
        Next.js 15 (Vercel)          Streamlit (HF Spaces)
        TS · Tailwind · TradingView   thin client, httpx only
                    └──────────┬──────────┘
                          REST + WebSocket
                    ┌──────────▼──────────┐
                    │   FastAPI /api/v1   │   28 endpoints, RFC 7807 errors
                    └──────────┬──────────┘
              ┌────────────────┼────────────────┐
    ingestion · features · validation · models · backtest · explain
              └────────────────┼────────────────┘
        ┌──────────────┬───────┴────────┬──────────────┐
   Postgres        DuckDB + Parquet    Redis      GitHub Actions
   hot: ~800       cold: full 7k       cache      nightly ingest
   tickers + all   ticker universe
   ML state
```

**Why two storage tiers.** The full US equity universe is ~7,000 tickers ×
~1,260 trading days ≈ 8.9M rows, roughly 1.5 GB in Postgres with indexes. Every
free Postgres tier is 0.5 GB. Parquet + zstd stores the same data in ~120 MB at
zero hosting cost, and DuckDB scans it fast enough to serve a request directly.
Symbols are promoted cold → hot on first use.

**Why the clients are thin.** Neither frontend may import the domain package —
[a test enforces it](backend/tests/unit/test_architecture.py). That is what
makes "client-agnostic API" a fact rather than a diagram.

---

## Quickstart

```bash
make install     # venv + backend deps + npm install
make seed        # schema, 6,638 instruments, sp500/nasdaq100/demo universes
make ingest      # 5 years of OHLCV for the demo universe, both tiers
make evaluate    # the full experiment; regenerates docs/RESULTS.md
```

Then, in separate terminals:

```bash
make api         # http://127.0.0.1:8000/docs
make web         # http://localhost:3000
make streamlit   # http://localhost:8501
```

No Docker required — local development runs on SQLite with zero external
services. With Docker, `docker compose up` brings up the production topology
(Postgres, Redis, API, both frontends).

---

## What's inside

| Layer | |
|---|---|
| **Ingestion** | 3 price providers (yfinance, Stooq, Tiingo) behind a failover router with per-provider token buckets and circuit breakers; OHLC coherence enforced by a DB CHECK constraint |
| **Features** | 46 registered features across 7 groups — all scale-free, all causal, each declaring its warm-up requirement |
| **Targets** | forward return, log return, direction, vol-scaled return, triple-barrier, volatility ratio |
| **Models** | 7 baselines · 6 linear · 6 tree · 3 classical (rolling one-step refit) · 4 sequence (LSTM/GRU/TCN/Transformer) |
| **Validation** | walk-forward + purged k-fold, 17 metrics, 4 skill variants, DM test, block bootstrap, PBO, deflated Sharpe |
| **Backtest** | event-driven, next-bar fills, costs on turnover, vol-targeted sizing, buy-and-hold benchmark, cost-sensitivity sweep |
| **Explain** | SHAP (tree / closed-form linear / permutation fallback) + a plain-English glossary for every displayed metric |

**135 tests**, including 42 per-feature causality checks, the shuffled-target
guard, next-bar-execution verification, and 6 architectural constraints.

---

## Three bugs worth reading about

Found while building this, each documented where it lives:

**1. The LSTMs weren't LSTMs.** The prototype reshaped to
`(n_samples, 1, n_features)` — a sequence of length one. With no previous hidden
state the gates reduce to fixed affine transforms and the layer collapses to a
dense layer. Six "deep sequence models", none of which saw a sequence.
→ [`windowing.py`](backend/src/forecaster/models/deep/windowing.py)

**2. The Transformer's positional encoding never trained.** It was applied to a
`tf.range` constant *outside* the functional graph, so Keras never tracked it,
it never entered `trainable_weights`, and it sat at random initialisation for
every run — while the model trained and predicted without error.
→ [`sequence.py`](backend/src/forecaster/models/deep/sequence.py)

**3. The naive baseline scored 0.0% hit rate.** `sign(0)` never matches
`sign(y)`, so a model that correctly declines to bet was recorded as *always
wrong*. Directional accuracy now excludes bars where the prediction is zero, and
skill is measured against the 54% base rate rather than a 50% coin flip.
→ [`metrics.py`](backend/src/forecaster/validation/metrics.py)

---

## Documentation

- [**RESULTS.md**](docs/RESULTS.md) — generated from the harness; the numbers
- [**METHODOLOGY.md**](docs/METHODOLOGY.md) — walk-forward, purging, baselines,
  significance, the leakage suite, and a frank limitations section
- [**RESUME.md**](docs/RESUME.md) — the claims this project supports, and the
  ones it does not

---

## Limitations

Stated because a project that lists none is not credible.

- The feature set was chosen once up front, not re-selected per fold.
- Universes are seeded from present-day index membership, so backtests inherit
  survivorship bias.
- Single-name only; no cross-sectional or pooled training.
- **Volatility is not directly tradable without options.** Forecasting a
  quantity well is not the same as making money from it, and no trading claim is
  made on the headline result.
- Costs are assumptions, not fills.

---

## Licence

MIT. Research and educational use. **Not financial advice.** Past performance
does not predict future results.

---

## The original prototype

The first commit (`c9b677a`) is the 3-file version this replaced: 895 lines of
Streamlit, a single 80/20 split, no baseline, and a "champion" model chosen by
the best R² on the same test set it was scored on. It is preserved in git
history rather than in the working tree — `git show c9b677a` if you want the
before-and-after.

The three bugs above all come from it.
