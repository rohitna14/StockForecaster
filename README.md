<div align="center">

# 📈 StockForecaster

**Stock forecasting that shows its work.**

Search any listed company, get a volatility forecast benchmarked against honest baselines,
and see exactly how much evidence stands behind it, including the results that *didn't* work.

[![CI](https://github.com/rohitna14/StockForecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/rohitna14/StockForecaster/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-200%20passing-2DD97B)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[What it does](#-what-it-does) ·
[Results](#-the-headline-numbers) ·
[Quickstart](#-quickstart) ·
[Architecture](#-architecture) ·
[API](#-api) ·
[Docs](#-documentation)

</div>

---

## 🔍 What it does

StockForecaster is a full-stack research platform for forecasting stocks without fooling yourself.

| | |
|---|---|
| 🔎 **Search anything** | Tickers, full company names, nicknames and typos all work: `aple`, `gogle`, `facebook`, `coca cola`, `sp500`. |
| ⚡ **Any listed stock, on demand** | Search a symbol that hasn't been loaded yet and it is fetched, stored and served in ~1–2 s. You don't need to run ingestion first. |
| 🔮 **Forecasts for every listing** | Decade-old mega caps get full walk-forward models. IPOs that are only a few weeks old still get a forecast from a pooled model trained on *other* companies. Every forecast states its method and confidence. |
| 🏢 **Company pages** | Price chart with indicators, risk metrics, fundamentals (market cap, P/E, EPS, margins, beta, dividend), analyst consensus and headlines. |
| 🏆 **Model leaderboard** | 26 models, from baselines to LSTMs and Transformers, compared fold by fold against a random walk, with significance tests. |
| 💸 **Backtester** | Event-driven, next-bar fills, transaction costs, vol-targeted sizing, and a buy-and-hold benchmark. |
| 🧠 **Explanations** | SHAP feature attributions plus a plain-English glossary for every metric on screen. |
| 🖥️ **Two frontends** | A Next.js web app and a Streamlit app, both thin clients of the same REST API. |

### How a forecast is chosen

Less history means wider error bars, not "no answer". The method is picked from how much data exists:

| Method | History needed | How it works |
|---|---|---|
| `walk_forward` | ≥ 600 bars | Per-stock model, 5+ walk-forward folds |
| `adaptive` | ≥ 150 bars | Per-stock model on shortened windows (≥ 3 folds, 126-bar training floor) |
| `transfer` | ≥ 25 bars | Pooled model trained on *other* companies. Validated leave-one-symbol-out: **+22.8% skill, positive on 12/12 held-out symbols** |
| `none` | < 25 bars | Refuses and explains why |

Confidence reflects how strong the evidence is, not how big the headline number is. A large number from thin data ranks below a modest one from thick data.

---

## 📊 The headline numbers

It forecasts 5-day realised volatility **14.2% better than a random walk**
(R² 0.229, Diebold–Mariano *p* < 0.01, positive on 12 of 12 tickers). It also
reports, just as prominently, that it finds **no directional price edge at any
horizon**.

| Metric | Result |
|---|---|
| **RMSE skill vs naive** | **+14.17%** (5-day realised volatility, LightGBM) |
| R² | 0.229 |
| Symbols with positive skill | **12 / 12** (+9.33% WMT → +22.85% TSLA) |
| Statistical significance | DM *p* < 0.01 on 11 of 12 |
| Conformal P10–P90 coverage | 0.810 against a 0.800 target |
| **Directional price prediction** | **−0.30% skill: worse than predicting zero** |

Every number is reproducible. `forecaster evaluate --write-results` regenerates
[`docs/RESULTS.md`](docs/RESULTS.md) from scratch, and nothing in that file is typed by hand.

---

## 🧪 Why the negative result matters

The obvious way to build this is to download history, compute indicators, split 80/20,
train a dozen models and report the best R². Every step after the indicators is wrong, and
each mistake makes the results look better than they are.

This repository's first commit did all of it. What replaced it:

- **Walk-forward validation** with rolling and anchored windows. Results are a distribution
  across market regimes, not one number from one arbitrary cut point.
- **Purging.** With an *h*-day target, the last *h* training labels come from bars inside
  the test window, so they are dropped. Every fold re-checks this at runtime and raises an error
  if it fails.
- **Seven baselines** that run through the same code path as every other model and cannot
  be filtered out of a leaderboard.
- **Honest model selection**: models are picked on every fold except the last, so the
  reported score is not the one used for selection.
- **Significance testing** (Diebold–Mariano, block-bootstrap CIs, PBO, deflated Sharpe).
  An 8% gain across 20 folds could be ±3% (real) or ±25% (noise).

Once all of that is in place, technical indicators stop predicting next-day direction. That is
what weak-form market efficiency looks like in data, and the result is
[published in full](docs/RESULTS.md#the-negative-result-stated-plainly).

### The test that keeps it honest

```python
def test_shuffled_target_yields_no_skill() -> None:
    """Permute the labels, then run the entire pipeline.

    If any stage leaks -- a feature peeking ahead, a scaler fitted on
    everything, an off-by-one in the splitter, an index misalignment --
    the model finds the target through the leak and posts positive skill
    on pure noise.
    """
    skill = _run_walk_forward(frame, shuffle_target=True)
    assert skill < 0.02
```

It is paired with a positive control on planted-signal data, so the suite can't pass just
because the pipeline is broken and predicts nothing. It also runs as its own CI job.

---

## 🚀 Quickstart

**Prerequisites:** Python 3.11+, Node.js 22+, `make`. Docker is optional.

```bash
git clone https://github.com/rohitna14/StockForecaster.git
cd StockForecaster
cp .env.example .env

make install     # venv + backend deps + npm install
make seed        # schema, ~6,600 instruments, sp500 / nasdaq100 / demo universes
make ingest      # 5 years of OHLCV for the demo universe
```

Then start the services, each in its own terminal:

```bash
make api         # FastAPI      → http://127.0.0.1:8000/docs
make web         # Next.js      → http://localhost:3000
make streamlit   # Streamlit    → http://localhost:8501
```

Local development runs on **SQLite with zero external services**. To run the production
topology (Postgres, Redis, API and both frontends), use:

```bash
docker compose up --build
```

### Useful commands

| Command | What it does |
|---|---|
| `make evaluate` | Runs the full experiment and regenerates `docs/RESULTS.md` |
| `make status` | Shows data coverage across the hot and cold tiers |
| `make test` | Runs the test suite (excluding network and deep-learning tests) |
| `make test-leakage` | Runs only the lookahead-bias guards |
| `make test-deep` | Runs the sequence-model tests (needs TensorFlow) |
| `make lint` / `make fmt` | Ruff lint / auto-format |
| `make help` | Lists every target |

### Configuration

All settings are environment variables prefixed with `FORECASTER_` (see
[`.env.example`](.env.example)). None are required for local use. yfinance and Stooq need no
API key, and any provider without credentials is skipped automatically.

---

## 🏗️ Architecture

```
        Next.js 15 (Vercel)           Streamlit (HF Spaces)
        TS · Tailwind · TradingView   thin client, httpx only
                    └──────────┬──────────┘
                          REST + WebSocket
                    ┌──────────▼──────────┐
                    │   FastAPI /api/v1   │   37 endpoints, RFC 7807 errors
                    └──────────┬──────────┘
              ┌────────────────┼────────────────┐
    ingestion · search · features · validation · models · backtest · explain
              └────────────────┼────────────────┘
        ┌──────────────┬───────┴────────┬──────────────┐
   Postgres        DuckDB + Parquet    Redis      GitHub Actions
   hot: ~800       cold: full ~7k      cache      nightly ingest
   tickers + all   ticker universe
   ML state
```

**Why two storage tiers.** The full US equity universe is ~7,000 tickers × ~1,260 trading
days ≈ 8.9M rows, roughly 1.5 GB in Postgres with indexes. Every free Postgres tier is 0.5 GB.
Parquet + zstd stores the same data in ~120 MB at no hosting cost, and DuckDB scans it fast
enough to serve a request directly. A symbol moves from the cold tier to the hot tier the first time it is used.

**Why the clients are thin.** Neither frontend may import the domain package, and
[a test enforces it](backend/tests/unit/test_architecture.py). Because of that test, the API
really is client-agnostic.

### What's inside

| Layer | |
|---|---|
| **Ingestion** | 3 price providers (yfinance, Stooq, Tiingo) behind a failover router with per-provider token buckets and circuit breakers. On-demand fetching uses per-symbol locks and negative caching. OHLC coherence is enforced by a DB CHECK constraint |
| **Search** | In-memory fuzzy index handling typos, nicknames and partial matches. Company prominence is part of the score, and ties between share classes go to the primary listing |
| **Features** | 46 registered features across 7 groups. All are scale-free and causal, and each declares its warm-up requirement |
| **Targets** | Forward return, log return, direction, vol-scaled return, triple-barrier, volatility ratio |
| **Models** | 7 baselines · 6 linear · 6 tree · 3 classical (rolling one-step refit) · 4 sequence (LSTM / GRU / TCN / Transformer) · pooled cold-start |
| **Validation** | Walk-forward + purged k-fold, adaptive windows, 17 metrics, 4 skill variants, DM test, block bootstrap, PBO, deflated Sharpe, conformal intervals |
| **Backtest** | Event-driven, next-bar fills, costs on turnover, vol-targeted sizing, buy-and-hold benchmark, cost-sensitivity sweep |
| **Explain** | SHAP (tree / closed-form linear / permutation fallback) + a plain-English glossary |

**200 tests**, including per-feature causality checks, the shuffled-target guard,
next-bar-execution verification, 36 search-relevance cases and architectural constraints.

### Project structure

```
StockForecaster/
├── backend/                  Python package `forecaster`
│   ├── src/forecaster/
│   │   ├── api/              FastAPI app, routers, schemas
│   │   ├── ingestion/        providers, failover router, on-demand fetch
│   │   ├── search/           fuzzy index + alias map
│   │   ├── features/         indicators, feature registry, targets
│   │   ├── models/           baselines, linear, trees, classical, deep, pooled
│   │   ├── validation/       splitters, harness, metrics, skill, stats
│   │   ├── backtest/         event-driven engine + cost models
│   │   ├── explain/          SHAP runner + narratives
│   │   ├── db/ · lake/       Postgres/SQLite hot tier · Parquet/DuckDB cold tier
│   │   └── cli.py            `forecaster` command-line interface
│   ├── alembic/              database migrations
│   └── tests/                unit, API and integration tests
├── frontend/                 Next.js 15 + Tailwind web app
├── streamlit_app/            Streamlit client
├── docs/                     RESULTS, METHODOLOGY, RESUME
├── deploy/                   free-tier deployment guide
├── .github/workflows/        CI + nightly ingestion
├── docker-compose.yml
└── Makefile
```

---

## 🔌 API

Interactive docs are served at **`/docs`** (Swagger) and **`/redoc`** once the API is running.
All routes live under `/api/v1`, and errors are returned as RFC 7807 problem+json.

| Area | Example endpoints |
|---|---|
| Search | `GET /search?q=aple` · `GET /resolve?q=google` |
| Instruments | `GET /instruments/{symbol}/summary` · `/ohlcv` · `/indicators` · `/risk` · `/profile` · `/quote` · `/news` |
| Forecasts | `GET /forecast/{symbol}` |
| Experiments | `POST /runs` · `POST /runs/async` · `GET /runs/{id}/metrics` · `/folds` · `/predictions` · `/explain` |
| Jobs | `GET /jobs/{id}` · `WS /ws/jobs/{id}` (live progress) |
| Backtests | `POST /backtests` · `GET /backtests/presets` |
| Catalog | `GET /models` · `GET /features` · `GET /baselines` · `GET /glossary` · `GET /methodology` |
| Health | `GET /health` · `GET /health/ready` |

---

## 🖥️ The web app

| Page | Route |
|---|---|
| Home: search, featured tickers, top movers, headline stats | `/` |
| Explore the universe by sector | `/explore` |
| Stock overview: chart, forecast, fundamentals, news | `/s/{symbol}` |
| Model leaderboard for that stock | `/s/{symbol}/models` |
| Backtest a strategy on that stock | `/s/{symbol}/backtest` |
| How the validation works | `/methodology` |

---

## 🐛 Three bugs worth reading about

Found while building this. Each is documented next to the code it affects:

**1. The LSTMs weren't LSTMs.** The prototype reshaped inputs to `(n_samples, 1, n_features)`,
a sequence of length one. With no previous hidden state the gates reduce to fixed affine
transforms and the layer collapses to a dense layer. There were six "deep sequence models"
and none of them ever saw a sequence.
→ [`windowing.py`](backend/src/forecaster/models/deep/windowing.py)

**2. The Transformer's positional encoding never trained.** It was applied to a `tf.range`
constant *outside* the functional graph, so Keras never tracked it and it stayed at random
initialisation for every run. The model still trained and predicted without any error.
→ [`sequence.py`](backend/src/forecaster/models/deep/sequence.py)

**3. The naive baseline scored a 0.0% hit rate.** `sign(0)` never matches `sign(y)`, so a model
that correctly declines to bet was recorded as *always wrong*. Directional accuracy now
excludes bars where the prediction is zero, and skill is measured against the 54% base rate
rather than a 50% coin flip.
→ [`metrics.py`](backend/src/forecaster/validation/metrics.py)

---

## 📚 Documentation

- [**RESULTS.md**](docs/RESULTS.md): the numbers, generated by the harness
- [**METHODOLOGY.md**](docs/METHODOLOGY.md): walk-forward, purging, baselines, significance,
  the leakage suite, and a frank limitations section
- [**RESUME.md**](docs/RESUME.md): the claims this project supports, and the ones it does not
- [**deploy/README.md**](deploy/README.md): running every component on free tiers
  (Neon, Upstash, Hugging Face Spaces, Vercel, Cloudflare R2, GitHub Actions)

---

## ⚠️ Limitations

A project that lists no limitations isn't credible, so here are this one's:

- The feature set was chosen once up front, not re-selected per fold.
- Universes are seeded from present-day index membership, so backtests inherit survivorship bias.
- Short-history forecasts (`adaptive`, `transfer`) carry less evidence and are labelled as such.
- **Volatility is not directly tradable without options.** Forecasting a quantity well is not the
  same as making money from it, and the headline result makes no trading claim.
- Quotes come from free feeds and are delayed ~15 minutes.
- Transaction costs are assumptions, not real fills.

---

## 👤 Author

**Rohit Nikumbh** · [@rohitna14](https://github.com/rohitna14)

## 📄 License

[MIT](LICENSE). For research and educational use.

> **Not financial advice.** Forecasts are uncertain and past performance does not predict
> future results. Do your own research before making any investment decision.
