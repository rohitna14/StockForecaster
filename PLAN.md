# StockForecaster — Rebuild Plan

> Target: a repository that a senior engineer opens, scrolls for 60 seconds, and thinks
> *"this person knows what they're doing."*
> Every claim on the resume must be reproducible by running one command in this repo.

---

## 0. Audit of the current code (what we are fixing)

| # | Problem | Where | Severity |
|---|---------|-------|----------|
| 1 | **No baseline exists.** The "~15% over naive last-price" claim has no source in code. | nowhere | 🔴 Critical |
| 2 | **No walk-forward validation.** Single 80/20 chronological holdout. | `models.py:66-69` | 🔴 Critical |
| 3 | **Model selection on the test set.** Champion = `max(R²)` over 15 models on the same test set the metrics are reported from. Winner is noise. | `app.py:478` | 🔴 Critical |
| 4 | **LSTMs have 1 timestep.** `reshape((n, 1, n_features))` makes every RNN algebraically a dense layer. No temporal structure in any of the 6 "deep" models. | `models.py:197,222,246,269,318` | 🔴 Critical |
| 5 | **R² on next-day returns + a fake rubric.** UI says "Above 0.7 = Excellent". Real daily-return R² is ~0.00–0.02. | `app.py:509-515` | 🔴 Critical |
| 6 | **ARIMA eval silently dead.** Length guard never true → `predictions = []` → block skipped, error swallowed. | `models.py:387-390` | 🟠 High |
| 7 | **DB stores zero prices.** One table = NASDAQ screener dump, 7,034 rows, `Last Sale` as TEXT `"$141.71"`. Every run re-hits yfinance. | `stocks.db` | 🟠 High |
| 8 | **`SELECT ... FROM {tables[0][0]}`** — first table by arbitrary order, behind a bare `except:`. | `utils.py:19-34` | 🟠 High |
| 9 | **15 models retrained per click, in-process, blocking.** Will time out / OOM on HF free tier (2 vCPU). | `app.py:872` | 🟠 High |
| 10 | **All heavy libs imported at module load** (TF + XGB + LGBM + CatBoost + statsmodels). Cold start ≫ 60s. | `models.py:1-18` | 🟡 Medium |
| 11 | **Errors `print()` to stdout**, invisible in deployment; every trainer is a swallow-all `try/except`. | throughout | 🟡 Medium |
| 12 | **HTML tags inside `st.text()` / `st.info()`** render literally as `<b>`. | `app.py:853-863`, `models.py:97` | 🟡 Medium |
| 13 | RSI uses SMA, not Wilder's smoothing → doesn't match any charting platform. | `utils.py:53-56` | 🟡 Medium |
| 14 | No `requirements.txt`, `.gitignore`, README, tests, LICENSE, CI. `__pycache__/` and `catboost_info/` committed. | repo root | 🟡 Medium |

**Non-issues** (so we don't waste time): the indicators themselves are causal (rolling/ewm look backward), and `StandardScaler` is correctly fit on train only. Those parts are fine.

---

## 1. Target architecture

```
                          ┌──────────────────────────────┐
                          │   Next.js 15 (Vercel)        │  ← the portfolio piece
                          │   TS · Tailwind · shadcn/ui  │
                          │   lightweight-charts · visx  │
                          └──────────────┬───────────────┘
                                         │  REST + WebSocket
    ┌───────────────────┐                │
    │ Streamlit client  │────────────────┤   ← proves API is client-agnostic
    │ (HF Spaces)       │                │      keeps the HF Spaces story alive
    └───────────────────┘                │
                          ┌──────────────▼───────────────┐
                          │      FastAPI  /api/v1        │
                          │  routers · schemas · deps    │
                          └──────┬────────────────┬──────┘
                                 │                │
             ┌───────────────────▼──┐      ┌──────▼─────────────────┐
             │   Domain services    │      │  APScheduler / Actions │
             │ ingestion · features │      │  nightly ingest, retrain│
             │ validation · models  │      └────────────────────────┘
             │ backtest  · explain  │
             └───┬──────────────┬───┘
                 │              │
   ┌─────────────▼───┐   ┌──────▼─────────────┐   ┌──────────────────┐
   │ Postgres (Neon) │   │ DuckDB + Parquet   │   │ Redis (Upstash)  │
   │ HOT tier 0.5 GB │   │ COLD tier, R2/local│   │ cache · quotas   │
   │ ~800 tickers 5y │   │ full 7,034 tickers │   │ job status       │
   │ runs, preds,    │   │ ~9M rows, on-demand│   └──────────────────┘
   │ backtests, news │   └────────────────────┘
   └─────────────────┘
```

### Why two storage tiers
Full US market daily OHLCV = 7,034 tickers × ~1,260 trading days ≈ **8.9M rows ≈ 1.5 GB with indexes** — over every free Postgres tier. So:

- **Hot (Postgres):** curated universe (S&P 500 + NASDAQ 100 + anything a user has ever requested, capped ~800 symbols), plus *all* application state — model runs, fold metrics, predictions, backtests, news, fundamentals. ~100–150 MB. Comfortable.
- **Cold (DuckDB over Hive-partitioned Parquet):** the complete 7,034-ticker history. Zero hosting cost, sub-second scans, and `promote_to_hot(symbol)` migrates a ticker up on first request.

This tiering is *itself* a strong interview talking point. Have the answer ready.

---

## 2. Repository layout

```
stock-forecaster/
├── README.md                      # hero GIF, quickstart, results table, methodology link
├── LICENSE                        # MIT
├── .gitignore                     # __pycache__, catboost_info, *.db, data/, .env, node_modules
├── .env.example
├── docker-compose.yml             # postgres + redis + api + web, one command
├── Makefile                       # make dev / test / lint / ingest / backtest / seed
├── .pre-commit-config.yaml        # ruff, black, mypy, prettier
├── .github/workflows/
│   ├── ci.yml                     # lint → typecheck → test → coverage gate
│   ├── ingest-nightly.yml         # cron: EOD ingestion (free for public repos)
│   └── deploy.yml
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md             # ⭐ walk-forward, purging, embargo, leakage tests
│   ├── DATA_SOURCES.md            # every free API, quota, failover order
│   ├── RESULTS.md                 # auto-generated from the eval harness
│   ├── API.md
│   └── adr/                       # architecture decision records 0001..N
│
├── backend/
│   ├── pyproject.toml             # uv/poetry; ruff + mypy + pytest config
│   ├── Dockerfile                 # multi-stage, slim, CPU-only torch/TF
│   ├── alembic.ini
│   ├── alembic/versions/
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/
│   │   │   ├── test_indicators.py       # golden values vs TA-Lib references
│   │   │   ├── test_splitters.py        # ⭐ no train/test overlap, embargo honored
│   │   │   ├── test_leakage.py          # ⭐ future-corruption invariance
│   │   │   ├── test_baselines.py
│   │   │   └── test_metrics.py
│   │   ├── integration/
│   │   │   ├── test_ingestion_failover.py
│   │   │   ├── test_repositories.py
│   │   │   └── test_backtest_engine.py
│   │   └── api/test_routes.py           # + schemathesis contract fuzzing
│   └── src/forecaster/
│       ├── config.py               # pydantic-settings, typed env
│       ├── logging.py              # structlog JSON, request-id middleware
│       ├── exceptions.py           # domain error hierarchy
│       │
│       ├── db/
│       │   ├── session.py          # async engine, session factory
│       │   ├── models.py           # SQLAlchemy 2.0 ORM (typed, Mapped[])
│       │   └── repositories/
│       │       ├── instruments.py
│       │       ├── ohlcv.py
│       │       ├── runs.py
│       │       ├── predictions.py
│       │       └── backtests.py
│       │
│       ├── lake/                   # cold tier
│       │   ├── writer.py           # df → partitioned parquet
│       │   ├── query.py            # DuckDB session, pushdown predicates
│       │   └── promote.py          # cold → hot migration
│       │
│       ├── ingestion/
│       │   ├── base.py             # Protocol: PriceProvider, NewsProvider, ...
│       │   ├── ratelimit.py        # token bucket, Redis-backed
│       │   ├── router.py           # failover chain + circuit breaker
│       │   ├── corporate_actions.py# split/dividend back-adjustment
│       │   ├── validators.py       # OHLC sanity, gap detection, outliers
│       │   └── providers/
│       │       ├── yfinance_provider.py
│       │       ├── stooq_provider.py
│       │       ├── alphavantage_provider.py
│       │       ├── finnhub_provider.py
│       │       ├── tiingo_provider.py
│       │       ├── fred_provider.py        # macro
│       │       ├── edgar_provider.py       # SEC XBRL fundamentals
│       │       └── news_provider.py
│       │
│       ├── features/
│       │   ├── registry.py         # @feature decorator, name → fn, dependency DAG
│       │   ├── indicators/
│       │   │   ├── trend.py        # SMA, EMA, MACD, ADX, Ichimoku, Aroon
│       │   │   ├── momentum.py     # RSI (Wilder), Stoch, CCI, ROC, Williams %R
│       │   │   ├── volatility.py   # ATR, Bollinger, Keltner, Donchian, Parkinson
│       │   │   ├── volume.py       # OBV, VWAP, MFI, A/D, Chaikin, volume z-score
│       │   │   └── statistical.py  # Hurst, realized vol, skew/kurt, autocorr
│       │   ├── calendar.py         # day-of-week, month-end, earnings proximity, holidays
│       │   ├── macro.py            # VIX, 10Y, 2s10s, DXY, CPI surprise (as-of joined)
│       │   ├── sentiment.py        # FinBERT over headlines, decayed aggregate
│       │   ├── cross_sectional.py  # sector-relative strength, market beta
│       │   ├── targets.py          # h-day fwd return, vol-scaled, direction, triple-barrier
│       │   └── pipeline.py         # sklearn Pipeline, fit inside each fold only
│       │
│       ├── models/
│       │   ├── base.py             # Forecaster ABC: fit/predict/save/load/params
│       │   ├── baselines.py        # ⭐ NaiveLastPrice, Drift, SeasonalNaive, EWMA, ZeroReturn
│       │   ├── linear.py           # Ridge, ElasticNet, LogisticRegression, HuberRegressor
│       │   ├── trees.py            # LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
│       │   ├── classical.py        # ARIMA/SARIMAX/ETS with rolling one-step refit
│       │   ├── deep/
│       │   │   ├── windowing.py    # ⭐ (n, LOOKBACK, n_features) — the actual fix
│       │   │   ├── lstm.py         # stacked LSTM, seq_len=60
│       │   │   ├── gru.py
│       │   │   ├── tcn.py          # dilated causal convs
│       │   │   ├── transformer.py  # temporal attention encoder
│       │   │   └── nbeats.py       # optional
│       │   ├── ensemble.py         # stacking w/ out-of-fold meta-features, rank-average
│       │   ├── conformal.py        # ⭐ split-conformal prediction intervals
│       │   └── registry.py         # name → class, versioned hyperparam specs
│       │
│       ├── validation/
│       │   ├── splitters.py        # ⭐ WalkForwardSplit, PurgedKFold, CombinatorialPurgedCV
│       │   ├── metrics.py          # regression · directional · probabilistic · trading
│       │   ├── skill.py            # ⭐ skill score vs baseline — the "15%" number
│       │   ├── stats.py            # Diebold-Mariano, block bootstrap CI, PBO, deflated Sharpe
│       │   ├── harness.py          # orchestrates: universe × models × horizons × folds
│       │   └── report.py           # → docs/RESULTS.md + JSON
│       │
│       ├── backtest/
│       │   ├── engine.py           # event-driven, next-bar fills, no lookahead
│       │   ├── costs.py            # commission + spread + slippage + borrow
│       │   ├── signals.py          # prediction → position sizing (vol targeting, Kelly-capped)
│       │   ├── portfolio.py        # positions, cash, equity curve
│       │   └── report.py           # tearsheet stats
│       │
│       ├── explain/
│       │   ├── shap_runner.py      # global + per-prediction attribution
│       │   └── narrative.py        # plain-English "why" strings for the UI
│       │
│       ├── jobs/
│       │   ├── scheduler.py        # APScheduler
│       │   ├── queue.py            # in-proc worker + status in Redis
│       │   └── tasks/
│       │       ├── ingest_eod.py
│       │       ├── refresh_features.py
│       │       ├── retrain.py
│       │       └── score_universe.py
│       │
│       ├── api/
│       │   ├── main.py             # app factory, CORS, middleware, lifespan
│       │   ├── deps.py             # DI: session, repos, current_settings
│       │   ├── errors.py           # RFC 7807 problem+json handlers
│       │   ├── schemas/            # Pydantic v2 request/response models
│       │   └── routers/
│       │       ├── health.py  instruments.py  ohlcv.py  features.py
│       │       ├── runs.py    predictions.py  backtests.py
│       │       ├── models.py  screener.py     ws.py
│       └── cli.py                  # typer: ingest, train, evaluate, backtest, seed, promote
│
├── frontend/                       # Next.js 15, App Router
│   ├── package.json  tsconfig.json  tailwind.config.ts  next.config.ts
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx  page.tsx           # landing
│   │   │   ├── s/[symbol]/
│   │   │   │   ├── page.tsx                   # overview
│   │   │   │   ├── models/page.tsx
│   │   │   │   ├── backtest/page.tsx
│   │   │   │   └── loading.tsx
│   │   │   ├── compare/page.tsx
│   │   │   ├── screener/page.tsx
│   │   │   └── methodology/page.tsx
│   │   ├── components/
│   │   │   ├── ui/                            # shadcn primitives
│   │   │   ├── charts/
│   │   │   │   ├── PriceChart.tsx             # lightweight-charts, candles + overlays
│   │   │   │   ├── ForecastFan.tsx            # ⭐ conformal P10/P50/P90 cone
│   │   │   │   ├── EquityCurve.tsx
│   │   │   │   ├── DrawdownChart.tsx
│   │   │   │   ├── ModelLeaderboard.tsx
│   │   │   │   ├── ShapWaterfall.tsx
│   │   │   │   └── CalibrationPlot.tsx
│   │   │   ├── CommandPalette.tsx             # ⌘K ticker search
│   │   │   ├── MetricTile.tsx                 # value + delta + info tooltip
│   │   │   ├── ExplainToggle.tsx              # ⭐ plain-English mode
│   │   │   └── RunProgress.tsx                # live WebSocket training progress
│   │   ├── lib/  api.ts  types.ts (generated from OpenAPI)  format.ts
│   │   └── hooks/  useOhlcv.ts  useRun.ts  useRunSocket.ts
│   └── e2e/                                   # Playwright
│
├── streamlit_app/                  # thin client — ZERO business logic
│   ├── Dockerfile
│   ├── requirements.txt            # streamlit, httpx, plotly ONLY
│   ├── app.py
│   ├── api_client.py               # typed httpx wrapper
│   ├── theme/.streamlit/config.toml
│   └── pages/  1_📈_Forecast.py  2_🧪_Models.py  3_🎯_Backtest.py  4_📚_Methodology.py
│
└── data/                           # gitignored except tiny samples
    ├── lake/ohlcv/symbol=AAPL/year=2024/part-0.parquet
    └── samples/                    # fixtures for tests + CI
```

**Rule that keeps this honest:** `streamlit_app/` and `frontend/` may not import from `forecaster/`. They only speak HTTP. If a feature needs logic, it goes in the backend and gets an endpoint.

---

## 3. Database schema (Postgres, hot tier)

```sql
-- ─────────────── reference data ───────────────
CREATE TABLE instruments (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT        NOT NULL UNIQUE,
    name            TEXT        NOT NULL,
    exchange        TEXT,
    asset_type      TEXT        NOT NULL DEFAULT 'equity',
    sector          TEXT,
    industry        TEXT,
    country         TEXT,
    currency        TEXT        NOT NULL DEFAULT 'USD',
    market_cap      NUMERIC(20,2),
    ipo_year        SMALLINT,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    tier            TEXT        NOT NULL DEFAULT 'cold',  -- 'hot' | 'cold'
    first_date      DATE,
    last_date       DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_instruments_search ON instruments
    USING gin (to_tsvector('english', symbol || ' ' || name));
CREATE INDEX idx_instruments_tier   ON instruments (tier) WHERE is_active;

-- ─────────────── prices ───────────────
CREATE TABLE ohlcv_daily (
    instrument_id   BIGINT      NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    ts              DATE        NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    adj_close       DOUBLE PRECISION NOT NULL,   -- split+dividend adjusted
    volume          BIGINT      NOT NULL,
    source          TEXT        NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (instrument_id, ts),
    CONSTRAINT ohlc_sane CHECK (high >= low AND high >= open AND high >= close
                                AND low <= open AND low <= close AND volume >= 0)
);
CREATE INDEX idx_ohlcv_ts ON ohlcv_daily (ts DESC);
-- Optional on Timescale: SELECT create_hypertable('ohlcv_daily','ts', chunk_time_interval => INTERVAL '1 year');

CREATE TABLE ohlcv_intraday (      -- 5m bars, last 60 days only, pruned nightly
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION,
    low DOUBLE PRECISION, close DOUBLE PRECISION, volume BIGINT,
    PRIMARY KEY (instrument_id, ts)
);

CREATE TABLE corporate_actions (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    ex_date DATE NOT NULL,
    action_type TEXT NOT NULL,          -- 'split' | 'dividend'
    ratio NUMERIC(12,6),
    amount NUMERIC(12,6),
    UNIQUE (instrument_id, ex_date, action_type)
);

-- ─────────────── fundamentals & macro & news ───────────────
CREATE TABLE fundamentals (
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    fiscal_period TEXT NOT NULL,        -- '2024Q3'
    report_date   DATE NOT NULL,        -- ⚠️ as-of date; NEVER join on fiscal_period
    revenue NUMERIC, net_income NUMERIC, eps_diluted NUMERIC,
    total_assets NUMERIC, total_debt NUMERIC, free_cash_flow NUMERIC,
    shares_outstanding NUMERIC, source TEXT NOT NULL,
    PRIMARY KEY (instrument_id, fiscal_period)
);

CREATE TABLE macro_series (
    code TEXT PRIMARY KEY,              -- 'DGS10','VIXCLS','T10Y2Y','CPIAUCSL','UNRATE','DFF'
    name TEXT NOT NULL, frequency TEXT, units TEXT, source TEXT NOT NULL DEFAULT 'FRED'
);
CREATE TABLE macro_observations (
    code TEXT NOT NULL REFERENCES macro_series(code) ON DELETE CASCADE,
    ts DATE NOT NULL, value DOUBLE PRECISION,
    PRIMARY KEY (code, ts)
);

CREATE TABLE news_articles (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT REFERENCES instruments(id) ON DELETE CASCADE,
    published_at TIMESTAMPTZ NOT NULL,
    headline TEXT NOT NULL, url TEXT UNIQUE, source TEXT,
    sentiment_label TEXT,               -- FinBERT
    sentiment_score DOUBLE PRECISION,
    scored_at TIMESTAMPTZ
);
CREATE INDEX idx_news_lookup ON news_articles (instrument_id, published_at DESC);

-- ─────────────── ML lifecycle ───────────────
CREATE TABLE feature_sets (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL, version TEXT NOT NULL,
    spec JSONB NOT NULL,                -- feature names + params, fully reproducible
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE model_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id  BIGINT REFERENCES instruments(id) ON DELETE CASCADE,
    universe_id    BIGINT REFERENCES universes(id),
    model_name     TEXT NOT NULL,       -- 'lightgbm' | 'lstm' | 'naive_last_price' ...
    model_version  TEXT NOT NULL,
    feature_set_id BIGINT REFERENCES feature_sets(id),
    horizon_days   SMALLINT NOT NULL,
    target_type    TEXT NOT NULL,       -- 'return' | 'direction' | 'vol_scaled_return'
    hyperparams    JSONB NOT NULL,
    split_config   JSONB NOT NULL,      -- ⭐ train/test window, step, purge, embargo
    status         TEXT NOT NULL DEFAULT 'queued',
    progress       REAL NOT NULL DEFAULT 0,
    error_message  TEXT,
    git_sha        TEXT,                -- ⭐ reproducibility
    random_seed    INT,
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    duration_seconds REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_runs_lookup ON model_runs (instrument_id, model_name, horizon_days, created_at DESC);

CREATE TABLE fold_metrics (             -- one row per walk-forward fold
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    fold_index SMALLINT NOT NULL,
    train_start DATE, train_end DATE, test_start DATE, test_end DATE,
    n_train INT, n_test INT,
    mae DOUBLE PRECISION, rmse DOUBLE PRECISION, mase DOUBLE PRECISION,
    r2 DOUBLE PRECISION, directional_accuracy DOUBLE PRECISION,
    mcc DOUBLE PRECISION, roc_auc DOUBLE PRECISION, brier DOUBLE PRECISION,
    skill_vs_naive DOUBLE PRECISION,    -- ⭐ 1 - MSE_model/MSE_baseline
    sharpe DOUBLE PRECISION,
    UNIQUE (run_id, fold_index)
);

CREATE TABLE run_metrics (              -- aggregate across folds + CIs
    run_id UUID PRIMARY KEY REFERENCES model_runs(id) ON DELETE CASCADE,
    metrics JSONB NOT NULL,             -- {mae:{mean,std,ci_low,ci_high}, ...}
    skill_vs_naive DOUBLE PRECISION,
    dm_stat DOUBLE PRECISION,           -- Diebold-Mariano vs baseline
    dm_pvalue DOUBLE PRECISION,
    n_folds SMALLINT
);

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    as_of_date  DATE NOT NULL,          -- info available up to here
    target_date DATE NOT NULL,          -- what we're predicting
    fold_index SMALLINT,
    is_out_of_sample BOOLEAN NOT NULL DEFAULT TRUE,
    y_pred DOUBLE PRECISION NOT NULL,
    y_pred_lower DOUBLE PRECISION,      -- conformal P10
    y_pred_upper DOUBLE PRECISION,      -- conformal P90
    prob_up DOUBLE PRECISION,
    y_true DOUBLE PRECISION,            -- NULL until realized
    UNIQUE (run_id, instrument_id, as_of_date, target_date)
);
CREATE INDEX idx_pred_lookup ON predictions (instrument_id, target_date DESC);

CREATE TABLE feature_importance (
    run_id UUID NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    importance DOUBLE PRECISION NOT NULL,
    method TEXT NOT NULL,               -- 'shap' | 'permutation' | 'gain'
    PRIMARY KEY (run_id, feature_name, method)
);

-- ─────────────── backtesting ───────────────
CREATE TABLE backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_run_id UUID REFERENCES model_runs(id) ON DELETE CASCADE,
    strategy_name TEXT NOT NULL,
    config JSONB NOT NULL,              -- costs, sizing, rebalance, constraints
    start_date DATE, end_date DATE,
    initial_capital NUMERIC(18,2),
    total_return DOUBLE PRECISION, cagr DOUBLE PRECISION,
    sharpe DOUBLE PRECISION, sortino DOUBLE PRECISION, calmar DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION, volatility DOUBLE PRECISION,
    win_rate DOUBLE PRECISION, profit_factor DOUBLE PRECISION,
    n_trades INT, turnover DOUBLE PRECISION, total_costs NUMERIC(18,2),
    benchmark_return DOUBLE PRECISION,  -- ⭐ always report buy & hold too
    alpha DOUBLE PRECISION, beta DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE backtest_equity (
    backtest_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    ts DATE NOT NULL, equity NUMERIC(18,2), cash NUMERIC(18,2),
    position DOUBLE PRECISION, drawdown DOUBLE PRECISION,
    benchmark_equity NUMERIC(18,2),
    PRIMARY KEY (backtest_id, ts)
);
CREATE TABLE backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    backtest_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    entry_date DATE, exit_date DATE, side TEXT,
    entry_price DOUBLE PRECISION, exit_price DOUBLE PRECISION,
    quantity DOUBLE PRECISION, pnl NUMERIC(18,2), pnl_pct DOUBLE PRECISION,
    costs NUMERIC(18,2), exit_reason TEXT
);

-- ─────────────── ops ───────────────
CREATE TABLE universes (
    id BIGSERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, description TEXT
);
CREATE TABLE universe_members (
    universe_id BIGINT REFERENCES universes(id) ON DELETE CASCADE,
    instrument_id BIGINT REFERENCES instruments(id) ON DELETE CASCADE,
    added_at DATE NOT NULL DEFAULT CURRENT_DATE, removed_at DATE,
    PRIMARY KEY (universe_id, instrument_id, added_at)
);
CREATE TABLE ingestion_log (
    id BIGSERIAL PRIMARY KEY, provider TEXT NOT NULL, symbol TEXT,
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    rows_written INT, status TEXT, error TEXT
);
CREATE TABLE api_quota (
    provider TEXT PRIMARY KEY, calls_used INT NOT NULL DEFAULT 0,
    window_start TIMESTAMPTZ NOT NULL, limit_per_window INT NOT NULL
);
```

**Migration of your existing `stocks.db`:** one-off script parses `Last Sale` (`"$141.71"` → numeric), `% Change` (`"0.049%"` → float), maps NASDAQ screener columns → `instruments`, and drops the price snapshot columns entirely (they're a stale point-in-time artifact, not time series). ~7,034 instruments seeded, all `tier='cold'`.

---

## 4. Data sources — all free

| Provider | Key? | Quota | Use for | Order |
|---|---|---|---|---|
| **yfinance** | No | Unofficial, soft-throttled | Daily + intraday OHLCV, splits, dividends | Primary |
| **Stooq** | No | Generous CSV endpoint | EOD OHLCV | Fallback 1 |
| **Tiingo** | Yes | ~1,000/day | EOD OHLCV, high quality | Fallback 2 |
| **Alpha Vantage** | Yes | 25/day | EOD, indicators | Fallback 3 |
| **Finnhub** | Yes | 60/min | Quotes, profiles, earnings calendar, news | Metadata + news |
| **FMP** | Yes | 250/day | Fundamentals, ratios | Fundamentals |
| **SEC EDGAR XBRL** | No (User-Agent required) | ~10 req/s | Authoritative fundamentals + **true filing dates** | Fundamentals ⭐ |
| **FRED** | Yes (free) | Unlimited | VIX, DGS10, T10Y2Y, CPI, UNRATE, DFF | Macro |
| **GDELT / RSS** | No | Free | Headlines | News |
| **HF Inference (ProsusAI/finbert)** | Yes (free) | Rate-limited | Headline sentiment | Sentiment |
| **pandas_market_calendars** | No | Local pkg | Trading days, holidays, early closes | Calendar |

**Failover router:** each provider implements `PriceProvider`. `router.py` tries in order, respects a Redis token bucket per provider, opens a circuit breaker after N consecutive failures, and logs every attempt to `ingestion_log`. Cross-validate: when two providers both return a bar, assert close prices agree within 0.5% or flag it.

⚠️ **EDGAR matters more than it looks.** It gives the *actual filing date*, so fundamental features can be joined as-of correctly instead of leaking a quarter's numbers backward into the quarter itself. That's a leakage bug most portfolio projects have. Not having it is a differentiator.

---

## 5. The ML core (this is what the resume rests on)

### 5.1 Walk-forward validation

```python
@dataclass(frozen=True)
class WalkForwardSplit:
    train_size: int          # bars, or None for anchored/expanding
    test_size: int           # bars per fold
    step: int                # bars to advance
    horizon: int             # prediction horizon h
    embargo: int = 0         # extra bars dropped after test
    mode: Literal["rolling", "anchored"] = "rolling"

    def split(self, index: pd.DatetimeIndex) -> Iterator[Fold]:
        """Yields (train_idx, test_idx) with a PURGE GAP of `horizon`
        bars between train_end and test_start, so no training label
        depends on a bar that appears in the test window."""
```

Two mandatory ideas:
- **Purging** — with an h-day forward target, the last `h` training labels are built from bars inside the test window. Drop them. Without this you leak, no matter how clean the split looks.
- **Embargo** — drop `e` bars after each test fold before the next train window, so serially-correlated information doesn't bleed backward.

Everything fold-local: scaler, imputer, feature selection, hyperparameter search, and **model selection**. `fit()` never sees test data. Ever.

### 5.2 Baselines — the thing that's currently missing entirely

```python
class NaiveLastPrice(Forecaster):      # ŷ_{t+h} = y_t  (⇒ predicted return = 0)
class DriftBaseline(Forecaster):       # linear extrapolation of recent slope
class SeasonalNaive(Forecaster):       # ŷ_{t+h} = y_{t+h-252}
class EWMABaseline(Forecaster):        # exponentially weighted mean return
class HistoricalMeanReturn(Forecaster) # in-sample mean drift
class CoinFlip(Forecaster):            # 50/50 direction, for hit-rate comparison
```

Baselines are **first-class models** — same `Forecaster` interface, same walk-forward harness, rows in `model_runs`. They appear on the leaderboard. The UI never hides them.

### 5.3 The skill score — where "~15%" comes from

```python
def skill_score(y_true, y_pred_model, y_pred_baseline, metric=mse) -> float:
    """1 - metric(model)/metric(baseline). Positive = model beats baseline.
    Reported per-fold, then aggregated with a block bootstrap CI."""
```

**Read this carefully — it's the difference between a bullet you can defend and one that ends the interview:**

Beating naive last-price by **15% RMSE on next-day close** is *not realistic*. Naive RMSE ≈ one day's volatility (~1.5–2%). A 15% reduction on that is renaissance-fund territory. If your harness reports it, you have a leak — go find it.

Where a real ~15% edge legitimately shows up:

1. **Directional accuracy, relative** — 50.0% → 57.5% hit rate is *+15% relative*. Achievable, honest, and precisely stateable.
2. **RMSE skill at longer horizons** — h=5 or h=10 day returns are more predictable than h=1. Skill scores of 5–15% are plausible here.
3. **MASE reduction** vs seasonal-naive.
4. **Risk-adjusted, after costs** — Sharpe of the model-driven strategy vs buy-and-hold.

**The plan is: run the harness, take whatever number is real, and state it with the metric, horizon, universe, fold count, and confidence interval attached.** `docs/RESULTS.md` gets auto-generated from the harness so the number in your resume is always the number the code produces. If it lands at 8%, the bullet says 8% — a defensible 8% beats an indefensible 15% every single time.

Also ship **Diebold-Mariano** vs the baseline. Being able to say *"and it's statistically significant at p=0.03 by DM test"* is a differentiator almost no portfolio project has.

### 5.4 Fixing the deep models

```python
def make_windows(X, y, lookback=60, horizon=1):
    """(n_samples, lookback, n_features) — real sequences, not (n, 1, f)."""
```

Then: stacked LSTM (`seq_len=60`), GRU, **TCN** (dilated causal convs — cheap, strong, and shows you know it), and a small **temporal Transformer** encoder. Train with early stopping on a fold-internal validation slice, `EarlyStopping(patience=10, restore_best_weights=True)`, and honor the "TensorFlow" claim on the resume.

Cost control: deep models train on-demand for a single ticker, or offline nightly for the hot universe — never synchronously in a request.

### 5.5 Metrics, honestly

| Class | Metrics |
|---|---|
| Regression | MAE, RMSE, **MASE**, sMAPE, R² *(shown with a caveat, never as the headline)* |
| Directional | Hit rate, precision/recall on up-moves, **MCC**, ROC-AUC |
| Probabilistic | Brier score, **calibration curve**, conformal interval coverage (does P10–P90 actually contain 80%?) |
| Trading | Sharpe, Sortino, Calmar, max DD, turnover, profit factor, **all after costs** |
| Statistical | **Diebold-Mariano** vs baseline, block-bootstrap CIs, **PBO**, deflated Sharpe |

Delete the "R² > 0.7 = Excellent" rubric. Replace with a methodology panel that says the quiet part out loud: *"Next-day return R² is near zero for everyone, including hedge funds. We report directional accuracy and cost-adjusted Sharpe because those are the metrics that survive contact with reality."* That single paragraph will impress more people than the entire current UI.

### 5.6 Leakage tests (make these loud in the README)

```python
def test_features_are_causal():
    """Corrupt all data after T; assert every feature value at t < T is byte-identical."""

def test_no_train_test_overlap_with_purge():
    """For every fold: max(train_idx) + horizon < min(test_idx)."""

def test_shuffled_target_yields_no_skill():
    """Train on shuffled y. Skill vs baseline must be ≈ 0 (within CI).
    If it isn't, the pipeline leaks."""

def test_scaler_never_sees_test():
    """Assert scaler.fit called exactly once per fold, on train rows only."""
```

That third test is the strongest signal in the whole repo. Put it in the README.

---

## 6. API surface (`/api/v1`)

```
GET    /health                              liveness
GET    /health/ready                        db + redis + provider checks

GET    /instruments?q=&sector=&tier=&limit=  typeahead search (⌘K)
GET    /instruments/{symbol}                 profile + coverage window
GET    /instruments/{symbol}/ohlcv           ?start&end&interval&adjusted (auto-promotes cold→hot)
GET    /instruments/{symbol}/indicators      ?set=trend,momentum,volatility
GET    /instruments/{symbol}/fundamentals
GET    /instruments/{symbol}/news            + FinBERT sentiment
GET    /instruments/{symbol}/risk            vol, Sharpe, Sortino, DD, VaR, CVaR

GET    /models                               registry catalog + default hyperparams
POST   /runs                                 {symbol, models[], horizon, split_config} → 202 {run_id}
GET    /runs/{id}                            status, progress, timings
GET    /runs/{id}/metrics                    per-fold + aggregate + skill + DM test
GET    /runs/{id}/predictions                ?fold=&oos_only=true
GET    /runs/{id}/explain                    SHAP global + per-date + narrative
GET    /runs/{id}/leaderboard                all models incl. baselines, ranked
DELETE /runs/{id}

POST   /backtests                            {run_id, strategy, costs, sizing} → 202
GET    /backtests/{id}                       stats + equity + drawdown + trades
GET    /backtests/{id}/tearsheet             full report payload

GET    /screener                             ?min_sharpe&sector&signal — ranked latest predictions
GET    /universes  /universes/{name}/members

WS     /ws/runs/{id}                         live fold-by-fold training progress
```

Conventions: Pydantic v2 schemas, RFC 7807 `problem+json` errors, cursor pagination, `ETag`/`Cache-Control` on OHLCV, `X-Request-ID` on everything, OpenAPI 3.1 → auto-generated TypeScript client for the frontend (`openapi-typescript`). Rate limited via slowapi.

---

## 7. Frontend design

### Stack
Next.js 15 (App Router, RSC) · TypeScript strict · Tailwind · shadcn/ui · TanStack Query · **lightweight-charts** (TradingView's own lib — the reason it looks like a real terminal) · visx for metric charts · Framer Motion · next-themes.

### Design system
- **Dark-first**, near-black `#0A0B0D` canvas, elevated `#141619` surfaces, one accent (electric indigo).
- **Tabular monospace numerals** for every price/metric. Non-negotiable — proportional digits make numbers jitter and instantly look amateur.
- Green/red **only** for P&L semantics, always paired with ▲/▼ glyphs (colorblind-safe).
- Skeleton loaders, never spinners. Optimistic UI on ticker switch.
- Motion is functional only: chart crosshair, number roll-ups, fold-progress. No `hue-rotate` animation on the page background.

### Pages
| Route | Content |
|---|---|
| `/` | Hero with live animated forecast fan, "try AAPL/NVDA/TSLA" chips, methodology teaser |
| `/s/[symbol]` | Price chart + overlays, **forecast fan (P10/P50/P90)**, metric tiles, risk panel, news sentiment strip |
| `/s/[symbol]/models` | Leaderboard **including baselines**, per-fold metric small-multiples, SHAP waterfall, calibration plot, live training via WebSocket |
| `/s/[symbol]/backtest` | Equity curve vs buy-and-hold, drawdown underwater chart, trade table, cost sensitivity slider |
| `/compare` | 2–4 tickers side by side |
| `/screener` | Sortable table of latest predictions across the hot universe |
| `/methodology` | ⭐ Walk-forward diagram, purge/embargo animation, leakage tests, "why R² is a bad metric here" |

### The "friendly vibe"
- **`ExplainToggle`** — a global switch that swaps every metric label for plain English. "Sortino Ratio 1.42" ⇄ "Rewards you 1.42× for each unit of *downside* risk — above 1.0 is solid."
- Info tooltip on **every** metric, no exceptions.
- Empty/error states with personality and a next action ("No data for ZZZZ yet — want us to fetch it?" + button).
- `⌘K` command palette for ticker search.
- Onboarding tour on first visit (3 steps, dismissible, remembered).
- Persistent, honest disclaimer banner. Confidence is friendly; overclaiming is not.

**Do not port the current CSS.** The `@keyframes gradientShift { filter: hue-rotate() }` on the page background, 80px glows, and animated gold "champion" card read as unserious. Restraint is what looks expensive.

---

## 8. Jobs & scheduling

| Job | Cadence | Runs on |
|---|---|---|
| `ingest_eod` | Weekdays 21:30 ET | GitHub Actions cron (free for public repos) |
| `refresh_features` | After ingest | GH Actions |
| `retrain_hot_universe` | Weekly, Sunday | GH Actions (matrix over model types) |
| `score_universe` | Daily after features | GH Actions → writes `predictions` |
| `realize_predictions` | Daily | Backfills `y_true` on matured predictions ⭐ |
| `prune_intraday` | Daily | APScheduler in API |
| `refresh_news_sentiment` | 4×/day | APScheduler |

⭐ `realize_predictions` enables a **live out-of-sample track record** — predictions logged before the fact, scored after. That is dramatically more convincing than any backtest, and almost nobody's portfolio project has it. Surface it on the landing page: *"Live OOS hit rate since 2026-08: 54.2% over 1,340 predictions."*

---

## 9. Testing, CI, quality gates

- **pytest** + `pytest-cov`, gate at 80% on `forecaster/` core (validation, features, models, backtest — the parts that matter).
- **Hypothesis** property tests on indicators: RSI ∈ [0,100]; SMA(k) of a constant series = that constant; ATR ≥ 0; Bollinger upper ≥ middle ≥ lower.
- **Golden-file tests**: indicator outputs pinned against known-good reference values.
- **Leakage suite** (§5.6) — its own CI job, named `leakage` so it's visible on the badge.
- **schemathesis** contract fuzzing against the OpenAPI spec.
- **Playwright** e2e: search → select → forecast renders → leaderboard shows a baseline row.
- **ruff** + **black** + **mypy --strict** on `forecaster/`; **prettier** + **eslint** on frontend; all via pre-commit.
- CI: lint → typecheck → unit → integration (Postgres service container) → e2e → coverage badge.

---

## 10. Deployment (every tier free)

| Component | Host | Free tier |
|---|---|---|
| Postgres | **Neon** | 0.5 GB, autosuspend |
| Redis | **Upstash** | 10k cmd/day |
| API | **HF Spaces (Docker)** or **Fly.io** | 2 vCPU / 16 GB |
| Next.js | **Vercel** | Hobby |
| Streamlit | **HF Spaces** | keeps the existing story |
| Parquet lake | **Cloudflare R2** | 10 GB + free egress |
| Jobs | **GitHub Actions** | free for public repos |
| Model artifacts | R2 via joblib, MLflow-style manifest in Postgres |

`docker-compose.yml` brings the whole stack up locally in one command. That plus a seeded demo dataset means a reviewer can run your project in 60 seconds — which is the single highest-ROI thing in this document.

---

## 11. Roadmap

### Week 1 — Foundations & data
- [ ] Monorepo scaffold, `.gitignore`, `pyproject.toml`, pre-commit, CI skeleton, MIT license
- [ ] **Purge `__pycache__/`, `catboost_info/`, `stocks.db` from git** (`git rm -r --cached`)
- [ ] `config.py`, `logging.py`, exception hierarchy
- [ ] SQLAlchemy models + Alembic initial migration; docker-compose (postgres + redis)
- [ ] Migrate `stocks.db` → `instruments` (parse `"$141.71"`, drop stale snapshot cols)
- [ ] `PriceProvider` protocol + yfinance, Stooq, Tiingo; router with token bucket + circuit breaker
- [ ] Parquet lake writer + DuckDB reader; ingest 5y for S&P 500 → hot, full universe → cold
- [ ] `forecaster ingest` CLI; validators (OHLC sanity, gaps, outliers)
- **Exit:** `make seed` populates a real database from scratch.

### Week 2 — Features & validation ⭐ the crown jewel
- [ ] Feature registry + indicator modules (Wilder's RSI this time)
- [ ] Target builders: h-day return, direction, vol-scaled, triple-barrier
- [ ] `WalkForwardSplit` + `PurgedKFold` with purge & embargo
- [ ] **All six baselines**
- [ ] Metrics module (all five classes) + `skill_score` + Diebold-Mariano + block bootstrap
- [ ] **Leakage test suite** — must be green before any real model is trained
- [ ] `harness.py`: universe × models × horizons × folds → `model_runs`/`fold_metrics`
- **Exit:** `forecaster evaluate --model naive_last_price --symbol AAPL` produces honest fold metrics. This is the week that makes the resume true.

### Week 3 — Models & backtesting
- [ ] `Forecaster` ABC; linear, tree, classical (rolling refit — fixes the dead ARIMA path)
- [ ] `make_windows` + LSTM/GRU/TCN/Transformer with real sequences
- [ ] Optuna hyperparameter search, **fold-internal only**
- [ ] Stacking ensemble on out-of-fold predictions
- [ ] Split-conformal prediction intervals
- [ ] Event-driven backtester: next-bar fills, commission + spread + slippage, vol-targeted sizing, buy-and-hold benchmark
- [ ] SHAP runner + plain-English narratives
- **Exit:** `docs/RESULTS.md` auto-generated. **The real headline number appears here.**

### Week 4 — API
- [ ] FastAPI app factory, DI, RFC 7807 errors, request-ID logging, rate limiting
- [ ] All routers + Pydantic schemas; OpenAPI 3.1
- [ ] Job queue + WebSocket progress
- [ ] Redis caching on OHLCV/indicators; ETag support
- [ ] API tests + schemathesis; generate TS client
- **Exit:** Swagger UI where every claim in the README can be reproduced via HTTP.

### Week 5 — Next.js frontend
- [ ] Scaffold, design tokens, shadcn, theme, layout shell
- [ ] `PriceChart`, `ForecastFan`, `EquityCurve`, `DrawdownChart`, `ModelLeaderboard`, `ShapWaterfall`, `CalibrationPlot`
- [ ] All routes; `⌘K` palette; `ExplainToggle`; live `RunProgress`
- [ ] `/methodology` page (walk-forward + purge/embargo visualization)
- [ ] Responsive, a11y pass (focus rings, ARIA, contrast), Lighthouse ≥ 95
- [ ] Playwright e2e
- **Exit:** deployed to Vercel.

### Week 6 — Streamlit, deploy, polish
- [ ] Streamlit thin client (httpx only — enforce with an import-lint rule)
- [ ] Deploy: Neon, Upstash, API to HF Spaces, Streamlit to HF Spaces
- [ ] GH Actions cron jobs live; `realize_predictions` starts the live OOS track record
- [ ] README: hero GIF, architecture diagram, results table, 60-second quickstart
- [ ] `docs/METHODOLOGY.md`, ADRs, `DATA_SOURCES.md`
- [ ] Perf pass, coverage badge, final security review (no secrets, env-only config)
- **Exit:** shippable. Update the resume.

---

## 12. Rewritten resume bullets

Fill the bracketed values from `docs/RESULTS.md` once the harness has run. **Only claim what the repo prints.**

> **AI-Powered Stock Market Forecasting Platform** — *Kennesaw State University* · Jun 2025 – Present
>
> - Built a **leakage-free walk-forward validation harness** (purged splits with configurable embargo) benchmarking 12+ forecasters — LightGBM, XGBoost, CatBoost, and TensorFlow LSTM/GRU/TCN/Transformer models — against **six naive baselines**, improving directional accuracy **[X]%** over a naive last-price baseline across **[N]** folds and **[M]** tickers, validated by Diebold-Mariano test (**p = [P]**).
> - Engineered a **60+ feature pipeline** (Wilder's RSI, MACD, ATR, Bollinger, OBV, realized volatility, macro factors from FRED, and FinBERT news sentiment) with **as-of joins on SEC EDGAR filing dates** to eliminate fundamental-data lookahead; enforced causality with an automated leakage test suite in CI.
> - Designed a **two-tier data platform** — Postgres for a curated universe and full ML lifecycle state, DuckDB over partitioned Parquet for **9M+ rows** across **7,000+ tickers** — served through a **FastAPI** service with async ingestion, multi-provider failover across 8 free APIs, Redis caching, and WebSocket job streaming.
> - Shipped an **event-driven backtester** with realistic transaction costs, slippage, and volatility-targeted position sizing, reporting cost-adjusted Sharpe, Sortino, Calmar, and max drawdown against a buy-and-hold benchmark with block-bootstrap confidence intervals.
> - Built a **Next.js/TypeScript dashboard** (TradingView charts, conformal prediction intervals, SHAP explanations, plain-English metric mode) on Vercel, plus a **Streamlit client on Hugging Face Spaces** — proving a client-agnostic API — with Docker, GitHub Actions CI/CD, and **80%+ test coverage**.

### Talking points to have loaded
1. *"Why is your R² near zero and why is that fine?"* — Efficient markets; next-day return R² ~0 for everyone. I report directional accuracy and cost-adjusted Sharpe instead, and I ship the baselines so the comparison is visible.
2. *"How do you know you don't have lookahead?"* — Purge + embargo, fold-local fitting, and a test that trains on shuffled targets and asserts zero skill. If the pipeline leaks, that test fails.
3. *"Why two databases?"* — Free-tier Postgres is 0.5 GB; the full universe is ~1.5 GB. Hot/cold tiering with on-demand promotion.
4. *"Is 1-timestep LSTM an LSTM?"* — No. It's a dense layer. My first version had that bug; I found it, wrote `make_windows`, and the sequence models now use a 60-bar lookback.

---

## 13. Immediate next steps (before any new code)

```bash
git rm -r --cached __pycache__ catboost_info stocks.db
printf '__pycache__/\n*.pyc\ncatboost_info/\n*.db\n.env\ndata/\nnode_modules/\n.next/\n' > .gitignore
git commit -m "chore: remove build artifacts and data from version control"
```

Then Week 1. Do not touch the UI until Week 5 — **the ML core is what makes the resume true, and everything else is decoration on top of it.**
