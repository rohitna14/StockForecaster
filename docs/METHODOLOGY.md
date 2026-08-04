# Methodology

How this project validates a forecast, and why each choice is made the way it
is. If you only read one document here, read this one — the modelling is
ordinary, the validation is the part worth defending.

---

## 1. The problem with the obvious approach

The natural first instinct for a price-forecasting project:

1. Download some history.
2. Compute technical indicators.
3. Split 80/20 chronologically.
4. Train a dozen models.
5. Report the best R² on the test set.

Every step after (2) is wrong, and the errors compound in the same direction —
they all make results look better than they are.

| Step | What goes wrong |
|---|---|
| Single 80/20 split | One number from one arbitrary cut point. Move the cut a month and the "best" model changes. |
| Train through bar *T*, test from *T+1* | With an *h*-day target, the last *h* training labels were computed from bars inside the test window. |
| Report best-of-*N* on the test set | With 15 models and ~100 test days, the winner is whichever got luckiest. |
| No baseline | "R² = 0.62" is meaningless without knowing what doing nothing scores. |
| R² as the headline | Next-day return R² is ~0 for everyone. A rubric calling 0.7 "good" is unreachable. |

This repository's first commit contained all five. The rest of this document is
what replaced them.

---

## 2. Walk-forward validation

Train on the past, test on the immediate future, roll forward, repeat.

```
fold 0  |=== train ===|·gap·|test|
fold 1        |=== train ===|·gap·|test|
fold 2              |=== train ===|·gap·|test|
fold 3                    |=== train ===|·gap·|test|
                                                    ──────────▶ time
```

Two modes, both implemented in [`splitters.py`](../backend/src/forecaster/validation/splitters.py):

- **Rolling** — fixed-length training window that slides. Adapts to regime
  change, discards old data.
- **Anchored** — training window grows from a fixed origin. Uses all history,
  adapts more slowly.

The output is a *distribution* of out-of-sample scores across many market
regimes, not a single number. That distribution is what gets a confidence
interval attached to it.

**Default geometry:** 504-bar train (~2 years), 63-bar test (~1 quarter),
step = test size so test windows never overlap. Overlapping test windows make
fold scores correlated, which silently narrows every confidence interval
computed from them.

---

## 3. Purging — the subtle one

This is the error that survives casual review, because the split *looks*
correct.

With an *h*-bar forward target, the label at bar *t* is computed from the close
at *t + h*:

```
y[t] = close[t+h] / close[t] - 1
```

Now train through bar *T* and test from *T+1*. The label `y[T]` was built from
`close[T+h]` — a bar **inside the test window**. So were `y[T-1]`, `y[T-2]`, …
down to `y[T-h+1]`. The model has seen the test period's prices through its own
labels, and the metrics are inflated.

The fix is to drop the last *h* training labels:

```
train:  |=====================|  purge  |  test  |
                               ◀── h ──▶
```

Concretely, for horizon = 5 and a test window opening at bar 1000, training
stops at bar 994, not bar 999.

`Fold.assert_no_leakage()` re-checks this on every fold at runtime and raises
rather than warns. The check is:

```python
max(train_idx) + horizon < min(test_idx)
```

**Embargo.** Returns are serially correlated, so bars immediately after a test
window still carry information about it. An embargo of *e* bars widens the gap
further. Total gap = `horizon + embargo`; the default embargo is 2.

---

## 4. Baselines are models

Seven of them, in [`baselines.py`](../backend/src/forecaster/models/baselines.py):

| Baseline | Forecast | Why it exists |
|---|---|---|
| `naive_last_price` | 0 | The random walk. **The reference for every skill score.** |
| `historical_mean` | in-sample mean return | Random walk with drift. |
| `drift` | extrapolated linear trend | Catches persistent trends. |
| `ewma` | EW mean of recent returns | Tracks regime shifts. |
| `seasonal_naive` | return one period ago | Control for calendar effects. |
| `always_long` | "up" | Equities drift up; this is the real directional bar. |
| `coin_flip` | random at base rate | Control for classification metrics. |

They run through the *same* code path as every other model — fitted per fold,
scored identically, ranked on the same leaderboard. They cannot be filtered out
of the UI. A leaderboard where LightGBM loses to `naive_last_price` is
impossible to hide, and in this project that is exactly what happens on
directional targets.

**Why `naive_last_price` predicts exactly zero:** for a *price* target the naive
forecast is `price[t+h] = price[t]`. Expressed as a return — which is what we
model — that is 0. Random-walk theory says this is close to optimal at short
horizons, and empirically it is very hard to beat.

---

## 5. Skill scores, and what "15% better" can honestly mean

```
skill = 1 − metric_model / metric_baseline
```

Positive beats the baseline; negative loses to it. Both are reported.

**A warning worth stating plainly.** Beating naive last-price by 15% *RMSE on
next-day close* is not realistic. The naive RMSE is roughly one day's
volatility (~1.5–2%), and a 15% reduction on that would be a world-class
result. If a harness reports it, look for a leak before celebrating.

Where a real edge of that magnitude can legitimately appear:

1. **Directional hit rate, relative.** 50.0% → 57.5% *is* +15% relative. Note
   it is +7.5 *points*, and conflating the two is misleading — `SkillResult`
   prints both raw scores for exactly that reason.
2. **Longer horizons.** Multi-day drift is more predictable than one-day noise.
3. **Volatility rather than direction.** This is where this project's real
   result lives — see [RESULTS.md](RESULTS.md).

The measurement target is the base rate, not a coin flip. Equities rise on
~54% of days, so a directional model scoring 54% has added nothing; measuring
against 50% would credit it for market drift it never predicted.

---

## 6. Significance, not just point estimates

A skill score of +8% across 20 folds might be +8% ± 3% (real) or +8% ± 25%
(noise). Three tools separate them, in [`stats.py`](../backend/src/forecaster/validation/stats.py):

- **Diebold–Mariano** — formal test of equal predictive accuracy between two
  forecasts on the same data, with the Harvey–Leybourne–Newbold small-sample
  correction and Newey–West style variance for multi-step horizons.
- **Moving-block bootstrap** — confidence intervals that resample contiguous
  blocks, preserving serial dependence. The i.i.d. bootstrap assumes
  independence and produces intervals far too narrow here, which is precisely
  how a spurious result acquires a convincing error bar.
- **Probability of backtest overfitting (PBO)** and **deflated Sharpe** — how
  likely the in-sample winner is to be below-median out-of-sample, and how much
  of an observed Sharpe is explained by having tried many strategies.

---

## 7. Model selection without peeking

Choosing `argmax(metric)` over the same folds you then report is how the
original prototype crowned its "champion". `select_model_honestly()` selects on
all folds **except the last**, and the held-out final fold gives an unbiased
estimate of the chosen model's performance.

The leaderboard is a *report*, not a selection step.

---

## 8. Preprocessing is fold-local

Everything fitted — scalers, imputers, feature selection, hyperparameter
search, conformal calibration — happens **inside** each fold, on training rows
only.

Fitting a `StandardScaler` on the whole series leaks the test period's mean and
variance into training. It rarely changes results dramatically, which is what
makes it easy to miss and easy to leave in.

---

## 9. Sequence models see sequences

The prototype reshaped its feature matrix to `(n_samples, 1, n_features)` and
fed that to an LSTM. A sequence of length **one** gives the recurrence no
previous hidden state, so the gates reduce to fixed affine transforms and the
layer collapses to a dense layer. Six "deep sequence models", none of which saw
a sequence.

[`make_windows`](../backend/src/forecaster/models/deep/windowing.py) builds real
`(samples, lookback, features)` tensors with a 60-bar lookback, applied **inside
each fold after the split** so no window spans the train/test boundary.

Two model-internal lookahead traps are guarded explicitly:

- The **TCN** uses `padding="causal"`. Plain `"same"` padding is centred and
  reads the future.
- The **Transformer** uses `use_causal_mask=True`. Without it, every position
  attends to every other position, including later ones.

Neither would be caught by any test on the *features* — the leak lives inside
the model. `test_prediction_is_causal` corrupts future inputs and asserts past
predictions do not move.

---

## 10. Prediction intervals

Split conformal: fit on a proper training subset, take absolute residuals on a
held-out calibration slice, use the appropriate finite-sample quantile as the
band half-width.

This costs ~20% of training rows. That is the honest price of an interval with
a coverage guarantee, versus a fabricated one derived from in-sample residuals
(which are too small by construction).

Measured coverage is reported alongside nominal in RESULTS.md. An uncalibrated
interval is worse than no interval, because it invites a decision it cannot
support.

---

## 11. Backtesting

- **Next-bar execution.** A signal from bar *t*'s close fills at bar *t+1*'s
  open. Filling on the signal bar assumes trading at a price you only knew after
  the close, and it is the single most common way to manufacture a fake edge.
- **Costs on turnover**, not per bar: commission + half-spread + volatility-
  scaled slippage + short borrow. Charging per bar unfairly punishes stable
  models.
- **Buy-and-hold benchmark**, always, over the identical window. A strategy
  returning 40% while the underlying returned 80% has destroyed value.
- **Cost sensitivity sweep** answers the question that decides whether an edge
  is real: at what cost level does it disappear?

---

## 12. The leakage test suite

Four independent guards in
[`test_leakage.py`](../backend/tests/unit/test_leakage.py), run as their own CI
job so a failure is visible rather than buried:

**1. Feature causality (per feature).** Corrupt all data after a cut point;
assert every value before it is byte-identical. Catches centred windows,
negative shifts, and whole-series statistics.

**2. Purge correctness.** For every fold and every horizon, assert
`max(train) + horizon < min(test)`.

**3. Shuffled target — the strongest one.** Permute the labels, destroying the
feature/target relationship while preserving the marginal distribution, then
run the entire pipeline. Skill versus baseline must collapse to ~0.

This does not check one specific mistake. It checks the *conclusion*. If any
stage leaks — a feature peeking ahead, a scaler fitted on everything, an
off-by-one, an index misalignment — the model finds the target through the leak
and posts positive skill on noise. Paired with a **positive control** on planted-
signal data, so the suite cannot pass by the pipeline simply being broken.

**4. Preprocessing isolation.** Assert scalers are fitted per fold on training
rows only.

---

## 13. Known limitations

Stated because a methodology section that lists none is not credible.

- **Feature-set selection.** The feature set was chosen once, up front, not
  re-selected per fold. Mild selection effect at that level cannot be ruled out.
- **Survivorship.** `universe_members` has point-in-time columns, but the
  current universes are seeded from *today's* index membership. A backtest over
  the S&P 500 therefore inherits some survivorship bias.
- **Single-name only.** No cross-sectional or pooled training yet, which is how
  the effect is usually exploited in practice.
- **Volatility is not directly tradable.** The headline result forecasts a
  quantity well; converting that into P&L requires options and a pricing layer
  this project does not have. No trading claim is made on the back of it.
- **Costs are assumptions**, not fills. Real slippage depends on size, venue and
  time of day.
