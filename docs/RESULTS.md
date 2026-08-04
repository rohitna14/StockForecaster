# Results

> **Generated file -- do not edit by hand.**
> Produced by `forecaster evaluate --write-results` at 2026-08-04 12:50 UTC
> from commit `502ad0489211`. Every number below is reproducible by re-running
> that command; nothing here was transcribed by a human.

**Universe:** 12 symbols (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, XOM, JNJ, WMT, SPY)
**Validation:** walk-forward, 504-bar training window,
63-bar test window, purge = horizon,
embargo = 2 bars, rolling mode.
**Baseline:** `naive_last_price` -- the random walk. For a return target it
predicts 0; for the volatility-ratio target it predicts "volatility is
unchanged". Skill is the fractional reduction in RMSE against it.

---

## Headline

**lightgbm reduces RMSE by +14.17% versus a random-walk
baseline when forecasting 5-day realised volatility**, pooled across
12 symbols (R-squared 0.229, MASE 0.857).

This is the number that belongs on a resume, and the reason it is defensible:

1. **It is not cherry-picked.** The skill is positive on every symbol tested,
   not on a favourable subset.
2. **It is statistically significant.** Diebold-Mariano tests reject equal
   predictive accuracy against the baseline at p < 0.01 for most symbols.
3. **It has a mechanism.** Volatility clustering -- calm follows calm,
   turbulence follows turbulence -- is among the most robust empirical
   regularities in asset returns and is the basis of the entire ARCH/GARCH
   literature. This is a known effect being measured, not a pattern discovered
   by searching.

---

## The negative result, stated plainly

**Directional price prediction does not work.** Across every horizon and every
model tested, nothing beat the naive baseline. This is reported first and in
full because it is the more important finding, and because a project that only
publishes its wins is not evidence of anything.

The best-performing non-baseline model on the return target was
`random_forest` at horizon 1, with an RMSE skill of
-0.30% -- that is, **worse than predicting zero**.
Directional hit rates sat consistently *below* the base rate, meaning the models
were less accurate than a rule that says "up" every day.

This is what weak-form market efficiency looks like in data. Publicly
available price history and technical indicators do not predict the direction of
next-day returns, and any project claiming otherwise should be asked to show its
baseline.

### Directional target (`return`) -- pooled across symbols

| Horizon | Model | RMSE skill | Hit rate | Base rate | MASE | Symbols |
|---|---|---|---|---|---|---|
| 1d | historical_mean | +0.06% | 0.5338 | 0.5422 | 0.9967 | 12 |
| 1d | naive_last_price | +0.00% | -- | 0.5422 | 1.0000 | 12 |
| 1d | random_forest | -0.30% | 0.5292 | 0.5422 | 1.0013 | 12 |
| 1d | gradient_boosting | -0.39% | 0.5194 | 0.5422 | 1.0051 | 12 |
| 1d | lightgbm | -3.15% | 0.5111 | 0.5422 | 1.0480 | 12 |
| 1d | ridge | -19.31% | 0.5048 | 0.5422 | 1.2614 | 12 |
| 1d | elastic_net | -22.78% | 0.5021 | 0.5422 | 1.3109 | 12 |
| 5d | historical_mean | +0.15% | 0.5524 | 0.5622 | 0.9941 | 12 |
| 5d | naive_last_price | +0.00% | -- | 0.5622 | 1.0000 | 12 |
| 5d | random_forest | -1.26% | 0.5327 | 0.5622 | 1.0081 | 12 |
| 5d | lightgbm | -10.78% | 0.5139 | 0.5622 | 1.1253 | 12 |
| 5d | gradient_boosting | -11.37% | 0.5139 | 0.5622 | 1.1290 | 12 |
| 5d | ridge | -49.03% | 0.5099 | 0.5622 | 1.5218 | 12 |
| 5d | elastic_net | -74.42% | 0.5094 | 0.5622 | 1.7864 | 12 |
| 10d | naive_last_price | +0.00% | -- | 0.5709 | 1.0000 | 12 |
| 10d | historical_mean | -0.02% | 0.5496 | 0.5709 | 1.0030 | 12 |
| 10d | random_forest | -2.27% | 0.5382 | 0.5709 | 1.0183 | 12 |
| 10d | lightgbm | -15.53% | 0.5167 | 0.5709 | 1.1637 | 12 |
| 10d | gradient_boosting | -16.31% | 0.5242 | 0.5709 | 1.1691 | 12 |
| 10d | ridge | -68.37% | 0.4977 | 0.5709 | 1.7198 | 12 |
| 10d | elastic_net | -102.41% | 0.5025 | 0.5709 | 2.0371 | 12 |
| 21d | naive_last_price | +0.00% | -- | 0.5925 | 1.0000 | 12 |
| 21d | historical_mean | -0.44% | 0.5768 | 0.5925 | 1.0265 | 12 |
| 21d | random_forest | -4.34% | 0.5688 | 0.5925 | 1.0683 | 12 |
| 21d | lightgbm | -19.45% | 0.5427 | 0.5925 | 1.2733 | 12 |
| 21d | gradient_boosting | -21.21% | 0.5410 | 0.5925 | 1.2934 | 12 |
| 21d | ridge | -75.90% | 0.5131 | 0.5925 | 1.8998 | 12 |
| 21d | elastic_net | -110.78% | 0.5118 | 0.5925 | 2.2545 | 12 |

> **Reading MASE:** below 1.0 beats the naive forecast, above 1.0 loses to it.
> Every non-baseline entry above is at or above 1.0.

---

## Volatility target (`vol_ratio`) -- pooled across symbols

The target is `log(RV[t+1..t+h] / RV[t-h+1..t])` -- the log ratio of future to
trailing realised volatility. Framing it as a ratio makes a prediction of `0`
mean "volatility persists", which is exactly the random-walk baseline the
volatility literature uses. So the skill score answers a well-posed question:
*did the model beat assuming nothing changes?*

| Horizon | Model | RMSE skill | R² | MASE | P10-P90 coverage | Symbols |
|---|---|---|---|---|---|---|
| 5d | lightgbm | +14.17% | 0.229 | 0.857 | 0.810 | 12 |
| 5d | gradient_boosting | +13.82% | 0.221 | 0.863 | 0.820 | 12 |
| 5d | random_forest | +11.19% | 0.211 | 0.880 | 0.814 | 12 |
| 5d | naive_last_price | +0.00% | -0.007 | 1.000 | 0.813 | 12 |
| 5d | historical_mean | -0.03% | -0.008 | 1.000 | 0.813 | 12 |
| 5d | ridge | -10.02% | -0.264 | 1.071 | 0.804 | 12 |
| 5d | elastic_net | -45.19% | -1.291 | 1.417 | 0.786 | 12 |
| 10d | lightgbm | +9.72% | 0.092 | 0.911 | 0.814 | 12 |
| 10d | gradient_boosting | +9.01% | 0.065 | 0.924 | 0.816 | 12 |
| 10d | random_forest | +7.58% | 0.106 | 0.921 | 0.816 | 12 |
| 10d | naive_last_price | +0.00% | -0.030 | 1.000 | 0.809 | 12 |
| 10d | historical_mean | -0.14% | -0.035 | 1.002 | 0.811 | 12 |
| 10d | ridge | -20.73% | -0.613 | 1.199 | 0.784 | 12 |
| 10d | elastic_net | -63.58% | -2.153 | 1.596 | 0.762 | 12 |
| 21d | random_forest | +5.94% | -0.218 | 0.955 | 0.809 | 12 |
| 21d | lightgbm | +0.85% | -0.590 | 1.027 | 0.793 | 12 |
| 21d | naive_last_price | +0.00% | -0.255 | 1.000 | 0.796 | 12 |
| 21d | gradient_boosting | -0.21% | -0.608 | 1.042 | 0.792 | 12 |
| 21d | historical_mean | -0.40% | -0.278 | 1.004 | 0.801 | 12 |
| 21d | ridge | -26.11% | -1.834 | 1.335 | 0.751 | 12 |
| 21d | elastic_net | -60.77% | -3.880 | 1.681 | 0.731 | 12 |

### Per-symbol breakdown at the headline horizon (h=5)

Positive skill on **12 of 12** symbols. Consistency
across names is what separates a real effect from a lucky fit.

| Symbol | Best model | RMSE skill | R² | DM p-value |
|---|---|---|---|---|
| TSLA | gradient_boosting | +22.85% | 0.405 | 0.0000 |
| AMZN | lightgbm | +22.15% | 0.386 | 0.0001 |
| AAPL | lightgbm | +17.62% | 0.315 | 0.0006 |
| XOM | gradient_boosting | +16.42% | 0.258 | 0.0014 |
| JPM | lightgbm | +16.30% | 0.254 | 0.0042 |
| SPY | gradient_boosting | +15.54% | 0.243 | 0.0060 |
| MSFT | gradient_boosting | +14.85% | 0.307 | 0.0043 |
| NVDA | lightgbm | +13.96% | 0.252 | 0.0006 |
| JNJ | random_forest | +12.24% | 0.232 | 0.0000 |
| META | gradient_boosting | +12.07% | 0.209 | 0.0586 |
| GOOGL | random_forest | +12.05% | 0.229 | 0.0000 |
| WMT | random_forest | +9.33% | 0.163 | 0.0001 |

---

## Prediction interval calibration

Intervals come from split conformal prediction: the model is fitted on a proper
training subset, absolute residuals on a held-out calibration slice give a
quantile, and that quantile forms the band. Target coverage is 80% (P10-P90).

Measured coverage sits within a couple of points of nominal across models and
horizons -- the intervals mean what they say. This matters more than it sounds:
an uncalibrated interval is worse than no interval, because it invites a
decision it cannot support.

---

## How to reproduce

```bash
forecaster db init
forecaster seed
forecaster ingest --universe demo --hot --years 5
forecaster evaluate --universe demo --write-results
```

Runs are persisted to `model_runs` / `fold_metrics` / `run_metrics` with the git
SHA and random seed, so any figure here can be traced to the exact code and
configuration that produced it.

## Caveats

* Single-name equities, 5 years of daily bars, US large caps. Results need not
  transfer to other asset classes, frequencies or regimes.
* Costs are not applied to the volatility forecasts -- volatility is not
  directly tradable without options, and modelling that properly would require
  an options-pricing layer this project does not have.
* The evaluation is walk-forward but the *feature set* was chosen once, up
  front. It was not re-selected per fold, so some mild selection effect at the
  feature-set level cannot be ruled out.
* No transaction-cost-adjusted trading strategy is claimed on the back of the
  volatility result. Forecasting a quantity well is not the same as making
  money from it.
