"""Statistical significance testing for forecast comparisons.

A skill score of +8% means nothing on its own. Across 20 folds it might be
+8% +/- 3% (real) or +8% +/- 25% (noise). This module supplies the machinery
that tells those apart, which is the difference between a claim that survives
an interview and one that does not.

* :func:`diebold_mariano` -- formal test of equal predictive accuracy between
  two forecasts on the same data.
* :func:`block_bootstrap_ci` -- confidence intervals that respect serial
  correlation, unlike the naive i.i.d. bootstrap.
* :func:`probability_of_backtest_overfitting` -- how likely the best-in-sample
  model is to be below-median out-of-sample.
* :func:`deflated_sharpe_ratio` -- Sharpe adjusted for how many strategies were
  tried before this one looked good.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

_EPS = 1e-12


# ═══════════════════════════════ Diebold-Mariano ═══════════════════════════
@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float
    n: int
    horizon: int
    loss: str
    mean_loss_differential: float

    @property
    def is_significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    def interpretation(self) -> str:
        if not np.isfinite(self.p_value):
            return "Test could not be computed."
        if self.p_value >= 0.05:
            return (
                f"No significant difference in predictive accuracy (p={self.p_value:.3f}). "
                "The apparent edge is within noise."
            )
        better = "model 1" if self.mean_loss_differential < 0 else "model 2"
        return f"{better} is significantly more accurate (p={self.p_value:.4f})."

    def as_dict(self) -> dict[str, Any]:
        return {
            "dm_stat": self.statistic,
            "dm_pvalue": self.p_value,
            "n": self.n,
            "significant": self.is_significant,
            "interpretation": self.interpretation(),
        }


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    *,
    horizon: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    harvey_correction: bool = True,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests H0: the two forecasts have equal expected loss. The statistic is the
    mean loss differential scaled by its long-run standard error, where the
    variance uses a Newey-West style correction with ``horizon - 1`` lags --
    necessary because multi-step forecast errors are autocorrelated by
    construction.

    ``harvey_correction`` applies the Harvey-Leybourne-Newbold small-sample
    adjustment, which matters at the fold sizes used here (~60 observations).

    A negative statistic favours ``y_pred_1``.
    """
    y_true = np.asarray(y_true, dtype="float64").ravel()
    p1 = np.asarray(y_pred_1, dtype="float64").ravel()
    p2 = np.asarray(y_pred_2, dtype="float64").ravel()

    n_common = min(len(y_true), len(p1), len(p2))
    y_true, p1, p2 = y_true[:n_common], p1[:n_common], p2[:n_common]
    mask = np.isfinite(y_true) & np.isfinite(p1) & np.isfinite(p2)
    y_true, p1, p2 = y_true[mask], p1[mask], p2[mask]

    n = len(y_true)
    if n < 10:
        return DieboldMarianoResult(float("nan"), float("nan"), n, horizon, loss, float("nan"))

    e1, e2 = y_true - p1, y_true - p2
    if loss == "squared":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)

    d_mean = float(d.mean())

    # Long-run variance with horizon-1 autocovariance lags.
    gamma_0 = float(np.sum((d - d_mean) ** 2) / n)
    lrv = gamma_0
    for lag in range(1, horizon):
        if lag >= n:
            break
        cov = float(np.sum((d[lag:] - d_mean) * (d[:-lag] - d_mean)) / n)
        lrv += 2.0 * cov

    if lrv <= _EPS:
        return DieboldMarianoResult(float("nan"), float("nan"), n, horizon, loss, d_mean)

    dm_stat = d_mean / math.sqrt(lrv / n)

    if harvey_correction and horizon > 1:
        adj = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat *= adj

    # Student-t with n-1 df (HLN recommendation) rather than the normal.
    p_value = 2.0 * (1.0 - _student_t_cdf(abs(dm_stat), n - 1))

    return DieboldMarianoResult(
        statistic=float(dm_stat),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        n=n,
        horizon=horizon,
        loss=loss,
        mean_loss_differential=d_mean,
    )


def _student_t_cdf(x: float, df: int) -> float:
    """Student-t CDF via the regularised incomplete beta function.

    Implemented locally so ``validation`` does not depend on the ``[ml]`` extra.
    """
    if df <= 0:
        return float("nan")
    if not np.isfinite(x):
        return 1.0 if x > 0 else 0.0
    prob = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - prob if x > 0 else prob


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) via a continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return (
        1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _beta_cf(b, a, 1.0 - x) / b
    )


def _beta_cf(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Lentz's algorithm for the beta continued fraction."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h


# ═══════════════════════════════ bootstrap ═════════════════════════════════
@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    std_error: float
    confidence: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        return bool(self.ci_low > 0 or self.ci_high < 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "confidence": self.confidence,
        }


def block_bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    block_size: int | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Moving-block bootstrap confidence interval.

    Resamples contiguous blocks rather than individual observations, preserving
    the serial dependence in financial time series. The i.i.d. bootstrap assumes
    independence and produces intervals that are far too narrow here -- which is
    exactly how a spurious result acquires a convincing error bar.

    Default block size is ``n^(1/3)``, the standard rule of thumb.
    """
    x = np.asarray(values, dtype="float64").ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), float("nan"), confidence, 0)

    block_size = block_size or max(1, int(round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)

    starts = rng.integers(0, max(1, n - block_size + 1), size=(n_resamples, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, :n]
    idx = np.clip(idx, 0, n - 1)
    samples = x[idx]

    stats = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(x.mean() if statistic == "mean" else np.median(x))

    return BootstrapCI(
        point_estimate=point,
        ci_low=float(low),
        ci_high=float(high),
        std_error=float(stats.std(ddof=1)),
        confidence=confidence,
        n_resamples=n_resamples,
    )


# ═══════════════════════════ overfitting diagnostics ═══════════════════════
def probability_of_backtest_overfitting(in_sample: np.ndarray, out_of_sample: np.ndarray) -> float:
    """Probability that the in-sample best performer ranks below median OOS.

    Bailey et al.'s PBO. Given per-configuration in-sample and out-of-sample
    scores across folds, it estimates how often "the winner" is really just the
    luckiest. Values above ~0.5 mean the selection procedure is not extracting
    signal.

    Args:
        in_sample: (n_folds, n_configs) in-sample scores.
        out_of_sample: (n_folds, n_configs) out-of-sample scores.
    """
    is_scores = np.asarray(in_sample, dtype="float64")
    oos_scores = np.asarray(out_of_sample, dtype="float64")
    if is_scores.ndim != 2 or is_scores.shape != oos_scores.shape:
        raise ValueError("in_sample and out_of_sample must be equal-shaped 2-D arrays")

    n_folds, n_configs = is_scores.shape
    if n_configs < 2:
        return float("nan")

    below_median = 0
    valid = 0
    for fold in range(n_folds):
        row_is, row_oos = is_scores[fold], oos_scores[fold]
        if not (np.isfinite(row_is).all() and np.isfinite(row_oos).all()):
            continue
        best = int(np.argmax(row_is))
        rank = float(np.mean(row_oos <= row_oos[best]))
        below_median += int(rank < 0.5)
        valid += 1

    return float(below_median / valid) if valid else float("nan")


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe ratio deflated for multiple testing.

    Trying 15 model configurations and reporting the best one's Sharpe without
    adjustment is the quantitative equivalent of p-hacking. This returns the
    probability that the observed Sharpe exceeds what the *best of n_trials*
    random strategies would produce.

    Values below ~0.95 mean the result is not distinguishable from selection
    luck.
    """
    if n_trials < 1 or n_observations < 2 or not np.isfinite(observed_sharpe):
        return float("nan")

    euler = 0.5772156649015329
    # Expected maximum Sharpe from n_trials independent zero-skill strategies.
    if n_trials == 1:
        expected_max = 0.0
    else:
        z1 = _inverse_normal_cdf(1.0 - 1.0 / n_trials)
        z2 = _inverse_normal_cdf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = (1 - euler) * z1 + euler * z2

    denom = math.sqrt(
        max(
            _EPS,
            1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2,
        )
    )
    statistic = (observed_sharpe - expected_max) * math.sqrt(n_observations - 1) / denom
    return float(_normal_cdf(statistic))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation to the normal quantile function."""
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
