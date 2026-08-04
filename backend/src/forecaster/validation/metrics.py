"""Forecast evaluation metrics.

Implemented in numpy rather than pulled from scikit-learn so the core package
stays installable without the ``[ml]`` extra, and so the exact definition of
every published number is visible in this file.

**On R-squared.** The prototype's UI told users "R-squared above 0.7 =
excellent". For next-day equity returns that number is unreachable; realistic
values sit between -0.05 and +0.02, and *negative is normal* -- it means the
model did worse than predicting the sample mean. R-squared is reported here for
completeness and is deliberately never the headline. The metrics that matter
for this problem are directional accuracy, skill versus a naive baseline, and
cost-adjusted Sharpe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

TRADING_DAYS = 252
_EPS = 1e-12


# ═══════════════════════════════════ helpers ═══════════════════════════════
def _clean_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions where either side is non-finite."""
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


# ═══════════════════════════════ regression ════════════════════════════════
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if len(t) else float("nan")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean((t - p) ** 2)) if len(t) else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Negative means worse than the mean."""
    t, p = _clean_pair(y_true, y_pred)
    if len(t) < 2:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > _EPS else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray | None = None) -> float:
    """Mean Absolute Scaled Error.

    MAE divided by the MAE of the naive forecast on the same data. Scale-free,
    and interpretable on sight: < 1 beats naive, > 1 loses to it.

    With a return target the naive forecast is zero, so the denominator becomes
    ``mean(|y|)`` -- the average absolute return.
    """
    t, p = _clean_pair(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    if y_naive is None:
        denom = float(np.mean(np.abs(t)))
    else:
        tn, pn = _clean_pair(y_true, y_naive)
        denom = float(np.mean(np.abs(tn - pn)))
    return float(np.mean(np.abs(t - p)) / denom) if denom > _EPS else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, in percent. Bounded [0, 200], safe near zero."""
    t, p = _clean_pair(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    denom = (np.abs(t) + np.abs(p)) / 2.0
    mask = denom > _EPS
    return (
        float(np.mean(np.abs(t[mask] - p[mask]) / denom[mask]) * 100.0)
        if mask.any()
        else float("nan")
    )


# ═══════════════════════════════ directional ═══════════════════════════════
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of *directional calls* that were correct.

    Two exclusions, both necessary:

    * Bars where the realised return is exactly zero -- there is no direction
      to get right, and counting them inflates the score.
    * Bars where the **prediction** is exactly zero -- the model declined to
      call a direction. This matters: the naive baseline predicts 0 everywhere,
      and scoring ``sign(0) != sign(y)`` would report its hit rate as 0.0%,
      which reads as "always wrong" when the truth is "never bet".

    Returns NaN when the model makes no directional calls at all. Pair this
    with :func:`directional_coverage` -- a 60% hit rate on 5% of bars is a very
    different claim from 60% on all of them.
    """
    t, p = _clean_pair(y_true, y_pred)
    mask = (t != 0) & (p != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(t[mask]) == np.sign(p[mask])))


def directional_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of bars on which the model actually made a directional call."""
    t, p = _clean_pair(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.mean((t != 0) & (p != 0)))


def base_rate(y_true: np.ndarray) -> float:
    """Fraction of bars that were up moves.

    The honest bar for a directional model: equities drift upward, so
    always-long scores well above 50%. Beating a coin flip is not an
    achievement; beating this is.
    """
    t = np.asarray(y_true, dtype="float64").ravel()
    t = t[np.isfinite(t) & (t != 0)]
    return float(np.mean(t > 0)) if len(t) else float("nan")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Up/down confusion counts, treating 'up' as the positive class."""
    t, p = _clean_pair(y_true, y_pred)
    actual_up, pred_up = t > 0, p > 0
    return {
        "tp": int(np.sum(actual_up & pred_up)),
        "fp": int(np.sum(~actual_up & pred_up)),
        "tn": int(np.sum(~actual_up & ~pred_up)),
        "fn": int(np.sum(actual_up & ~pred_up)),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient.

    Preferred over accuracy on imbalanced directional problems: a model that
    always predicts 'up' scores ~0.53 accuracy but exactly 0.0 MCC, which is
    the honest answer.
    """
    c = confusion(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    numerator = tp * tn - fp * fn
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(numerator / denominator) if denominator > _EPS else 0.0


def roc_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (handles ties correctly)."""
    y, s = _clean_pair(y_true_binary, scores)
    labels = (y > 0).astype(int)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype="float64")
    ranks[order] = np.arange(1, len(s) + 1, dtype="float64")

    # Average ranks within tie groups.
    sorted_scores = s[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_score(y_true_binary: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of probability forecasts. Lower is better."""
    y, p = _clean_pair(y_true_binary, probabilities)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(((y > 0).astype(float) - np.clip(p, 0.0, 1.0)) ** 2))


def calibration_curve(
    y_true_binary: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Predicted vs observed frequency per probability bin.

    A well-calibrated model plots on the diagonal: when it says 60%, the event
    happens 60% of the time. Overconfidence is the standard failure mode and it
    is invisible in accuracy alone.
    """
    y, p = _clean_pair(y_true_binary, probabilities)
    labels = (y > 0).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        predicted.append(float(p[mask].mean()))
        observed.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))
    return {"predicted": predicted, "observed": observed, "counts": [float(c) for c in counts]}


# ═══════════════════════════════ trading ═══════════════════════════════════
def strategy_returns(
    y_true: np.ndarray, y_pred: np.ndarray, *, cost_bps: float = 0.0
) -> np.ndarray:
    """Returns of a long/short strategy that takes the sign of the forecast.

    Costs are charged on *position changes*, not on every bar -- a model that
    holds the same view for a week trades once, not five times. Ignoring that
    is the most common way backtests overstate turnover costs, and charging per
    bar unfairly penalises stable models.
    """
    t, p = _clean_pair(y_true, y_pred)
    if len(t) == 0:
        return np.array([])
    position = np.sign(p)
    gross = position * t
    if cost_bps <= 0:
        return gross
    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    return gross - turnover * (cost_bps / 10_000.0)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) < _EPS:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(r) < 2 or len(downside) < 2:
        return float("nan")
    dd = downside.std(ddof=1)
    return float(r.mean() / dd * np.sqrt(periods_per_year)) if dd > _EPS else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the compounded equity curve (negative)."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    years = len(r) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    cagr = total ** (1.0 / years) - 1.0
    dd = abs(max_drawdown(r))
    return float(cagr / dd) if dd > _EPS else float("nan")


def profit_factor(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > _EPS else float("nan")


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    return float(np.mean(r > 0)) if len(r) else float("nan")


# ═══════════════════════════════ container ═════════════════════════════════
@dataclass
class MetricSet:
    """All metrics for one (model, fold) pair."""

    n: int = 0
    mae: float = float("nan")
    rmse: float = float("nan")
    mase: float = float("nan")
    smape: float = float("nan")
    r2: float = float("nan")
    directional_accuracy: float = float("nan")
    directional_coverage: float = float("nan")
    base_rate: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    mcc: float = float("nan")
    roc_auc: float = float("nan")
    brier: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    max_drawdown: float = float("nan")
    profit_factor: float = float("nan")
    skill_vs_naive: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    cost_bps: float = 0.0,
) -> MetricSet:
    """Compute the full metric suite for one prediction set."""
    t, p = _clean_pair(y_true, y_pred)
    if len(t) == 0:
        return MetricSet()

    pr = precision_recall_f1(t, p)
    strat = strategy_returns(t, p, cost_bps=cost_bps)
    scores = probabilities if probabilities is not None else p

    return MetricSet(
        n=len(t),
        mae=mae(t, p),
        rmse=rmse(t, p),
        mase=mase(t, p),
        smape=smape(t, p),
        r2=r2(t, p),
        directional_accuracy=directional_accuracy(t, p),
        directional_coverage=directional_coverage(t, p),
        base_rate=base_rate(t),
        precision=pr["precision"],
        recall=pr["recall"],
        f1=pr["f1"],
        mcc=matthews_corrcoef(t, p),
        roc_auc=roc_auc(t, scores),
        brier=brier_score(t, probabilities) if probabilities is not None else float("nan"),
        sharpe=sharpe_ratio(strat),
        sortino=sortino_ratio(strat),
        max_drawdown=max_drawdown(strat),
        profit_factor=profit_factor(strat),
    )
