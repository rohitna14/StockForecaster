"""Skill scores -- performance *relative to a baseline*.

This module produces the number the resume claim rests on, so it is worth being
precise about what each variant means.

A skill score answers: "by how much did the model reduce error compared to
doing nothing?"::

    skill = 1 - metric_model / metric_baseline

* ``skill > 0``  -- the model beat the baseline
* ``skill = 0``  -- indistinguishable from the baseline
* ``skill < 0``  -- worse than the baseline (common, and not reported away)

**Which variant to quote, and why it matters.**

``rmse_skill`` at horizon 1 is the hardest bar in the entire project. The naive
RMSE is roughly one day's volatility (~1.5-2%), and a 15% reduction on that
would be a world-class result -- if you measure it, suspect a leak before
celebrating. Realistic values at h=1 are 0-3%.

``directional_skill`` is the variant where a real, defensible edge of ~10-15%
*relative* can show up: 50.0% -> 57.5% hit rate is a 15% relative improvement,
and that is an honest sentence.

``rmse_skill`` at h=5 or h=10 is also more forgiving -- multi-day drift is more
predictable than one-day noise.

The rule this project follows: **run the harness, quote whichever variant the
data actually supports, and always state the metric, horizon, fold count and
confidence interval alongside the number.**
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from forecaster.validation.metrics import (
    _clean_pair,
    directional_accuracy,
    mae,
    mse,
    rmse,
)

_EPS = 1e-12


@dataclass(frozen=True)
class SkillResult:
    """A skill score plus everything needed to interpret it."""

    metric: str
    model_score: float
    baseline_score: float
    skill: float
    #: Relative improvement expressed as a percentage, for prose.
    improvement_pct: float
    n: int
    baseline_name: str
    higher_is_better: bool

    def summary(self) -> str:
        direction = "better than" if self.improvement_pct > 0 else "worse than"
        return (
            f"{self.metric}: {self.model_score:.6f} vs baseline "
            f"{self.baseline_score:.6f} ({abs(self.improvement_pct):.2f}% "
            f"{direction} {self.baseline_name}, n={self.n})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "model_score": self.model_score,
            "baseline_score": self.baseline_score,
            "skill": self.skill,
            "improvement_pct": self.improvement_pct,
            "n": self.n,
            "baseline": self.baseline_name,
        }


def skill_score(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = mse,
    metric_name: str = "mse",
    baseline_name: str = "naive_last_price",
    higher_is_better: bool = False,
) -> SkillResult:
    """Generic skill score for an error metric (lower is better by default)."""
    t, m = _clean_pair(y_true, y_pred_model)
    _, b = _clean_pair(y_true, y_pred_baseline)

    n = min(len(t), len(b))
    if n == 0:
        return SkillResult(metric_name, float("nan"), float("nan"), float("nan"),
                           float("nan"), 0, baseline_name, higher_is_better)

    model_score = float(metric(t[:n], m[:n]))
    baseline_score = float(metric(t[:n], b[:n]))

    if not np.isfinite(baseline_score) or abs(baseline_score) < _EPS:
        return SkillResult(metric_name, model_score, baseline_score, float("nan"),
                           float("nan"), n, baseline_name, higher_is_better)

    if higher_is_better:
        skill = (model_score - baseline_score) / abs(baseline_score)
    else:
        skill = 1.0 - model_score / baseline_score

    return SkillResult(
        metric=metric_name,
        model_score=model_score,
        baseline_score=baseline_score,
        skill=float(skill),
        improvement_pct=float(skill * 100.0),
        n=n,
        baseline_name=baseline_name,
        higher_is_better=higher_is_better,
    )


def rmse_skill(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray,
    baseline_name: str = "naive_last_price",
) -> SkillResult:
    """RMSE reduction versus the baseline. The strictest variant."""
    return skill_score(
        y_true, y_pred_model, y_pred_baseline,
        metric=rmse, metric_name="rmse", baseline_name=baseline_name,
    )


def mse_skill(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray,
    baseline_name: str = "naive_last_price",
) -> SkillResult:
    """MSE reduction -- the classical Brier/Murphy skill score."""
    return skill_score(
        y_true, y_pred_model, y_pred_baseline,
        metric=mse, metric_name="mse", baseline_name=baseline_name,
    )


def mae_skill(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray,
    baseline_name: str = "naive_last_price",
) -> SkillResult:
    return skill_score(
        y_true, y_pred_model, y_pred_baseline,
        metric=mae, metric_name="mae", baseline_name=baseline_name,
    )


def directional_skill(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray | None = None,
    baseline_name: str = "coin_flip",
) -> SkillResult:
    """Relative improvement in directional hit rate.

    When ``y_pred_baseline`` is None the comparison is against a 50% coin flip,
    which makes the arithmetic transparent: 57.5% vs 50% is +15.0%.

    Note this measures *relative* improvement, not percentage points. Quoting it
    without saying so would be misleading, so :meth:`SkillResult.summary`
    always prints both scores.
    """
    t, m = _clean_pair(y_true, y_pred_model)
    model_acc = directional_accuracy(t, m)

    if y_pred_baseline is None:
        baseline_acc = 0.5
    else:
        _, b = _clean_pair(y_true, y_pred_baseline)
        baseline_acc = directional_accuracy(t[: len(b)], b)

    if not np.isfinite(baseline_acc) or baseline_acc < _EPS:
        return SkillResult("directional_accuracy", model_acc, baseline_acc,
                           float("nan"), float("nan"), len(t), baseline_name, True)

    skill = (model_acc - baseline_acc) / baseline_acc
    return SkillResult(
        metric="directional_accuracy",
        model_score=float(model_acc),
        baseline_score=float(baseline_acc),
        skill=float(skill),
        improvement_pct=float(skill * 100.0),
        n=len(t),
        baseline_name=baseline_name,
        higher_is_better=True,
    )


def all_skill_scores(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    baseline_name: str = "naive_last_price",
) -> dict[str, SkillResult]:
    """Every skill variant at once.

    The harness stores all of them so that the choice of headline metric is a
    reporting decision made *after* seeing the numbers -- and so the ones that
    look worse are still on the record.
    """
    return {
        "rmse": rmse_skill(y_true, y_pred_model, y_pred_baseline, baseline_name),
        "mse": mse_skill(y_true, y_pred_model, y_pred_baseline, baseline_name),
        "mae": mae_skill(y_true, y_pred_model, y_pred_baseline, baseline_name),
        "directional": directional_skill(y_true, y_pred_model, y_pred_baseline, baseline_name),
    }
