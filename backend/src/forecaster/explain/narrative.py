"""Plain-English explanations.

Powers the frontend's "explain like I'm five" toggle. Every metric shown to a
user has an entry here, because a dashboard that displays "Sortino 1.42" with
no explanation is decoration, not information.

Two rules the copy follows:

1. **Never overstate.** Where a metric is weak or a result is negative, the text
   says so. The tooltips are the main defence against a user reading a 53% hit
   rate as a money printer.
2. **Say what to do with it.** A definition without a decision rule is trivia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricExplanation:
    label: str
    short: str
    plain: str
    good_direction: str  # "higher" | "lower" | "context"
    caveat: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "short": self.short,
            "plain": self.plain,
            "good_direction": self.good_direction,
            "caveat": self.caveat,
        }


METRIC_GLOSSARY: dict[str, MetricExplanation] = {
    "rmse": MetricExplanation(
        label="RMSE",
        short="Typical size of the model's error.",
        plain=(
            "On average, how far off the forecast was — with big misses "
            "counting extra. Lower is better, but the number only means "
            "something next to the baseline's."
        ),
        good_direction="lower",
    ),
    "mase": MetricExplanation(
        label="MASE",
        short="Error relative to the naive forecast.",
        plain=(
            "Below 1.0 means the model beat 'assume nothing changes'. Above 1.0 "
            "means it lost to it. This is the single most honest number on the "
            "page because it has the comparison built in."
        ),
        good_direction="lower",
    ),
    "r2": MetricExplanation(
        label="R²",
        short="Share of variance explained.",
        plain=(
            "How much of the movement the model accounts for. For next-day stock "
            "returns this is near zero for everyone, and negative is normal — it "
            "just means the model did worse than guessing the average."
        ),
        good_direction="higher",
        caveat=(
            "Do not judge a return forecast by R². Values above ~0.05 on daily "
            "returns almost always indicate a data leak rather than skill."
        ),
    ),
    "directional_accuracy": MetricExplanation(
        label="Hit rate",
        short="How often the up/down call was right.",
        plain=(
            "The share of directional calls the model got correct. Compare it to "
            "the base rate, not to 50% — stocks rise on roughly 54% of days, so "
            "a 54% hit rate has added nothing."
        ),
        good_direction="higher",
        caveat="Meaningless without the base rate and the number of calls made.",
    ),
    "base_rate": MetricExplanation(
        label="Base rate",
        short="How often the stock actually rose.",
        plain=(
            "The percentage of days the price went up. This is the bar a "
            "directional model has to clear — always predicting 'up' scores "
            "exactly this."
        ),
        good_direction="context",
    ),
    "skill_vs_naive": MetricExplanation(
        label="Skill vs naive",
        short="Error reduction against doing nothing.",
        plain=(
            "How much smaller the model's error is than the naive baseline's. "
            "+10% means it cut the error by a tenth. Negative means the naive "
            "forecast was better, which happens more often than most projects "
            "admit."
        ),
        good_direction="higher",
    ),
    "dm_pvalue": MetricExplanation(
        label="Diebold-Mariano p-value",
        short="Is the difference real or luck?",
        plain=(
            "The probability of seeing this much difference between the model and "
            "the baseline if they were genuinely equally good. Below 0.05 means "
            "the gap is unlikely to be noise. Above it, treat any apparent edge "
            "as unproven."
        ),
        good_direction="lower",
    ),
    "sharpe": MetricExplanation(
        label="Sharpe ratio",
        short="Return per unit of risk.",
        plain=(
            "How much return the strategy earned for the amount it bounced "
            "around. Above 1.0 is decent, above 2.0 is very good — but only "
            "after costs, and only next to buy-and-hold's own Sharpe."
        ),
        good_direction="higher",
        caveat="A Sharpe quoted without transaction costs is a marketing number.",
    ),
    "sortino": MetricExplanation(
        label="Sortino ratio",
        short="Return per unit of *downside* risk.",
        plain=(
            "Like Sharpe, but it only penalises moves against you. Upside "
            "volatility is not risk, and Sortino is the version that agrees."
        ),
        good_direction="higher",
    ),
    "max_drawdown": MetricExplanation(
        label="Max drawdown",
        short="Worst peak-to-trough loss.",
        plain=(
            "The most the account fell from a high point before recovering. This "
            "is the number that decides whether a strategy is actually holdable — "
            "most people abandon a good strategy during a bad drawdown."
        ),
        good_direction="higher",
    ),
    "interval_coverage": MetricExplanation(
        label="Interval coverage",
        short="Do the prediction bands mean what they say?",
        plain=(
            "The share of actual outcomes that landed inside the model's "
            "80% band. Close to 0.80 means the uncertainty estimate is honest. "
            "Much lower means the model is overconfident."
        ),
        good_direction="context",
    ),
    "volatility": MetricExplanation(
        label="Volatility",
        short="How much the price swings.",
        plain=(
            "The annualised size of typical price moves. Higher means bigger "
            "swings in both directions — it is a measure of uncertainty, not of "
            "whether the price will go up or down."
        ),
        good_direction="context",
    ),
}


#: Human-readable descriptions of feature groups, for the SHAP panel.
FEATURE_GROUP_PLAIN: dict[str, str] = {
    "returns": "recent price changes",
    "momentum": "whether the recent trend has been strong or exhausted",
    "trend": "where the price sits relative to its moving averages",
    "volatility": "how turbulent trading has been lately",
    "volume": "whether trading activity is unusually heavy or light",
    "statistical": "the shape and persistence of recent returns",
    "calendar": "day-of-week and month-end effects",
}


def explain_metric(name: str, value: float | None = None) -> dict[str, Any]:
    """Glossary entry, optionally with a verdict for a specific value."""
    entry = METRIC_GLOSSARY.get(name)
    if entry is None:
        return {"label": name, "short": "", "plain": "", "good_direction": "context"}

    payload = entry.as_dict()
    if value is not None and np.isfinite(value):
        payload["verdict"] = _verdict(name, float(value))
    return payload


def _verdict(name: str, value: float) -> str:
    if name == "mase":
        return "beats the naive baseline" if value < 1.0 else "loses to the naive baseline"
    if name == "skill_vs_naive":
        return "better than naive" if value > 0 else "worse than naive"
    if name == "dm_pvalue":
        return "statistically significant" if value < 0.05 else "not statistically significant"
    if name == "interval_coverage":
        return "well calibrated" if 0.75 <= value <= 0.85 else "poorly calibrated"
    if name == "sharpe":
        if value > 2.0:
            return "excellent"
        return "solid" if value > 1.0 else "weak"
    return ""


def describe_forecast(
    *,
    symbol: str,
    model_name: str,
    prediction: float,
    lower: float | None,
    upper: float | None,
    horizon: int,
    skill_pct: float | None,
    dm_pvalue: float | None,
    target_type: str = "return",
) -> str:
    """One honest paragraph describing a forecast, for the UI.

    Deliberately leads with uncertainty rather than the point estimate. A
    headline number with the caveat buried underneath is how forecasting tools
    mislead people, and this project's whole argument is that it should not.
    """
    # The vol_ratio target lives in log space: a prediction of 0.2 means
    # exp(0.2) - 1 = +22% volatility, not +20%. The bounds must go through the
    # same transform as the point estimate, or the interval reads as nonsense
    # (a raw log lower bound of -0.98 would print as "-98%", implying volatility
    # could fall to nearly zero).
    is_log_ratio = target_type == "vol_ratio"

    def to_pct(value: float) -> float:
        return float(np.expm1(value) * 100) if is_log_ratio else float(value * 100)

    if is_log_ratio:
        direction = "more volatile" if prediction > 0 else "calmer"
        core = (
            f"Over the next {horizon} trading days, the model expects {symbol} to be "
            f"about {abs(to_pct(prediction)):.0f}% {direction} than it has been recently."
        )
    else:
        direction = "up" if prediction > 0 else "down"
        core = (
            f"Over the next {horizon} trading day{'s' if horizon > 1 else ''}, "
            f"{model_name} forecasts {symbol} {direction} {abs(prediction) * 100:.2f}%."
        )

    parts = [core]

    if lower is not None and upper is not None and np.isfinite(lower) and np.isfinite(upper):
        low_pct, high_pct = to_pct(lower), to_pct(upper)
        span = (
            f"{low_pct:+.0f}% to {high_pct:+.0f}% change in volatility"
            if is_log_ratio
            else f"{low_pct:+.2f}% to {high_pct:+.2f}%"
        )
        parts.append(
            f"The 80% range runs from {span} — that width is the honest part of this forecast."
        )

    if skill_pct is not None and np.isfinite(skill_pct):
        if skill_pct <= 0:
            parts.append(
                f"Be aware: on historical data this model was **{abs(skill_pct):.1f}% worse** "
                f"than simply assuming no change. Treat the forecast as uninformative."
            )
        elif dm_pvalue is not None and np.isfinite(dm_pvalue) and dm_pvalue >= 0.05:
            parts.append(
                f"It beat the naive baseline by {skill_pct:.1f}% historically, but that "
                f"gap is not statistically significant (p={dm_pvalue:.2f}), so it may be luck."
            )
        else:
            significance = (
                f", significant at p={dm_pvalue:.3f}"
                if dm_pvalue is not None and np.isfinite(dm_pvalue)
                else ""
            )
            parts.append(
                f"Historically this model reduced forecast error by {skill_pct:.1f}% "
                f"versus assuming no change{significance}."
            )

    parts.append("Not financial advice. Past performance does not predict future results.")
    return " ".join(parts)


def describe_drivers(top_features: list[tuple[str, float]], n: int = 3) -> str:
    """Turn SHAP importances into a sentence about what drove the forecast."""
    if not top_features:
        return "No feature attribution is available for this model."

    from forecaster.features.registry import get_spec

    described: list[str] = []
    for name, _ in top_features[:n]:
        try:
            group = get_spec(name).group
            described.append(FEATURE_GROUP_PLAIN.get(group, name))
        except Exception:  # noqa: BLE001 -- derived column names may not be registered
            described.append(name.replace("_", " "))

    unique = list(dict.fromkeys(described))
    if len(unique) == 1:
        return f"This forecast was driven mainly by {unique[0]}."
    return f"This forecast was driven mainly by {', '.join(unique[:-1])} and {unique[-1]}."
