"""Model leaderboard page."""

from __future__ import annotations

import api_client as api
import streamlit as st

st.set_page_config(page_title="Models · StockForecaster", page_icon="◈", layout="wide")

TARGETS = {
    "vol_ratio": (
        "Volatility",
        "Forecast whether the next period is calmer or more turbulent. "
        "This is where the measurable edge is.",
    ),
    "return": (
        "Direction",
        "Forecast the next period's return. Included so the null result is "
        "visible rather than hidden.",
    ),
}

DEFAULT_MODELS = ("ridge", "elastic_net", "random_forest", "gradient_boosting", "lightgbm")

st.title("Model comparison")
st.caption("Walk-forward with purged splits. Every run is computed live by the API.")

symbol = st.sidebar.text_input("Symbol", value="AAPL").strip().upper()
target = st.sidebar.selectbox(
    "Target", list(TARGETS), format_func=lambda k: TARGETS[k][0], index=0
)
horizon = st.sidebar.select_slider("Horizon (trading days)", options=[1, 5, 10, 21], value=5)

st.sidebar.caption(TARGETS[target][1])

if not symbol:
    st.info("Enter a symbol to begin.")
    st.stop()

with st.spinner(f"Running walk-forward evaluation for {symbol}…"):
    try:
        result = api.evaluate(symbol, DEFAULT_MODELS, horizon, target)
    except api.ApiError as exc:
        st.error(str(exc))
        st.stop()

meta = st.columns(4)
meta[0].metric("Folds", result["n_folds"])
meta[1].metric("Samples", f"{result['n_samples']:,}")
meta[2].metric("Features", result["n_features"])
meta[3].metric("Selected", result["selected_model"])

rows = result["leaderboard"]
best = next(
    (r for r in rows if not r["is_baseline"] and (r["rmse_skill_pct"] or -1) > 0), None
)

if best is None:
    st.warning(
        "**No model beat the baseline.** Every model below performed worse than "
        "assuming no change. For directional price targets this is the expected "
        "result, and it is shown rather than hidden.",
        icon="⚠",
    )
else:
    st.success(
        f"**{best['display_name'] or best['model']}** reduced forecast error by "
        f"**{best['rmse_skill_pct']:+.2f}%** versus the naive baseline.",
        icon="✓",
    )

st.subheader("Leaderboard")


def _fmt(value: float | None, spec: str = "{:.4f}") -> str:
    return "—" if value is None else spec.format(value)


table = []
for row in rows:
    table.append(
        {
            "Model": (row["display_name"] or row["model"])
            + ("  (baseline)" if row["is_baseline"] else ""),
            "Skill vs naive": _fmt(row["rmse_skill_pct"], "{:+.2f}%"),
            "MASE": _fmt(row["mase"], "{:.3f}"),
            "R²": _fmt(row["r2"], "{:.3f}"),
            "Hit rate": _fmt(row["hit_rate"], "{:.1%}") if target != "vol_ratio" else "—",
            "Base rate": _fmt(row["base_rate"], "{:.1%}") if target != "vol_ratio" else "—",
            "DM p-value": _fmt(row["dm_pvalue"], "{:.4f}"),
            "Coverage": _fmt(row["interval_coverage"], "{:.3f}"),
        }
    )

st.dataframe(table, use_container_width=True, hide_index=True)

st.caption(
    "Baselines run through the same code path as every other model and cannot be "
    "removed from this table. A leaderboard without one is not a result. "
    "**MASE** below 1.0 beats the naive forecast; above 1.0 loses to it."
)

with st.expander("How the selected model was chosen"):
    st.markdown(
        """
Selection runs on every fold **except the last**, and the held-out final fold
gives an unbiased estimate of its performance.

Picking the top row of this table and then reporting that row's score would be
selecting and reporting on the same data — with six models and a handful of
folds, the winner would mostly be whichever got luckiest. That is precisely the
mistake this project was built to avoid.
        """
    )

with st.expander("Metric glossary"):
    for entry in api.get_glossary():
        st.markdown(f"**{entry['label']}** — {entry['plain']}")
        if entry["caveat"]:
            st.caption(f"⚠ {entry['caveat']}")
