"""Methodology page — rendered from the API so claims live in one place."""

from __future__ import annotations

import api_client as api
import streamlit as st

st.set_page_config(
    page_title="Methodology · StockForecaster", page_icon="◈", layout="wide"
)

st.title("Methodology")
st.caption(
    "The modelling here is ordinary. The validation is the part worth defending."
)

try:
    methodology = api.get_methodology()
    baselines = api.get_baselines()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

# ── findings first ────────────────────────────────────────────────────────
st.subheader("What this project found")

findings = methodology["known_findings"]
left, right = st.columns(2)

with left:
    st.error(f"**Directional price prediction**\n\n{findings['directional_price_prediction']}", icon="✕")
with right:
    st.success(f"**Volatility forecasting**\n\n{findings['volatility_forecasting']}", icon="✓")

st.caption(
    "The negative result is given equal prominence deliberately. A project that "
    "only publishes what worked is not evidence of anything."
)

st.divider()

# ── validation ────────────────────────────────────────────────────────────
st.subheader("Validation scheme")
for key, value in methodology["validation"].items():
    st.markdown(f"**{key.replace('_', ' ').title()}** — {value}")

st.code(
    """fold 0  |=== train ===|·gap·|test|
fold 1        |=== train ===|·gap·|test|
fold 2              |=== train ===|·gap·|test|
                                            ────▶ time""",
    language=None,
)

with st.expander("Why the gap matters", expanded=True):
    st.markdown(
        """
With an *h*-bar forward target, the label at bar *t* is computed from the close
at *t + h*. Train through bar *T* and the labels at *T−h+1 … T* were all built
from bars **inside the test window** — the model has seen the test period's
prices through its own labels.

So the last *h* training labels are dropped. For horizon 5 with a test window
opening at bar 1000, training stops at bar 994, not 999. Every fold re-checks
this at runtime and raises rather than warns.
        """
    )

st.divider()

# ── baselines ─────────────────────────────────────────────────────────────
st.subheader("Baselines")
st.caption(baselines["note"])

st.dataframe(
    [
        {
            "Baseline": b["display_name"],
            "Name": b["name"],
            "Reference": "✓" if b["name"] == baselines["reference"] else "",
        }
        for b in baselines["baselines"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── guards ────────────────────────────────────────────────────────────────
st.subheader("Leakage guards")
for i, guard in enumerate(methodology["leakage_guards"], 1):
    st.markdown(f"{i}. {guard}")

st.info(
    "The shuffled-target test does not check one specific mistake — it checks "
    "the *conclusion*. If any stage of the pipeline leaks, the model finds the "
    "target through the leak and posts positive skill on pure noise. It is "
    "paired with a positive control on planted-signal data, so the suite cannot "
    "pass by the pipeline simply being broken.",
    icon="ⓘ",
)

st.divider()

# ── stats & intervals ─────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Significance testing")
    for test in methodology["significance_tests"]:
        st.markdown(f"- `{test}`")
with right:
    st.subheader("Prediction intervals")
    intervals = methodology["prediction_intervals"]
    st.markdown(
        f"- Method: **{intervals['method']}**\n"
        f"- Nominal coverage: **{intervals['nominal_coverage']:.0%}**\n"
        f"- An uncalibrated interval is worse than no interval, because it "
        f"invites a decision it cannot support."
    )

st.divider()

st.subheader("Known limitations")
st.caption("Listed because a methodology section that claims none is not credible.")
for limitation in methodology["limitations"]:
    st.markdown(f"- {limitation}")

st.divider()
st.caption(methodology["disclaimer"])
