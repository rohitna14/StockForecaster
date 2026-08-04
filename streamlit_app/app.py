"""StockForecaster — Streamlit client.

A deliberately thin client over the FastAPI service. It exists to demonstrate
that the API is client-agnostic: the Next.js app and this share one backend and
one set of guarantees, and neither contains any forecasting logic.

Run with:  streamlit run app.py
Requires:  FORECASTER_API_URL pointing at a running API (default localhost:8000)
"""

from __future__ import annotations

import api_client as api
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="StockForecaster",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS. Deliberately restrained: the previous version of this project
# had an animated hue-rotating gradient background, six competing colour
# schemes and glowing gold "champion" cards. Restraint is what reads as
# credible in a tool that reports negative results.
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .stMetric {background: #121417; border: 1px solid #22262D;
                 border-radius: 12px; padding: 14px;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;
                                     font-size: 1.5rem;}
      [data-testid="stMetricLabel"] {text-transform: uppercase;
                                     letter-spacing: 0.05em; font-size: 0.7rem;}
      code {color: #6366F1;}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> str:
    st.title("◈ StockForecaster")
    st.caption(
        "Walk-forward validated forecasting · purged splits · honest baselines · "
        "published negative results"
    )

    with st.sidebar:
        st.subheader("Ticker")
        symbol = st.text_input("Symbol", value="AAPL", key="symbol").strip().upper()

        st.divider()
        st.caption(f"API: `{api.API_URL}`")
        if api.health():
            st.success("API connected", icon="✅")
        else:
            st.error("API unreachable", icon="❌")
            st.caption(
                "Start it with:\n\n"
                "`uvicorn forecaster.api.main:app --reload`\n\n"
                "from the `backend/` directory."
            )

        st.divider()
        st.caption(
            "**This client contains no forecasting logic.** Every number is "
            "computed by the API and rendered here — the same backend serving "
            "the Next.js frontend."
        )

    return symbol


def main() -> None:
    symbol = render_header()
    if not symbol:
        st.info("Enter a ticker symbol in the sidebar to begin.")
        return

    try:
        summary = api.get_summary(symbol)
    except api.ApiError as exc:
        if exc.is_missing_data:
            st.warning(f"**No data for {symbol} yet.**")
            st.caption(
                f"This ticker hasn't been ingested. Run "
                f"`forecaster ingest {symbol} --hot` to fetch five years of history."
            )
        else:
            st.error(str(exc))
        return

    # ── snapshot ──────────────────────────────────────────────────────────
    columns = st.columns(5)
    columns[0].metric(
        "Last close",
        f"${summary['last_close']:,.2f}",
        f"{summary['change_1d'] * 100:+.2f}%",
    )
    for column, (label, key) in zip(
        columns[1:4],
        [("1 week", "change_1w"), ("1 month", "change_1m"), ("1 year", "change_1y")],
        strict=False,
    ):
        value = summary.get(key)
        column.metric(label, "—" if value is None else f"{value * 100:+.2f}%")
    columns[4].metric("Bars", f"{summary['n_bars']:,}")

    # ── price chart ───────────────────────────────────────────────────────
    st.subheader("Price history")
    bars = api.get_ohlcv(symbol, limit=500)["bars"]

    figure = go.Figure(
        go.Candlestick(
            x=[bar["ts"] for bar in bars],
            open=[bar["open"] for bar in bars],
            high=[bar["high"] for bar in bars],
            low=[bar["low"] for bar in bars],
            close=[bar["close"] for bar in bars],
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            name=symbol,
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A0B0D",
        font={"color": "#9AA3AF", "size": 11},
        xaxis={"gridcolor": "#1A1D23", "rangeslider": {"visible": False}},
        yaxis={"gridcolor": "#1A1D23"},
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Split- and dividend-adjusted. The whole OHLC bar is back-adjusted and "
        "volume inverted, so gap and range features stay correct across splits."
    )

    # ── risk ──────────────────────────────────────────────────────────────
    try:
        risk = api.get_risk(symbol)
    except api.ApiError:
        risk = None

    if risk:
        st.subheader("Realised risk")
        risk_columns = st.columns(5)
        fields = [
            ("Volatility", risk["annual_volatility"], "{:.1%}"),
            ("Sharpe", risk["sharpe"], "{:.2f}"),
            ("Sortino", risk["sortino"], "{:.2f}"),
            ("Max drawdown", risk["max_drawdown"], "{:.1%}"),
            ("VaR 95%", risk["var_95"], "{:.2%}"),
        ]
        for column, (label, value, fmt) in zip(risk_columns, fields, strict=True):
            column.metric(label, "—" if value is None else fmt.format(value))

    st.divider()
    st.info(
        "**Before you read anything into a forecast:** on this project's own "
        "testing, no model beat a naive baseline at predicting *direction* for "
        "any symbol or horizon. The measurable edge is in forecasting "
        "*volatility*. See the Forecast and Methodology pages.",
        icon="ℹ️",
    )


if __name__ == "__main__":
    main()
