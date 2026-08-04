"""HTTP client for the StockForecaster API.

This is the *only* module in the Streamlit app allowed to know anything about
where data comes from. Nothing here imports pandas, scikit-learn or the
``forecaster`` package — the whole point of this client is to demonstrate that
the API is genuinely client-agnostic, and importing the domain layer would
quietly make that false.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("FORECASTER_API_URL", "http://127.0.0.1:8000/api/v1")
TIMEOUT = float(os.environ.get("FORECASTER_API_TIMEOUT", "120"))


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0, problem: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.problem = problem or {}

    @property
    def is_missing_data(self) -> bool:
        return self.status == 404


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, f"{API_URL}{path}", **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(
            f"Cannot reach the API at {API_URL}. Is the backend running?"
        ) from exc

    if response.status_code >= 400:
        problem: dict[str, Any] = {}
        try:
            problem = response.json()
        except Exception:  # noqa: BLE001 -- non-JSON error body
            pass
        raise ApiError(
            problem.get("detail", f"{response.status_code} {response.reason_phrase}"),
            response.status_code,
            problem,
        )

    return response.json()


# Cached reads. TTLs are short enough that a fresh ingest shows up quickly and
# long enough that flipping between pages does not re-hit the API.
@st.cache_data(ttl=300, show_spinner=False)
def get_summary(symbol: str) -> dict[str, Any]:
    return _request("GET", f"/instruments/{symbol}/summary")


@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv(symbol: str, limit: int = 500) -> dict[str, Any]:
    return _request("GET", f"/instruments/{symbol}/ohlcv", params={"limit": limit})


@st.cache_data(ttl=300, show_spinner=False)
def get_risk(symbol: str) -> dict[str, Any]:
    return _request("GET", f"/instruments/{symbol}/risk")


@st.cache_data(ttl=600, show_spinner=False)
def search_instruments(query: str, limit: int = 20) -> dict[str, Any]:
    return _request("GET", "/instruments", params={"q": query, "limit": limit})


@st.cache_data(ttl=3600, show_spinner=False)
def get_models() -> list[dict[str, Any]]:
    return _request("GET", "/models")


@st.cache_data(ttl=3600, show_spinner=False)
def get_glossary() -> list[dict[str, Any]]:
    return _request("GET", "/glossary")


@st.cache_data(ttl=3600, show_spinner=False)
def get_methodology() -> dict[str, Any]:
    return _request("GET", "/methodology")


@st.cache_data(ttl=3600, show_spinner=False)
def get_baselines() -> dict[str, Any]:
    return _request("GET", "/baselines")


@st.cache_data(ttl=900, show_spinner=False)
def evaluate(
    symbol: str, models: tuple[str, ...], horizon: int, target_type: str
) -> dict[str, Any]:
    return _request(
        "POST",
        "/runs",
        json={
            "symbol": symbol,
            "models": list(models),
            "horizon": horizon,
            "target_type": target_type,
            "train_size": 504,
            "test_size": 63,
        },
    )


@st.cache_data(ttl=900, show_spinner=False)
def backtest(symbol: str, model: str, cost_preset: str) -> dict[str, Any]:
    return _request(
        "POST",
        "/backtests",
        json={
            "symbol": symbol,
            "model": model,
            "horizon": 1,
            "cost_preset": cost_preset,
            "sizing": "vol_target",
        },
    )


def health() -> bool:
    try:
        return _request("GET", "/health").get("status") == "ok"
    except ApiError:
        return False
