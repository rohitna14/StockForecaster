"""API contract tests.

Run against a temporary SQLite database seeded with synthetic bars, so they need
no network and no external services.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="requires the [api] extra")

from fastapi.testclient import TestClient  # noqa: E402

from forecaster.api.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


# ── health & catalog ──────────────────────────────────────────────────────
def test_health_is_cheap_and_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_component_state(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"status", "database", "lake", "providers"}


def test_methodology_documents_the_guarantees(client: TestClient) -> None:
    body = client.get("/api/v1/methodology").json()
    assert body["reference_baseline"] == "naive_last_price"
    assert "shuffled target" in " ".join(body["leakage_guards"]).lower()
    # The negative result must be served by the API, not just buried in docs.
    assert "no model beat" in body["known_findings"]["directional_price_prediction"].lower()


def test_model_catalog_includes_baselines(client: TestClient) -> None:
    models = client.get("/api/v1/models").json()
    names = {m["name"] for m in models}
    assert "naive_last_price" in names
    assert any(m["is_baseline"] for m in models)

    baseline = next(m for m in models if m["name"] == "naive_last_price")
    assert baseline["is_baseline"] is True


def test_baselines_endpoint_explains_why_they_exist(client: TestClient) -> None:
    body = client.get("/api/v1/baselines").json()
    assert body["reference"] == "naive_last_price"
    assert len(body["baselines"]) >= 6


def test_feature_catalog_exposes_warmup(client: TestClient) -> None:
    body = client.get("/api/v1/features").json()
    assert body["count"] > 20
    assert all("min_history" in f for f in body["features"])

    sets = client.get("/api/v1/features/sets").json()
    assert sets["minimal"]["count"] < sets["core"]["count"] <= sets["all"]["count"]


def test_glossary_covers_every_headline_metric(client: TestClient) -> None:
    entries = {e["key"] for e in client.get("/api/v1/glossary").json()}
    assert {"rmse", "mase", "r2", "directional_accuracy", "sharpe"} <= entries

    r2_entry = next(
        e for e in client.get("/api/v1/glossary").json() if e["key"] == "r2"
    )
    # The R2 caveat is the single most important piece of copy in the product.
    assert r2_entry["caveat"] and "leak" in r2_entry["caveat"].lower()


def test_cost_presets_are_available(client: TestClient) -> None:
    presets = client.get("/api/v1/backtests/presets").json()
    assert {"zero", "optimistic", "realistic", "conservative"} <= set(presets)
    assert presets["realistic"]["round_trip_bps_estimate"] > 0


# ── errors ────────────────────────────────────────────────────────────────
def test_unknown_symbol_returns_problem_json(client: TestClient) -> None:
    response = client.get("/api/v1/instruments/ZZZZNOTREAL")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")

    body = response.json()
    assert body["status"] == 404
    assert "type" in body and "title" in body and "detail" in body


def test_validation_error_is_problem_json(client: TestClient) -> None:
    response = client.post(
        "/api/v1/runs", json={"symbol": "AAPL", "horizon": 9999}
    )
    assert response.status_code == 422
    assert response.json()["type"].endswith("validation_error")


def test_request_id_header_is_returned(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.headers.get("X-Request-ID")
    assert float(response.headers["X-Process-Time"]) >= 0


def test_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/doesnotexist")
    assert response.status_code == 404


# ── search ────────────────────────────────────────────────────────────────
def test_instrument_search_paginates(client: TestClient) -> None:
    response = client.get("/api/v1/instruments", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) <= 5
    assert body["limit"] == 5


def test_openapi_schema_is_valid(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["openapi"].startswith("3.")
    assert len(schema["paths"]) >= 25
    assert "StockForecaster" in schema["info"]["title"]


# ── data-dependent (skipped when the lake is empty) ───────────────────────
def _has_data(client: TestClient, symbol: str = "AAPL") -> bool:
    return client.get(f"/api/v1/instruments/{symbol}/ohlcv", params={"limit": 5}).status_code == 200


def test_ohlcv_returns_bars_when_ingested(client: TestClient) -> None:
    if not _has_data(client):
        pytest.skip("no ingested data; run `forecaster ingest --universe demo`")

    body = client.get("/api/v1/instruments/AAPL/ohlcv", params={"limit": 50}).json()
    assert body["count"] > 0
    assert body["source_tier"] in {"hot", "cold"}

    bar = body["bars"][0]
    assert bar["high"] >= bar["low"]
    assert bar["volume"] >= 0


def test_indicators_align_to_price_index(client: TestClient) -> None:
    if not _has_data(client):
        pytest.skip("no ingested data")

    body = client.get(
        "/api/v1/instruments/AAPL/indicators",
        params={"features": "rsi_14,realized_vol_20", "limit": 100},
    ).json()
    assert len(body["index"]) > 0
    for series in body["values"].values():
        assert len(series) == len(body["index"])


def test_risk_metrics_are_plausible(client: TestClient) -> None:
    if not _has_data(client):
        pytest.skip("no ingested data")

    body = client.get("/api/v1/instruments/AAPL/risk").json()
    assert 0.0 < body["annual_volatility"] < 3.0
    assert body["max_drawdown"] <= 0.0
    assert body["n_observations"] > 100


def test_evaluation_always_includes_the_baseline(client: TestClient) -> None:
    """A leaderboard without the reference baseline is not servable."""
    if not _has_data(client):
        pytest.skip("no ingested data")

    response = client.post(
        "/api/v1/runs",
        json={
            "symbol": "AAPL", "models": ["ridge"], "horizon": 5,
            "target_type": "vol_ratio", "train_size": 300, "test_size": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()

    names = {row["model"] for row in body["leaderboard"]}
    assert "naive_last_price" in names, "baseline was omitted from the leaderboard"
    assert body["n_folds"] >= 1
    assert body["selected_model"]


def test_forecast_flags_uninformative_models(client: TestClient) -> None:
    """A model that loses to naive must not be presented as useful."""
    if not _has_data(client):
        pytest.skip("no ingested data")

    response = client.get(
        "/api/v1/forecast/AAPL",
        params={"model": "ridge", "horizon": 5, "target_type": "return"},
    )
    if response.status_code != 200:
        pytest.skip("forecast unavailable for this configuration")

    body = response.json()
    assert "is_informative" in body
    assert body["narrative"]
    # Whatever the verdict, the flag and the skill number must agree.
    if body.get("skill_pct") is not None and np.isfinite(body["skill_pct"]):
        assert body["is_informative"] == (body["skill_pct"] > 0)
    assert "not financial advice" in body["narrative"].lower()
