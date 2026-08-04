"""Evaluation runs: submit, poll, stream, and read results."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from forecaster.api.deps import OHLCVRepoDep, RunRepoDep, load_bars
from forecaster.api.schemas.common import (
    EvaluationRequest,
    EvaluationResponse,
    ExplanationResponse,
    FoldMetricRow,
    ForecastResponse,
    GlossaryEntry,
    LeaderboardRow,
    PredictionRow,
)
from forecaster.exceptions import RunNotFoundError
from forecaster.explain.narrative import (
    METRIC_GLOSSARY,
    describe_drivers,
    describe_forecast,
)
from forecaster.features.targets import TargetType
from forecaster.jobs.queue import get_queue
from forecaster.logging import get_logger
from forecaster.models.baselines import REFERENCE_BASELINE

log = get_logger(__name__)
router = APIRouter(tags=["runs"])


def _run_evaluation(
    request: EvaluationRequest, frame: pd.DataFrame, *, progress: Any = None
) -> Any:
    """Executed on a worker thread by the job queue."""
    from forecaster.validation.harness import EvaluationConfig, EvaluationHarness

    if progress is not None:
        progress.set_total(len(request.models) + 1)
        progress.update(f"Preparing features for {request.symbol}")

    config = EvaluationConfig(
        symbol=request.symbol.upper(),
        models=request.models,
        horizon=request.horizon,
        target_type=TargetType(request.target_type),
        feature_set=request.feature_set,
        train_size=request.train_size,
        test_size=request.test_size,
        embargo=request.embargo,
    )
    return EvaluationHarness(config).run(frame, progress=progress)


def _to_leaderboard(report: Any) -> list[LeaderboardRow]:
    frame = report.leaderboard("rmse")
    rows: list[LeaderboardRow] = []
    for _, row in frame.iterrows():
        payload = {
            k: (None if (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        rows.append(
            LeaderboardRow(
                model=payload.get("model", ""),
                display_name=payload.get("display_name"),
                family=payload.get("family", "unknown"),
                is_baseline=bool(payload.get("is_baseline", False)),
                rmse=payload.get("rmse"),
                mase=payload.get("mase"),
                r2=payload.get("r2"),
                hit_rate=payload.get("hit_rate"),
                base_rate=payload.get("base_rate"),
                rmse_skill_pct=payload.get("rmse_skill_pct"),
                directional_skill_pct=payload.get("directional_skill_pct"),
                dm_pvalue=payload.get("dm_pvalue"),
                interval_coverage=payload.get("interval_coverage"),
                sharpe=payload.get("sharpe"),
            )
        )
    return rows


@router.post("/runs", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
async def create_run(request: EvaluationRequest, ohlcv: OHLCVRepoDep) -> EvaluationResponse:
    """Run a walk-forward evaluation synchronously and return the leaderboard.

    Suitable for a handful of fast models. For sequence models or a large model
    set, use ``POST /runs/async`` and poll, or subscribe to ``/ws/runs/{id}``.
    """
    from forecaster.validation.harness import select_model_honestly
    from forecaster.validation.persistence import persist_report

    frame, _ = await load_bars(request.symbol, ohlcv)
    report = await asyncio.to_thread(_run_evaluation, request, frame)

    run_ids: dict[str, uuid.UUID] = {}
    try:
        run_ids = await persist_report(report)
    except Exception as exc:  # noqa: BLE001 -- results are still valid unpersisted
        log.warning("persist_failed", symbol=request.symbol, error=str(exc))

    return EvaluationResponse(
        symbol=request.symbol.upper(),
        horizon=request.horizon,
        target_type=request.target_type,
        n_folds=report.n_folds,
        n_samples=report.n_samples,
        n_features=len(report.feature_names),
        leaderboard=_to_leaderboard(report),
        selected_model=select_model_honestly(report),
        duration_seconds=round(report.duration_seconds, 2),
        run_ids=run_ids,
    )


@router.post("/runs/async", status_code=status.HTTP_202_ACCEPTED)
async def create_run_async(request: EvaluationRequest, ohlcv: OHLCVRepoDep) -> dict[str, Any]:
    """Queue an evaluation. Returns a job id to poll or stream."""
    frame, _ = await load_bars(request.symbol, ohlcv)
    job = get_queue().submit(
        "evaluation",
        _run_evaluation,
        request,
        frame,
        metadata={
            "symbol": request.symbol.upper(),
            "horizon": request.horizon,
            "models": request.models,
        },
    )
    return {
        **job.as_dict(),
        "poll": f"/api/v1/jobs/{job.id}",
        "stream": f"/api/v1/ws/jobs/{job.id}",
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = get_queue().get(job_id)
    if job is None:
        raise RunNotFoundError(f"Unknown job {job_id}", job_id=job_id)

    payload = job.as_dict()
    if job.state.value == "succeeded" and job.result is not None:
        from forecaster.validation.harness import select_model_honestly

        payload["result"] = {
            "leaderboard": [row.model_dump() for row in _to_leaderboard(job.result)],
            "n_folds": job.result.n_folds,
            "n_samples": job.result.n_samples,
            "selected_model": select_model_honestly(job.result),
        }
    return payload


@router.get("/jobs")
async def list_jobs(limit: Annotated[int, Query(ge=1, le=200)] = 25) -> list[dict[str, Any]]:
    return [job.as_dict() for job in get_queue().list_jobs(limit=limit)]


@router.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    """Push job progress as it changes.

    Sends on every state change rather than on a timer, so a fast job does not
    wait for the next tick and a slow one does not spam the socket.
    """
    await websocket.accept()
    queue = get_queue()

    job = queue.get(job_id)
    if job is None:
        await websocket.send_json({"error": "unknown job", "job_id": job_id})
        await websocket.close()
        return

    try:
        await websocket.send_json(job.as_dict())
        while not job.is_terminal:
            updated = await queue.wait_for_update(job_id, timeout=30.0)
            if updated is None:
                break
            job = updated
            await websocket.send_json(job.as_dict())
        await websocket.close()
    except WebSocketDisconnect:
        log.debug("ws_client_disconnected", job_id=job_id)


# ── persisted run reads ───────────────────────────────────────────────────
@router.get("/runs/{run_id}/folds", response_model=list[FoldMetricRow])
async def get_run_folds(run_id: uuid.UUID, runs: RunRepoDep) -> list[FoldMetricRow]:
    """Per-fold metrics -- the audit trail behind any aggregate number."""
    folds = await runs.folds_for_run(run_id)
    if not folds:
        raise RunNotFoundError(f"No folds for run {run_id}", run_id=str(run_id))
    return [FoldMetricRow.model_validate(f) for f in folds]


@router.get("/runs/{run_id}/predictions", response_model=list[PredictionRow])
async def get_run_predictions(
    run_id: uuid.UUID,
    runs: RunRepoDep,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> list[PredictionRow]:
    rows = await runs.predictions_for_run(run_id, limit=limit)
    return [PredictionRow.model_validate(r) for r in rows]


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: uuid.UUID, runs: RunRepoDep) -> dict[str, Any]:
    aggregate = await runs.aggregate_for_run(run_id)
    run = await runs.get(run_id)
    if run is None:
        raise RunNotFoundError(f"Unknown run {run_id}", run_id=str(run_id))

    return {
        "run_id": str(run_id),
        "model_name": run.model_name,
        "horizon_days": run.horizon_days,
        "target_type": run.target_type,
        "status": run.status,
        "git_sha": run.git_sha,
        "random_seed": run.random_seed,
        "split_config": run.split_config,
        "metrics": aggregate.metrics if aggregate else {},
        "skill_vs_naive": aggregate.skill_vs_naive if aggregate else None,
        "dm_stat": aggregate.dm_stat if aggregate else None,
        "dm_pvalue": aggregate.dm_pvalue if aggregate else None,
        "n_folds": aggregate.n_folds if aggregate else 0,
    }


@router.get("/runs/{run_id}/explain", response_model=ExplanationResponse)
async def explain_run(run_id: uuid.UUID, runs: RunRepoDep) -> ExplanationResponse:
    """Feature attribution for a persisted run."""
    run = await runs.get(run_id)
    if run is None:
        raise RunNotFoundError(f"Unknown run {run_id}", run_id=str(run_id))

    importance = await runs.importance_for_run(run_id)
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:15]

    return ExplanationResponse(
        run_id=run_id,
        model=run.model_name,
        method="native",
        top_features=[{"feature": name, "importance": value} for name, value in top],
        narrative=(
            f"{run.model_name} trained on {run.horizon_days}-day-ahead {run.target_type} targets."
        ),
        drivers=describe_drivers(top),
    )


# ── forecast + glossary ───────────────────────────────────────────────────
@router.get("/forecast/{symbol}")
async def get_forecast(
    symbol: str,
    ohlcv: OHLCVRepoDep,
    model: str = "lightgbm",
    horizon: Annotated[int, Query(ge=1, le=60)] = 5,
    target_type: str = "vol_ratio",
) -> dict[str, Any]:
    """Forecast for **any** listed symbol, however new.

    The method is chosen from the history available, and always reported:

    * ``walk_forward`` / ``adaptive`` -- enough history to validate on the stock
      itself. Skill is measured on its own out-of-sample folds.
    * ``transfer`` -- a recent listing. A pooled model trained on other
      companies is applied; the stock's own history is used only to compute the
      latest feature row. Expected accuracy comes from leave-one-symbol-out
      validation on stocks the model had never seen.

    ``confidence`` reflects the strength of the *evidence*, not how good the
    headline number looks.
    """
    from forecaster.validation.coldstart import (
        Method,
        choose_method,
        confidence_for,
        forecast_cold_start,
    )
    from forecaster.validation.harness import EvaluationConfig, EvaluationHarness

    frame, tier = await load_bars(symbol, ohlcv)
    method = choose_method(len(frame))

    # ── recent listing: pooled cross-sectional model ──────────────────────
    if method is Method.TRANSFER:
        cold = await forecast_cold_start(symbol.upper(), frame, horizon=horizon)
        payload = cold.as_dict()
        payload.update(
            {
                "model": "pooled_lightgbm",
                "target_type": target_type,
                "source_tier": tier,
                "is_informative": bool(
                    cold.expected_skill_pct and cold.expected_skill_pct > 0
                ),
                "narrative": describe_forecast(
                    symbol=symbol.upper(),
                    model_name="a pooled model",
                    prediction=cold.prediction or 0.0,
                    lower=cold.lower,
                    upper=cold.upper,
                    horizon=horizon,
                    skill_pct=cold.expected_skill_pct,
                    dm_pvalue=None,
                    target_type=target_type,
                ),
            }
        )
        return payload

    # ── enough history to validate on the stock itself ────────────────────
    config = EvaluationConfig(
        symbol=symbol.upper(),
        models=[model],
        horizon=horizon,
        target_type=TargetType(target_type),
        max_folds=3,  # keep the interactive path responsive
    )
    report = await asyncio.to_thread(lambda: EvaluationHarness(config).run(frame))

    result = report.results.get(model)
    if result is None or not result.succeeded:
        raise RunNotFoundError(
            f"Model {model} produced no result for {symbol.upper()}", model=model
        )

    last_fold = result.folds[-1]
    prediction = float(last_fold.y_pred[-1])
    lower = float(last_fold.y_lower[-1]) if last_fold.y_lower is not None else None
    upper = float(last_fold.y_upper[-1]) if last_fold.y_upper is not None else None
    skill_pct = result.skill.get("rmse", {}).get("improvement_pct")
    dm_p = result.dm_test.get("dm_pvalue")

    return {
        "symbol": symbol.upper(),
        "model": model,
        "method": method.value,
        "confidence": confidence_for(method, len(frame), skill_pct).value,
        "horizon": horizon,
        "target_type": target_type,
        "as_of": last_fold.dates[-1].date().isoformat(),
        "prediction": prediction,
        "lower": lower,
        "upper": upper,
        "n_bars": len(frame),
        "expected_skill_pct": skill_pct,
        "dm_pvalue": dm_p,
        "source_tier": tier,
        "basis": f"Walk-forward on {symbol.upper()}'s own history "
                 f"({report.n_folds} folds, {report.n_samples:,} samples)",
        "caveats": [],
        "narrative": describe_forecast(
            symbol=symbol.upper(), model_name=model, prediction=prediction,
            lower=lower, upper=upper, horizon=horizon,
            skill_pct=skill_pct, dm_pvalue=dm_p, target_type=target_type,
        ),
        "is_informative": bool(skill_pct is not None and skill_pct > 0),
    }


@router.get("/glossary", response_model=list[GlossaryEntry])
async def glossary() -> list[GlossaryEntry]:
    """Plain-English metric definitions, powering the UI's explain mode."""
    return [GlossaryEntry(key=key, **entry.as_dict()) for key, entry in METRIC_GLOSSARY.items()]


@router.get("/baselines")
async def baselines() -> dict[str, Any]:
    """The baselines every leaderboard is measured against."""
    from forecaster.models.baselines import BASELINES

    return {
        "reference": REFERENCE_BASELINE,
        "note": (
            "Baselines run through the same code path as every other model and "
            "cannot be filtered out of a leaderboard. A result without one is "
            "not a result."
        ),
        "baselines": [
            {"name": name, "display_name": cls.display_name, "is_classifier": cls.is_classifier}
            for name, cls in BASELINES.items()
        ],
    }
