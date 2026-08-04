"""Bridge from an in-memory :class:`EvaluationReport` to database rows."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import numpy as np

from forecaster.db.models import RunStatus
from forecaster.db.repositories.instruments import InstrumentRepository
from forecaster.db.repositories.runs import RunRepository
from forecaster.db.session import session_scope
from forecaster.logging import get_logger
from forecaster.validation.harness import EvaluationReport, ModelResult

log = get_logger(__name__)


async def persist_report(report: EvaluationReport) -> dict[str, uuid.UUID]:
    """Write every model's results as its own ``model_run``.

    One run per (symbol, model, horizon) rather than one per evaluation: it
    makes the leaderboard a simple query, lets a single model be re-run without
    invalidating its peers, and keeps ``skill_vs_naive`` attached to the row it
    describes.
    """
    cfg = report.config
    run_ids: dict[str, uuid.UUID] = {}

    async with session_scope() as session:
        instruments = InstrumentRepository(session)
        instrument = await instruments.get_by_symbol(cfg.symbol)
        instrument_id = instrument.id if instrument else None

        runs = RunRepository(session)

        for model_name, result in report.results.items():
            if not result.succeeded:
                log.debug("skipping_failed_model", model=model_name, error=result.error)
                continue

            run = await runs.create(
                instrument_id=instrument_id,
                model_name=model_name,
                horizon_days=cfg.horizon,
                target_type=str(cfg.target_type),
                hyperparams={},
                split_config=cfg.splitter().describe(),
                random_seed=cfg.random_seed,
            )
            await runs.mark_running(run.id)

            await runs.save_folds(run.id, _fold_rows(result))
            await runs.save_aggregate(
                run.id,
                metrics=result.aggregate,
                skill_vs_naive=_skill(result, "rmse"),
                dm_stat=result.dm_test.get("dm_stat"),
                dm_pvalue=result.dm_test.get("dm_pvalue"),
                n_folds=len(result.folds),
            )

            if instrument_id is not None:
                await runs.save_predictions(
                    run.id, instrument_id, _prediction_rows(result, cfg.horizon)
                )
            if result.feature_importance:
                await runs.save_feature_importance(run.id, result.feature_importance)

            await runs.mark_finished(run.id, status=RunStatus.SUCCEEDED)
            run_ids[model_name] = run.id

    log.info("report_persisted", symbol=cfg.symbol, runs=len(run_ids))
    return run_ids


def _skill(result: ModelResult, key: str) -> float | None:
    value = result.skill.get(key, {}).get("skill")
    return float(value) if value is not None and np.isfinite(value) else None


def _fold_rows(result: ModelResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in result.folds:
        m = fold.metrics
        rows.append(
            {
                "fold_index": fold.fold_index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "mae": _f(m.mae),
                "rmse": _f(m.rmse),
                "mase": _f(m.mase),
                "r2": _f(m.r2),
                "directional_accuracy": _f(m.directional_accuracy),
                "mcc": _f(m.mcc),
                "roc_auc": _f(m.roc_auc),
                "brier": _f(m.brier),
                "skill_vs_naive": _f(m.skill_vs_naive),
                "sharpe": _f(m.sharpe),
            }
        )
    return rows


def _prediction_rows(result: ModelResult, horizon: int) -> list[dict[str, Any]]:
    """One row per out-of-sample prediction.

    ``as_of_date`` is the bar the forecast was made from; ``target_date`` is the
    bar it describes. Storing both is what makes lookahead auditable with a SQL
    query -- ``WHERE target_date <= as_of_date`` should return nothing, and the
    DB CHECK constraint enforces it.
    """
    rows: list[dict[str, Any]] = []
    for fold in result.folds:
        for i, as_of in enumerate(fold.dates):
            as_of_date = as_of.date() if hasattr(as_of, "date") else as_of
            # Calendar-day approximation of the target bar; the realize job
            # snaps it to the nearest actual trading day.
            target_date = as_of_date + dt.timedelta(days=_calendar_days(horizon))
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "target_date": target_date,
                    "fold_index": fold.fold_index,
                    "is_out_of_sample": True,
                    "y_pred": float(fold.y_pred[i]),
                    "y_pred_lower": _f(fold.y_lower[i]) if fold.y_lower is not None else None,
                    "y_pred_upper": _f(fold.y_upper[i]) if fold.y_upper is not None else None,
                    "prob_up": (
                        _f(fold.probabilities[i]) if fold.probabilities is not None else None
                    ),
                    "y_true": _f(fold.y_true[i]),
                }
            )
    return rows


def _calendar_days(horizon_bars: int) -> int:
    """Trading bars -> calendar days (5 bars/week)."""
    return max(1, int(round(horizon_bars * 7 / 5)))


def _f(value: Any) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None
