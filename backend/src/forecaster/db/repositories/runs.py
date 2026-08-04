"""Model-run persistence.

Every published number is traceable back to a row here: the model, its
hyperparameters, the exact walk-forward configuration, the git SHA, the random
seed, and the per-fold metrics that were averaged to produce it.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from sqlalchemy import delete, desc, select

from forecaster.db.models import (
    FeatureImportance,
    FoldMetric,
    Instrument,
    ModelRun,
    Prediction,
    RunMetric,
    RunStatus,
)
from forecaster.db.repositories.base import Repository
from forecaster.db.types import utcnow
from forecaster.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def current_git_sha() -> str | None:
    """Short SHA of the working tree, recorded on every run for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class RunRepository(Repository):
    async def create(
        self,
        *,
        instrument_id: int | None,
        model_name: str,
        horizon_days: int,
        target_type: str,
        hyperparams: dict[str, Any],
        split_config: dict[str, Any],
        model_version: str = "1",
        feature_set_id: int | None = None,
        universe_id: int | None = None,
        random_seed: int | None = None,
    ) -> ModelRun:
        run = ModelRun(
            id=uuid.uuid4(),
            instrument_id=instrument_id,
            universe_id=universe_id,
            feature_set_id=feature_set_id,
            model_name=model_name,
            model_version=model_version,
            horizon_days=horizon_days,
            target_type=target_type,
            hyperparams=hyperparams,
            split_config=split_config,
            status=RunStatus.QUEUED,
            git_sha=current_git_sha(),
            random_seed=random_seed,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def mark_running(self, run_id: uuid.UUID) -> None:
        run = await self.session.get(ModelRun, run_id)
        if run:
            run.status = RunStatus.RUNNING
            run.started_at = utcnow()

    async def mark_finished(
        self, run_id: uuid.UUID, *, status: RunStatus, error: str | None = None
    ) -> None:
        run = await self.session.get(ModelRun, run_id)
        if not run:
            return
        run.status = status
        run.finished_at = utcnow()
        run.error_message = error
        run.progress = 1.0 if status is RunStatus.SUCCEEDED else run.progress
        if run.started_at:
            run.duration_seconds = (run.finished_at - run.started_at).total_seconds()

    async def update_progress(self, run_id: uuid.UUID, progress: float) -> None:
        run = await self.session.get(ModelRun, run_id)
        if run:
            run.progress = float(np.clip(progress, 0.0, 1.0))

    async def save_folds(self, run_id: uuid.UUID, folds: list[dict[str, Any]]) -> int:
        if not folds:
            return 0
        rows = [{"run_id": run_id, **fold} for fold in folds]
        return await self.upsert(
            FoldMetric, rows, conflict_cols=["run_id", "fold_index"]
        )

    async def save_aggregate(
        self,
        run_id: uuid.UUID,
        *,
        metrics: dict[str, Any],
        skill_vs_naive: float | None,
        dm_stat: float | None,
        dm_pvalue: float | None,
        n_folds: int,
    ) -> None:
        await self.session.execute(delete(RunMetric).where(RunMetric.run_id == run_id))
        self.session.add(
            RunMetric(
                run_id=run_id,
                metrics=_json_safe(metrics),
                skill_vs_naive=_finite(skill_vs_naive),
                dm_stat=_finite(dm_stat),
                dm_pvalue=_finite(dm_pvalue),
                n_folds=n_folds,
            )
        )

    async def save_predictions(
        self, run_id: uuid.UUID, instrument_id: int, rows: list[dict[str, Any]]
    ) -> int:
        if not rows:
            return 0
        payload = [{"run_id": run_id, "instrument_id": instrument_id, **row} for row in rows]
        return await self.upsert(
            Prediction,
            payload,
            conflict_cols=["run_id", "instrument_id", "as_of_date", "target_date"],
        )

    async def save_feature_importance(
        self, run_id: uuid.UUID, importances: dict[str, float], method: str = "native"
    ) -> int:
        if not importances:
            return 0
        rows = [
            {"run_id": run_id, "feature_name": name, "method": method,
             "importance": float(value)}
            for name, value in importances.items()
            if np.isfinite(value)
        ]
        return await self.upsert(
            FeatureImportance, rows, conflict_cols=["run_id", "feature_name", "method"]
        )

    # ── queries ───────────────────────────────────────────────────────────
    async def get(self, run_id: uuid.UUID) -> ModelRun | None:
        return await self.session.get(ModelRun, run_id)

    async def latest_for_symbol(
        self, symbol: str, *, model_name: str | None = None,
        horizon: int | None = None, limit: int = 20,
    ) -> list[ModelRun]:
        stmt = (
            select(ModelRun)
            .join(Instrument, Instrument.id == ModelRun.instrument_id)
            .where(Instrument.symbol == symbol.upper(), ModelRun.status == RunStatus.SUCCEEDED)
            .order_by(desc(ModelRun.created_at))
            .limit(limit)
        )
        if model_name:
            stmt = stmt.where(ModelRun.model_name == model_name)
        if horizon:
            stmt = stmt.where(ModelRun.horizon_days == horizon)
        return list((await self.session.execute(stmt)).scalars().all())

    async def folds_for_run(self, run_id: uuid.UUID) -> list[FoldMetric]:
        stmt = (
            select(FoldMetric)
            .where(FoldMetric.run_id == run_id)
            .order_by(FoldMetric.fold_index)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def aggregate_for_run(self, run_id: uuid.UUID) -> RunMetric | None:
        return await self.session.get(RunMetric, run_id)

    async def predictions_for_run(
        self, run_id: uuid.UUID, *, oos_only: bool = True, limit: int | None = None
    ) -> list[Prediction]:
        stmt = select(Prediction).where(Prediction.run_id == run_id)
        if oos_only:
            stmt = stmt.where(Prediction.is_out_of_sample.is_(True))
        stmt = stmt.order_by(Prediction.target_date)
        if limit:
            stmt = stmt.limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def importance_for_run(self, run_id: uuid.UUID) -> dict[str, float]:
        stmt = (
            select(FeatureImportance.feature_name, FeatureImportance.importance)
            .where(FeatureImportance.run_id == run_id)
            .order_by(desc(FeatureImportance.importance))
        )
        return {name: float(value) for name, value in (await self.session.execute(stmt)).all()}

    async def realize_predictions(
        self, instrument_id: int, realized: dict[dt.date, float]
    ) -> int:
        """Backfill ``y_true`` on predictions whose target bar has now occurred.

        This is what turns the ``predictions`` table into a genuine live
        out-of-sample track record: forecasts are written before the fact and
        scored after, with no opportunity to revise them.
        """
        if not realized:
            return 0
        stmt = select(Prediction).where(
            Prediction.instrument_id == instrument_id,
            Prediction.y_true.is_(None),
            Prediction.target_date.in_(list(realized)),
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        for row in rows:
            row.y_true = float(realized[row.target_date])
        log.info("predictions_realized", instrument_id=instrument_id, count=len(rows))
        return len(rows)


def _finite(value: float | None) -> float | None:
    """NaN and inf are not valid JSON and break Postgres float columns."""
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so JSONB accepts the payload."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    return obj
