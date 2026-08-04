"""Model and feature catalog."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from forecaster.api.schemas.common import ModelInfo

router = APIRouter(tags=["catalog"])


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    family: Annotated[str | None, Query(description="baseline|linear|tree|classical|deep")] = None,
) -> list[ModelInfo]:
    """Every registered model, baselines included and flagged as such."""
    from forecaster.models.registry import describe_all

    rows = describe_all()
    if family:
        rows = [r for r in rows if r["family"] == family]
    return [ModelInfo(**row) for row in rows]


@router.get("/models/sets")
async def list_model_sets() -> dict[str, list[str]]:
    """Named model bundles the UI can offer as presets."""
    from forecaster.models.registry import model_set

    out: dict[str, list[str]] = {}
    for name in ("baselines", "fast", "linear", "trees", "classical", "standard", "deep"):
        try:
            out[name] = model_set(name)
        except Exception:  # noqa: BLE001 -- optional extras may be absent
            continue
    return out


@router.get("/features")
async def list_features(
    group: str | None = None,
    include_optional: bool = True,
) -> dict[str, Any]:
    """Feature catalog with warm-up requirements."""
    from forecaster import features as F

    specs = F.list_features(group=group, include_optional=include_optional)
    return {
        "groups": F.groups(),
        "count": len(specs),
        "features": [
            {
                "name": spec.name,
                "group": spec.group,
                "description": spec.description,
                "min_history": spec.min_history,
                "optional": spec.optional,
            }
            for spec in specs
        ],
    }


@router.get("/features/sets")
async def list_feature_sets() -> dict[str, dict[str, Any]]:
    from forecaster import features as F

    out: dict[str, dict[str, Any]] = {}
    for name in ("minimal", "core", "full", "all"):
        names = F.feature_set(name)
        out[name] = {
            "count": len(names),
            "warmup_bars": F.required_history(names),
            "features": names,
        }
    return out


@router.get("/universes")
async def list_universes() -> list[dict[str, Any]]:
    """Named symbol universes available for batch evaluation."""
    from sqlalchemy import func, select

    from forecaster.db.models import Universe, UniverseMember
    from forecaster.db.session import session_scope

    async with session_scope() as session:
        stmt = (
            select(Universe.name, Universe.description, func.count(UniverseMember.instrument_id))
            .outerjoin(UniverseMember, UniverseMember.universe_id == Universe.id)
            .group_by(Universe.id, Universe.name, Universe.description)
            .order_by(Universe.name)
        )
        rows = (await session.execute(stmt)).all()

    return [
        {"name": name, "description": description, "members": int(count)}
        for name, description, count in rows
    ]
