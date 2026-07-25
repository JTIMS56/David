"""
ModelRegistry — records every DHJ/Dirac prediction call with its exact parameters
and git commit hash. Links model runs to agent decisions for full auditability.

Usage:
    from services.model_registry import model_registry
    run_id = await model_registry.record(model="DHJ", pair="EURUSD", spot=1.085, ...)
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

from database import AsyncSessionLocal
from models.orm import ModelRun

logger = logging.getLogger("popper.model_registry")

# Cache the git commit hash for the lifetime of this process
_GIT_COMMIT: Optional[str] = None


def _get_git_commit() -> Optional[str]:
    global _GIT_COMMIT
    if _GIT_COMMIT is not None:
        return _GIT_COMMIT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            _GIT_COMMIT = result.stdout.strip()
    except Exception:
        pass
    return _GIT_COMMIT


class ModelRegistryService:

    async def record(
        self,
        model: str,
        pair: str,
        spot: float,
        horizon_days: float,
        params: Optional[dict] = None,
        mean_model: Optional[float] = None,
        call_model: Optional[float] = None,
        call_bs: Optional[float] = None,
        chiral_charge: Optional[float] = None,
        n_steps: Optional[int] = None,
        mass_loss_fraction: Optional[float] = None,
        negative_count: Optional[int] = None,
        agent_decision_id: Optional[int] = None,
    ) -> int:
        """Persist a model run. Returns the new ModelRun.id."""
        try:
            async with AsyncSessionLocal() as db:
                run = ModelRun(
                    model=model,
                    pair=pair,
                    spot=spot,
                    horizon_days=horizon_days,
                    params=json.dumps(params) if params else None,
                    git_commit=_get_git_commit(),
                    mean_model=mean_model,
                    call_model=call_model,
                    call_bs=call_bs,
                    chiral_charge=chiral_charge,
                    n_steps=n_steps,
                    mass_loss_fraction=mass_loss_fraction,
                    negative_count=negative_count,
                    agent_decision_id=agent_decision_id,
                )
                db.add(run)
                await db.commit()
                await db.refresh(run)
                return run.id
        except Exception:
            logger.exception("Failed to record model run (model=%s pair=%s)", model, pair)
            return -1

    async def get_runs(
        self,
        model: Optional[str] = None,
        pair: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        from sqlalchemy import select, desc
        async with AsyncSessionLocal() as db:
            q = select(ModelRun).order_by(desc(ModelRun.timestamp))
            if model:
                q = q.where(ModelRun.model == model.upper())
            if pair:
                q = q.where(ModelRun.pair == pair.upper())
            result = await db.execute(q.limit(limit))
            rows = result.scalars().all()
        return [
            {
                "id":                 r.id,
                "timestamp":          r.timestamp.isoformat(),
                "model":              r.model,
                "pair":               r.pair,
                "spot":               r.spot,
                "horizon_days":       r.horizon_days,
                "params":             json.loads(r.params) if r.params else None,
                "git_commit":         r.git_commit,
                "mean_model":         r.mean_model,
                "call_model":         r.call_model,
                "call_bs":            r.call_bs,
                "chiral_charge":      r.chiral_charge,
                "n_steps":            r.n_steps,
                "mass_loss_fraction": r.mass_loss_fraction,
                "negative_count":     r.negative_count,
                "agent_decision_id":  r.agent_decision_id,
            }
            for r in rows
        ]


model_registry = ModelRegistryService()
