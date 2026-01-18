from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class StatsSummary(BaseModel):
    datasets: int
    projects: int
    training_runs_total: int
    training_runs_by_status: Dict[str, int]
    model_versions: int
    deployments: int
    deployments_active: int

