from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.dataset import Dataset
from train_platform.models.deployment import Deployment
from train_platform.models.model_registry import ModelVersion
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.schemas.v2.stats import StatsSummary


router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/summary", response_model=StatsSummary)
def get_summary(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).count()
    projects = db.query(Project).count()

    training_runs_total = db.query(TrainingRun).count()
    rows = db.query(TrainingRun.status, func.count(TrainingRun.run_id)).group_by(TrainingRun.status).all()
    by_status = {getattr(st, "value", str(st)): int(cnt) for st, cnt in rows}

    model_versions = db.query(ModelVersion).count()
    deployments = db.query(Deployment).count()
    deployments_active = db.query(Deployment).filter(Deployment.is_active == True).count()  # noqa: E712

    return StatsSummary(
        datasets=int(datasets),
        projects=int(projects),
        training_runs_total=int(training_runs_total),
        training_runs_by_status=by_status,
        model_versions=int(model_versions),
        deployments=int(deployments),
        deployments_active=int(deployments_active),
    )

