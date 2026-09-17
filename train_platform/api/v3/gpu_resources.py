from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import get_db
from train_platform.domains.training.resources.queries import (
    get_gpu_resources_overview,
    get_gpu_workers_overview,
    get_training_run_resource_status,
)
from train_platform.schemas.v3.gpu_resources import (
    GpuResourcesResponse,
    GpuWorkersResponse,
    TrainingRunResourcesResponse,
)


router = APIRouter(tags=["gpu-resources"])


@router.get("/gpu-resources", response_model=GpuResourcesResponse)
def get_gpu_resources(
    node_id: str | None = Query(None),
    gpu_uuid: str | None = Query(None),
    db: Session = Depends(get_db),
):
    payload = get_gpu_resources_overview(
        db,
        node_id=node_id,
        gpu_uuid=gpu_uuid,
        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
    )
    return GpuResourcesResponse(**payload)


@router.get("/gpu-workers", response_model=GpuWorkersResponse)
def get_gpu_workers(db: Session = Depends(get_db)):
    payload = get_gpu_workers_overview(
        db,
        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
    )
    return GpuWorkersResponse(**payload)


@router.get(
    "/training-runs/{run_id}/resources",
    response_model=TrainingRunResourcesResponse,
)
def get_training_run_resources(
    run_id: str,
    db: Session = Depends(get_db),
):
    payload = get_training_run_resource_status(db, run_id)
    return TrainingRunResourcesResponse(**payload)
