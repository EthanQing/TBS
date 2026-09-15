from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import get_db
from train_platform.domains.training.resources.queries import (
    list_gpu_resources,
    list_gpu_workers,
)
from train_platform.domains.training.runs.service import TrainingRunService
from train_platform.schemas.v3.gpu_resources import (
    GpuResourcesResponse,
    GpuWorkersResponse,
    TrainingRunResourcesResponse,
)
from train_platform.schemas.v3.training_runs import TrainingRunResourceRequestOut


router = APIRouter(tags=["gpu-resources"])


@router.get("/gpu-resources", response_model=GpuResourcesResponse)
def get_gpu_resources(
    node_id: str | None = Query(None),
    gpu_uuid: str | None = Query(None),
    db: Session = Depends(get_db),
):
    items = list_gpu_resources(
        db,
        node_id=node_id,
        gpu_uuid=gpu_uuid,
        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
    )
    return GpuResourcesResponse(items=items)


@router.get("/gpu-workers", response_model=GpuWorkersResponse)
def get_gpu_workers(db: Session = Depends(get_db)):
    items = list_gpu_workers(
        db,
        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
    )
    return GpuWorkersResponse(items=items)


@router.get(
    "/training-runs/{run_id}/resources",
    response_model=TrainingRunResourcesResponse,
)
def get_training_run_resources(
    run_id: str,
    db: Session = Depends(get_db),
):
    run = TrainingRunService().get_run(db, run_id)
    request = run.resource_request
    payload = (
        None
        if request is None
        else TrainingRunResourceRequestOut.model_validate(request)
    )
    reason_code = (
        "legacy_device_mode"
        if request is None
        else "resource_scheduler_not_enabled"
    )
    return TrainingRunResourcesResponse(
        run_id=run_id,
        resource_request=payload,
        legacy_device_mode=request is None,
        reason_code=reason_code,
    )
