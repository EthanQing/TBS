from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import get_db
from train_platform.domains.training.resources.queries import (
    list_gpu_resources,
    list_gpu_workers,
    get_training_run_resources as query_training_run_resources,
)
from train_platform.domains.training.runs.service import TrainingRunService
from train_platform.schemas.v3.gpu_resources import (
    GpuResourcesResponse,
    GpuWorkersResponse,
    TrainingRunResourcesResponse,
)
from train_platform.schemas.v3.training_runs import TrainingRunResourceRequestOut
from train_platform.models.v3.gpu_allocation import GpuNodeSchedulingState


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
    enabled = any(item["allocation_enabled"] for item in items)
    return GpuResourcesResponse(
        scheduler_stage="managed_allocation" if any(item["scheduling_mode"] == "managed" for item in items) else "inventory_only",
        allocation_enabled=enabled,
        shared_execution_enabled=any(item["shared_execution_enabled"] for item in items),
        items=items,
    )


@router.get("/gpu-workers", response_model=GpuWorkersResponse)
def get_gpu_workers(db: Session = Depends(get_db)):
    items = list_gpu_workers(
        db,
        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
    )
    enabled = any(item["accepting_tasks"] for item in items)
    return GpuWorkersResponse(
        scheduler_stage="managed_allocation" if any(item["scheduling_mode"] == "managed" for item in items) else "inventory_only",
        allocation_enabled=enabled,
        items=items,
    )


@router.get(
    "/training-runs/{run_id}/resources",
    response_model=TrainingRunResourcesResponse,
)
def get_training_run_resources(
    run_id: str,
    db: Session = Depends(get_db),
):
    run = TrainingRunService().get_run(db, run_id)
    payload = query_training_run_resources(db, run)
    request = run.resource_request
    payload["resource_request"] = None if request is None else TrainingRunResourceRequestOut.model_validate(request)
    nodes = db.query(GpuNodeSchedulingState).filter(GpuNodeSchedulingState.managed.is_(True))
    target_node = request.node_id if request else None
    if payload["allocation"]:
        target_node = payload["allocation"]["node_id"]
    if target_node:
        nodes = nodes.filter(GpuNodeSchedulingState.node_id == target_node)
    managed_nodes = nodes.all()
    enabled = any(node.accepting_allocations for node in managed_nodes)
    if payload["reason_code"] is None and not enabled and run.resource_request is not None:
        payload["reason_code"] = "scheduler_disabled"
    return TrainingRunResourcesResponse(
        scheduler_stage="managed_allocation" if managed_nodes else "inventory_only",
        allocation_enabled=enabled,
        **payload,
    )
