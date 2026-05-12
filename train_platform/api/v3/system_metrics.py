from __future__ import annotations

from fastapi import APIRouter, Query

from train_platform.schemas.v3.system_metrics import ClusterOverviewOut, SystemMetricHistoryOut, SystemMetricOut
from train_platform.services.v3.system_metrics_service import SystemMetricsService


router = APIRouter(prefix="/system-metrics", tags=["system-metrics"])


@router.get("/summary", response_model=SystemMetricOut)
def get_system_summary(
    node_id: str = Query("backend", description="Node ID, e.g. backend/worker-yolo"),
    node_type: str = Query("backend", description="Node type, e.g. backend/worker/inference-worker"),
):
    return SystemMetricsService.get_system_metrics(node_id=node_id, node_type=node_type)


@router.get("/history", response_model=SystemMetricHistoryOut)
def get_system_history(
    minutes: int = Query(10, ge=1, le=1440, description="History window in minutes, default 10, max 1440"),
    node: str = Query("backend", description="Node ID, e.g. backend/worker-yolo"),
    node_type: str = Query("backend", description="Node type, e.g. backend/worker/inference-worker"),
    step_seconds: int = Query(5, ge=1, le=300, description="Down-sampling step in seconds"),
):
    return SystemMetricsService.get_system_metrics_history(
        minutes=int(minutes),
        node_id=str(node),
        node_type=str(node_type),
        step_seconds=int(step_seconds),
    )


@router.get("/nodes", response_model=ClusterOverviewOut)
def get_cluster_overview():
    return SystemMetricsService.get_cluster_overview()
