from __future__ import annotations

from fastapi import APIRouter, Query

from train_platform.domains.monitoring.metrics.service import (
    get_cluster_overview as get_monitoring_cluster_overview,
    get_system_metrics_history,
    sample,
)
from train_platform.schemas.v3.system_metrics import ClusterOverviewOut, SystemMetricHistoryOut, SystemMetricOut


router = APIRouter(prefix="/system-metrics", tags=["system-metrics"])


@router.get("/summary", response_model=SystemMetricOut)
def get_system_summary(
    node_id: str = Query("backend", description="Node ID, e.g. backend/worker-yolo"),
    node_type: str = Query("backend", description="Node type, e.g. backend/worker/inference-worker"),
):
    return sample(node_id=node_id, node_type=node_type)


@router.get("/history", response_model=SystemMetricHistoryOut)
def get_system_history(
    minutes: int = Query(10, ge=1, le=1440, description="History window in minutes, default 10, max 1440"),
    node: str = Query("backend", description="Node ID, e.g. backend/worker-yolo"),
    node_type: str = Query("backend", description="Node type, e.g. backend/worker/inference-worker"),
    step_seconds: int = Query(5, ge=1, le=300, description="Down-sampling step in seconds"),
):
    return get_system_metrics_history(
        minutes=int(minutes),
        node_id=str(node),
        node_type=str(node_type),
        step_seconds=int(step_seconds),
    )


@router.get("/nodes", response_model=ClusterOverviewOut)
def get_cluster_overview():
    return get_monitoring_cluster_overview()
