from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

class GpuDeviceMetricOut(BaseModel):
    gpu_index: int = Field(..., ge=0, description="index of the GPU device")
    name: str = Field(..., min_length=1, description="name of the GPU device")
    uuid: str | None = Field(None, description="GPU UUID(Optional)")

    utilization_percent: float | None = Field(None, ge=0, le=100, description="GPU Usage Rate(%)")
    memory_used_mb: float | None = Field(None, ge=0, description="GPU Memory Used(MB)")
    memory_total_mb: float | None = Field(None, ge=0, description="GPU Memory Total(MB)")
    memory_percent: float | None = Field(None, ge=0, le=100, description="GPU Memory Usage Rate(%)")

    model_config = {"from_attributes": True, "extra": "forbid"}

class SystemMetricCoreOut(BaseModel):
    cpu_percent: float | None = Field(None, ge=0, le=100, description="CPU Usage Rate(%)")
    
    memory_percent: float | None = Field(None, ge=0, le=100, description="Memory Usage Rate(%)")
    memory_used_mb: float | None = Field(None, ge=0, description="Memory Used(MB)")
    memory_total_mb: float | None = Field(None, ge=0, description="Memory Total(MB)")

    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_count: int = Field(0, ge=0, description="Number of GPU devices")

    gpu_percent: float | None = Field(None, ge=0, le=100, description="GPU Usage Rate(%)")
    gpu_used_mb: float | None = Field(None, ge=0, description="GPU Memory Used(MB)")
    gpu_total_mb: float | None = Field(None, ge=0, description="GPU Memory Total(MB)")  

    gpus: list[GpuDeviceMetricOut] = Field(default_factory=list, description="List of GPU device metrics")

    model_config = {"from_attributes": True, "extra": "forbid"}

class SystemMetricOut(SystemMetricCoreOut):
    """单节点实时指标"""
    timestamp: datetime = Field(..., description="采集时间(UTC ISO8601)")
    node_id: str = Field("backend", min_length=1, description="节点ID，如 backend/worker-yolo")
    node_type: Literal["backend", "worker", "inference-worker", "unknown"] = Field(
        "backend",
        description="节点类型"
    )

    model_config = {"from_attributes": True, "extra": "forbid"}


class HistoryMetricOut(SystemMetricCoreOut):
    """历史曲线中的一个点"""
    timestamp: datetime = Field(..., description="采集时间(UTC ISO8601)")

    model_config = {"from_attributes": True, "extra": "forbid"}


class SystemMetricHistoryOut(BaseModel):
    """历史曲线响应"""
    node_id: str = Field(..., min_length=1)
    node_type: Literal["backend", "worker", "inference-worker", "unknown"] = "backend"

    window_seconds: int = Field(..., ge=1, description="查询窗口时长(秒)")
    step_seconds: int = Field(..., ge=1, description="采样间隔(秒)")
    points: list[HistoryMetricOut] = Field(default_factory=list)

    model_config = {"from_attributes": True, "extra": "forbid"}


class ClusterOverviewOut(BaseModel):
    """全局概览（多节点聚合）"""
    timestamp: datetime
    total_nodes: int = Field(..., ge=0)
    online_nodes: int = Field(..., ge=0)

    # 聚合概览（可选）
    cpu_percent_avg: float | None = Field(None, ge=0, le=100)
    memory_percent_avg: float | None = Field(None, ge=0, le=100)
    gpu_percent_avg: float | None = Field(None, ge=0, le=100)

    nodes: list[SystemMetricOut] = Field(default_factory=list)

    model_config = {"from_attributes": True, "extra": "forbid"}
