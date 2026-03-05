from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import LogLevel, TrainingRunStatus


class TrainingRunParametersIn(BaseModel):
    epochs: int = Field(100, gt=0)
    batch_size: int = Field(16, gt=0)
    image_size: int = Field(640, gt=0)
    learning_rate: float = Field(0.01, gt=0)
    patience: int = Field(50, ge=0)
    device: str = Field("auto")
    workers: int = Field(8, ge=-1)
    use_pretrained: bool = True
    optimizer: str = Field("AdamW")
    augmentation: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None


class TrainingRunParametersOut(TrainingRunParametersIn):
    param_id: int
    run_id: str

    model_config = {"from_attributes": True}


class TrainingRunCreate(BaseModel):
    project_id: int
    architecture_id: int
    dataset_version_id: Optional[int] = Field(None, description="不传则使用 dataset.active_version_id")
    name: Optional[str] = Field(None, max_length=255)
    parameters: TrainingRunParametersIn


class TrainingRunUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)


class TrainingRunResultOut(BaseModel):
    result_id: int
    run_id: str
    best_weights_path: Optional[str] = None
    last_weights_path: Optional[str] = None
    results_dir: Optional[str] = None
    final_metrics: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    flops: Optional[int] = None

    model_config = {"from_attributes": True}


class TrainingRunMetaOut(BaseModel):
    run_id: str
    creator: Optional[str] = None
    group_name: Optional[str] = Field(None, serialization_alias="group")
    tags: Optional[Any] = None
    notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TrainingRunMetaUpdate(BaseModel):
    creator: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[Any] = None
    notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class TrainingRunOut(BaseModel):
    run_id: str
    project_id: int
    dataset_version_id: int
    architecture_id: int

    name: str
    status: TrainingRunStatus
    progress: int
    current_epoch: int
    total_epochs: Optional[int] = None

    queued_at: Optional[datetime] = None
    claimed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    pid: Optional[int] = None
    heartbeat_at: Optional[datetime] = None
    cancel_requested_at: Optional[datetime] = None
    cancel_reason: Optional[str] = None
    delete_requested_at: Optional[datetime] = None
    hidden: bool

    run_dir: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    parameters: Optional[TrainingRunParametersOut] = None
    result: Optional[TrainingRunResultOut] = None
    meta: Optional[TrainingRunMetaOut] = None

    model_config = {"from_attributes": True}


class TrainingRunEventOut(BaseModel):
    event_id: int
    run_id: str
    level: LogLevel
    event_type: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunArtifactOut(BaseModel):
    artifact_id: int
    run_id: str
    kind: str
    name: str
    path: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunEpochMetricOut(BaseModel):
    metric_id: int
    run_id: str
    epoch: int
    metrics: Dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunCompareRequest(BaseModel):
    run_ids: List[str] = Field(..., min_length=2, max_length=20)


class TrainingRunCompareItem(BaseModel):
    run_id: str
    name: str
    status: TrainingRunStatus
    project_id: int
    dataset_version_id: int
    architecture_id: int
    created_at: datetime
    engine: Optional[str] = None
    framework_key: Optional[str] = None
    framework_label: Optional[str] = None
    family: Optional[str] = None
    variant: Optional[str] = None
    parameters: Dict[str, Any]
    best_metrics: Optional[Dict[str, Any]] = None
    final_metrics: Optional[Dict[str, Any]] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None


class TrainingRunCompareResponse(BaseModel):
    runs: List[TrainingRunCompareItem]
    parameter_diff: Dict[str, Dict[str, Any]]


class TrainingRunLogTailOut(BaseModel):
    run_id: str
    which: str
    lines: int
    text: str


class TrainingRunExportRequest(BaseModel):
    """
    Export/convert a completed YOLOv8 training run's weights to a deployable format.

    Currently supported:
    - pt: raw weights download (best/last)
    - onnx: Ultralytics export (YOLOv8 -> ONNX)
    """

    format: str = Field("pt", description="pt | onnx")
    weights: str = Field("best", description="best | last")

    # ONNX options (best-effort; ignored for pt)
    opset: int | None = Field(None, ge=9, le=20)
    dynamic: bool = Field(True, description="dynamic axes (ONNX)")
    imgsz: int | None = Field(None, ge=32, le=4096, description="image size (ONNX)")


class TrainingRunExportOut(BaseModel):
    run_id: str
    format: str
    weights: str
    download_url: str
    artifact: Optional[TrainingRunArtifactOut] = None
