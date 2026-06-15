from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from train_platform.schemas.v3.inference_jobs import InferenceModelCandidate


EvaluationScope = Literal["all", "test", "val", "train"]
EvaluationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
EvaluationPhase = Literal["preparing", "inferring", "calculating", "done", "failed", "cancelled"]


class ModelEvaluationCreate(BaseModel):
    model_version_id: Optional[int] = None
    run_id: Optional[str] = Field(None, min_length=1, max_length=64)
    standard_dataset_id: int = Field(..., gt=0)
    scope: EvaluationScope = "all"
    conf: float = Field(0.25, gt=0, le=1)
    iou: float = Field(0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _validate_model_ref(self) -> "ModelEvaluationCreate":
        if self.model_version_id is None and not self.run_id:
            raise ValueError("Either model_version_id or run_id is required")
        return self


class ModelEvaluationClassMetric(BaseModel):
    class_id: int
    class_name: Optional[str] = None
    gt_count: int = 0
    pred_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0


class ModelEvaluationMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    evaluated_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_targets: int = 0
    total_predictions: int = 0
    elapsed_ms: Optional[float] = None
    class_metrics: List[ModelEvaluationClassMetric] = Field(default_factory=list)


class ModelEvaluationItem(BaseModel):
    result_id: int
    filename: str
    image_path: Optional[str] = None
    status: Literal["success", "failed", "skipped"]
    gt_count: int = 0
    prediction_count: int = 0
    inference_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class ModelEvaluationResult(BaseModel):
    metrics: Optional[ModelEvaluationMetrics] = None
    items: List[ModelEvaluationItem] = Field(default_factory=list)


class ModelEvaluationOut(BaseModel):
    job_id: str
    status: EvaluationStatus
    phase: Optional[EvaluationPhase] = None
    progress: int = 0
    processed: int = 0
    total: int = 0
    seq: int = 0
    last_result_id: int = 0

    model_version_id: Optional[int] = None
    run_id: Optional[str] = None
    standard_dataset_id: int
    dataset_name: Optional[str] = None
    scope: EvaluationScope = "all"
    conf: float = 0.25
    iou: float = 0.5
    engine: Optional[str] = None
    family: Optional[str] = None
    variant: Optional[str] = None

    result: Optional[ModelEvaluationResult] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelEvaluationWsMessage(BaseModel):
    type: Literal["snapshot", "progress", "item", "done", "error", "ping"]
    data: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "InferenceModelCandidate",
    "ModelEvaluationCreate",
    "ModelEvaluationOut",
]
