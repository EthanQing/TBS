from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


InferenceMode = Literal["image", "batch", "video"]
InferenceJobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
InferenceJobPhase = Literal["preparing", "inferring", "rendering", "finalizing", "done", "failed", "cancelled"]


class InferenceModelCandidate(BaseModel):
    source: Literal["model_version", "training_run"]
    model_version_id: Optional[int] = None
    run_id: str
    project_id: int
    architecture_id: int
    engine: str
    family: Optional[str] = None
    variant: Optional[str] = None
    version: Optional[str] = None
    label: str
    weights_path: str
    config_path: Optional[str] = None
    inferable: bool = True
    infer_reason: Optional[str] = None
    created_at: Optional[datetime] = None


class InferenceJobCreate(BaseModel):
    mode: InferenceMode = "image"
    model_version_id: Optional[int] = None
    run_id: Optional[str] = Field(None, min_length=1, max_length=64)

    input_tokens: List[str] = Field(default_factory=list)
    video_token: Optional[str] = Field(None, min_length=1, max_length=500)

    conf: float = Field(0.5, gt=0, le=1)
    iou: float = Field(0.45, gt=0, le=1)
    show_labels: bool = True
    show_confidence: bool = True

    @model_validator(mode="after")
    def _validate_inputs(self) -> "InferenceJobCreate":
        if self.model_version_id is None and not self.run_id:
            raise ValueError("Either model_version_id or run_id is required")

        if self.mode == "video":
            if not self.video_token:
                raise ValueError("video_token is required for video mode")
            return self

        if not self.input_tokens:
            raise ValueError("input_tokens is required for image/batch mode")
        return self


class InferenceJobItemResult(BaseModel):
    result_id: int
    filename: str
    token: Optional[str] = None
    status: Literal["success", "failed"]
    detections: int = 0
    inference_time_ms: Optional[float] = None
    source_url: Optional[str] = None
    output_url: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class InferenceJobVideoResult(BaseModel):
    output_url: str
    total_frames: int
    processed_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_time_ms: Optional[float] = None


class InferenceJobResult(BaseModel):
    mode: InferenceMode
    items: List[InferenceJobItemResult] = Field(default_factory=list)
    video: Optional[InferenceJobVideoResult] = None


class InferenceJobOut(BaseModel):
    job_id: str
    status: InferenceJobStatus
    phase: Optional[InferenceJobPhase] = None
    mode: InferenceMode
    progress: int = 0
    processed: int = 0
    total: int = 0

    seq: int = 0
    last_result_id: int = 0

    model_version_id: Optional[int] = None
    run_id: Optional[str] = None
    engine: Optional[str] = None
    family: Optional[str] = None
    variant: Optional[str] = None

    result: Optional[InferenceJobResult] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class InferenceJobWsMessage(BaseModel):
    type: Literal["snapshot", "progress", "item", "done", "error", "ping"]
    data: Dict[str, Any] = Field(default_factory=dict)
