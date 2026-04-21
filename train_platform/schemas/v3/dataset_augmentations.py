from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


DatasetAugmentationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
DatasetAugmentationPhase = Literal["preparing", "scanning", "augmenting", "finalizing", "done", "failed", "cancelled"]


class DatasetAugmentationSliceConfig(BaseModel):
    enabled: bool = True
    scales: List[int] = Field(default_factory=lambda: [640])
    overlap: float = Field(0.2, ge=0, lt=1)
    min_area_ratio: float = Field(0.3, ge=0, le=1)
    min_visibility: float = Field(0.15, ge=0, le=1)
    min_pixel_size: int = Field(4, ge=1, le=4096)

    @model_validator(mode="after")
    def _normalize_scales(self) -> "DatasetAugmentationSliceConfig":
        vals = []
        for s in self.scales:
            try:
                v = int(s)
            except Exception:
                continue
            if v >= 32:
                vals.append(v)
        uniq = sorted(set(vals))
        self.scales = uniq or [640]
        return self


class DatasetAugmentationRotateConfig(BaseModel):
    enabled: bool = False
    angles: List[float] = Field(default_factory=list)
    border_value: int = Field(114, ge=0, le=255)

    @model_validator(mode="after")
    def _normalize_angles(self) -> "DatasetAugmentationRotateConfig":
        vals = []
        for a in self.angles:
            try:
                vals.append(float(a))
            except Exception:
                continue
        self.angles = sorted(set(vals))
        return self


class DatasetAugmentationOffset(BaseModel):
    dx: float = Field(..., ge=-1, le=1)
    dy: float = Field(..., ge=-1, le=1)


class DatasetAugmentationTranslateConfig(BaseModel):
    enabled: bool = False
    offsets: List[DatasetAugmentationOffset] = Field(default_factory=list)
    border_value: int = Field(114, ge=0, le=255)


class DatasetAugmentationConfig(BaseModel):
    version_id: Optional[int] = None
    include_original: bool = True
    max_outputs_per_image: int = Field(200, ge=1, le=5000)
    slice: DatasetAugmentationSliceConfig = Field(default_factory=DatasetAugmentationSliceConfig)
    rotate: DatasetAugmentationRotateConfig = Field(default_factory=DatasetAugmentationRotateConfig)
    translate: DatasetAugmentationTranslateConfig = Field(default_factory=DatasetAugmentationTranslateConfig)


class DatasetAugmentationPreviewRequest(DatasetAugmentationConfig):
    pass


class DatasetAugmentationCreate(DatasetAugmentationConfig):
    pass


class DatasetAugmentationPreviewOut(BaseModel):
    total_images: int = 0
    with_labels: int = 0
    estimated_generated_outputs: int = 0
    estimated_total_outputs: int = 0
    per_transform: Dict[str, int] = Field(default_factory=dict)
    note: Optional[str] = None


class DatasetAugmentationItemResult(BaseModel):
    result_id: int
    source_image: str
    generated: int
    processed_at: Optional[datetime] = None


class DatasetAugmentationResult(BaseModel):
    output_url: Optional[str] = None
    output_file_count: int = 0
    generated_images: int = 0
    generated_labels: int = 0
    published_standard_dataset_id: Optional[int] = None


class DatasetAugmentationJobOut(BaseModel):
    job_id: str
    standard_dataset_id: int
    status: DatasetAugmentationStatus
    phase: Optional[DatasetAugmentationPhase] = None
    progress: int = 0
    processed: int = 0
    total: int = 0

    seq: int = 0
    last_result_id: int = 0

    config: Optional[DatasetAugmentationConfig] = None
    result: Optional[DatasetAugmentationResult] = None
    cancel_requested: bool = False
    error_message: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetAugmentationCancelOut(BaseModel):
    job_id: str
    status: DatasetAugmentationStatus
    cancel_requested: bool = True


class DatasetAugmentationPublishIn(BaseModel):
    activate: bool = False
    message: Optional[str] = None
    created_by: Optional[str] = None


class DatasetAugmentationPublishOut(BaseModel):
    standard_dataset_id: int
    job_id: str
    source_standard_dataset_id: int


class DatasetAugmentationWsMessage(BaseModel):
    type: Literal["snapshot", "progress", "item", "done", "error", "ping"]
    data: Dict[str, Any] = Field(default_factory=dict)
