from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from train_platform.models.v3.enums import DatasetSplit, DatasetType, DatasetVersionStatus
from train_platform.schemas.v3.common import PageMeta


class IllegalDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_type: DatasetType
    format: str = Field("yolo", min_length=1, max_length=50)
    description: Optional[str] = None


class IllegalDatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class IllegalDatasetOut(BaseModel):
    illegal_dataset_id: int
    name: str
    dataset_type: DatasetType
    format: str
    storage_path: str
    description: Optional[str] = None
    active_version_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IllegalDatasetVersionOut(BaseModel):
    version_id: int
    illegal_dataset_id: int
    version: int
    parent_version_id: Optional[int] = None
    status: DatasetVersionStatus
    message: Optional[str] = None
    manifest_path: Optional[str] = None
    snapshot_path: Optional[str] = None
    file_count: Optional[int] = None
    size_bytes: Optional[int] = None
    meta: Optional[dict] = None
    created_by: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class IllegalDatasetEventOut(BaseModel):
    event_id: int
    illegal_dataset_id: int
    version_id: Optional[int] = None
    event_type: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetFileOut(BaseModel):
    path: str
    size_bytes: int
    mtime: float
    url: Optional[str] = None
    exists: bool = True


class DatasetImageOut(BaseModel):
    image_id: int
    path: str
    split: Optional[DatasetSplit] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetStatisticsOut(BaseModel):
    total_files: int
    total_size_bytes: int
    total_size_mb: float
    size_mb: float = 0.0
    dataset_size_mb: float = 0.0
    total_images: int
    num_images: int = 0
    image_count: int = 0
    num_classes: int = 0
    class_count: int = 0
    declared_class_count: int = 0
    used_class_count: int = 0
    annotations_count: Optional[int] = None
    target_count: int = 0
    total_targets: int = 0
    object_count: int = 0
    total_objects: int = 0


class IllegalDatasetListOut(IllegalDatasetOut):
    statistics: Optional[DatasetStatisticsOut] = None


class DatasetImageUploadOut(BaseModel):
    saved_count: int
    saved_files: list[str] = Field(default_factory=list)
    total_bytes: int
    created_at: datetime
    version_id: Optional[int] = None
    version: Optional[int] = None
    active_version_id: Optional[int] = None


class CategoryInfo(BaseModel):
    class_id: int
    name: str
    count: int


class ViewImageItem(BaseModel):
    id: int
    name: str
    path: str
    url: str
    thumbnail_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    object_count: int = 0
    classes: list[int] = Field(default_factory=list)


class ViewMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class DatasetViewOut(BaseModel):
    categories: list[CategoryInfo]
    items: list[ViewImageItem]
    meta: ViewMeta


class DatasetImageAnnotationBoxOut(BaseModel):
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float


class DatasetImageAnnotationsOut(BaseModel):
    image_path: str
    image_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    object_count: int = 0
    boxes: list[DatasetImageAnnotationBoxOut] = Field(default_factory=list)


class IllegalDatasetLabelMappingRow(BaseModel):
    raw_label: str = Field(..., min_length=1, max_length=255)
    mapped_label: str = Field(..., min_length=1, max_length=255)


class IllegalDatasetLabelMappingsOut(BaseModel):
    items: list[IllegalDatasetLabelMappingRow] = Field(default_factory=list)


class IllegalDatasetLabelMappingsUpdate(BaseModel):
    items: list[IllegalDatasetLabelMappingRow] = Field(default_factory=list)


class IllegalDatasetRawLabelsOut(BaseModel):
    labels: list[str] = Field(default_factory=list)


class IllegalDatasetPublishRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version_id: Optional[int] = None
    label_filters: list[str] = Field(default_factory=list)
    label_mapping_overrides: Dict[str, str] = Field(default_factory=dict)
    split: Dict[str, Any] = Field(default_factory=dict)
    publish_config: Dict[str, Any] = Field(default_factory=dict)


class IllegalDatasetPublishOut(BaseModel):
    standard_dataset_id: int
    name: str
    source_illegal_dataset_id: int
    source_illegal_version_id: int
    publish_config: Dict[str, Any] = Field(default_factory=dict)


class IllegalDatasetDetailOut(BaseModel):
    dataset: IllegalDatasetOut
    statistics: Optional[DatasetStatisticsOut] = None
    active_version: Optional[IllegalDatasetVersionOut] = None
    versions: list[IllegalDatasetVersionOut] = Field(default_factory=list)
    events: list[IllegalDatasetEventOut] = Field(default_factory=list)


class DatasetSplitSummary(BaseModel):
    total_images: int
    train_count: int
    val_count: int
    test_count: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: Optional[int] = None
    shuffle: Optional[bool] = None


class DatasetSplitResultOut(BaseModel):
    summary: DatasetSplitSummary
    items: list[DatasetImageOut]
    meta: PageMeta
