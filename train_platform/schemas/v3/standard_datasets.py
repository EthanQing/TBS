from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from train_platform.models.v3.enums import DatasetSplit, DatasetType
from train_platform.schemas.v3.common import PageMeta


class StandardDatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_type: DatasetType
    format: str = Field("yolo", min_length=1, max_length=50)
    description: Optional[str] = None


class StandardDatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class StandardDatasetOut(BaseModel):
    standard_dataset_id: int
    name: str
    dataset_type: DatasetType
    format: str
    storage_path: str
    description: Optional[str] = None
    source_type: Optional[str] = None
    publish_config: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StandardDatasetEventOut(BaseModel):
    event_id: int
    standard_dataset_id: int
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


class DatasetSplitRequest(BaseModel):
    train_ratio: float = Field(0.9, gt=0, le=1)
    val_ratio: Optional[float] = Field(None, ge=0, lt=1)
    test_ratio: Optional[float] = Field(None, ge=0, lt=1)
    seed: Optional[int] = None
    shuffle: bool = True
    overwrite: bool = True


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


class StandardDatasetListOut(StandardDatasetOut):
    statistics: Optional[DatasetStatisticsOut] = None


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


class StandardDatasetDetailOut(BaseModel):
    dataset: StandardDatasetOut
    statistics: Optional[DatasetStatisticsOut] = None
    events: list[StandardDatasetEventOut] = Field(default_factory=list)


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
