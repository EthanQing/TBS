from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import DatasetSplit, DatasetType, DatasetVersionStatus
from train_platform.schemas.v2.common import PageMeta


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_type: DatasetType
    storage_path: str = Field(..., min_length=1, max_length=500, description="建议存 token/相对路径（相对于 BASE_DATASETS_DIR）")
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    active_version_id: Optional[int] = None


class DatasetOut(BaseModel):
    dataset_id: int
    name: str
    dataset_type: DatasetType
    storage_path: str
    description: Optional[str] = None
    active_version_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetVersionCreate(BaseModel):
    message: Optional[str] = Field(None, description="版本说明/变更说明")
    created_by: Optional[str] = None
    create_snapshot: bool = Field(False, description="是否创建快照拷贝（大数据集会很慢）")


class DatasetVersionOut(BaseModel):
    version_id: int
    dataset_id: int
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


class DatasetStatisticsOut(BaseModel):
    dataset_id: int
    version_id: int
    version: int
    total_files: int
    total_size_bytes: int
    total_size_mb: float
    total_images: int
    annotations_count: Optional[int] = None


class DatasetVersionDiffSummary(BaseModel):
    added: int
    removed: int
    modified: int


class DatasetVersionDiffTruncated(BaseModel):
    added: bool
    removed: bool
    modified: bool


class DatasetVersionDiffOut(BaseModel):
    dataset_id: int
    base_version_id: int
    base_version: int
    version_id: int
    version: int
    summary: DatasetVersionDiffSummary
    added: list[str]
    removed: list[str]
    modified: list[str]
    truncated: DatasetVersionDiffTruncated


class DatasetEventOut(BaseModel):
    event_id: int
    dataset_id: int
    version_id: Optional[int] = None
    event_type: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetImageUploadOut(BaseModel):
    dataset_id: int
    event_id: int
    version_id: Optional[int] = None
    version: Optional[int] = None
    active_version_id: Optional[int] = None
    relative_dir: str
    saved_count: int
    saved_files: list[str]
    truncated: bool
    total_bytes: int
    labels_relative_dir: Optional[str] = None
    saved_label_count: int = 0
    saved_label_files: list[str] = Field(default_factory=list)
    max_class_id: Optional[int] = None
    nc_before: Optional[int] = None
    nc_after: Optional[int] = None
    added_class_ids: list[int] = Field(default_factory=list)
    class_names_updated: Optional[bool] = None
    created_at: datetime


class DatasetFileOut(BaseModel):
    path: str
    size_bytes: int
    mtime: float
    url: Optional[str] = None
    exists: bool = True


class DatasetImageOut(BaseModel):
    image_id: int
    dataset_id: int
    dataset_version_id: int
    path: str
    split: Optional[DatasetSplit] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetSplitRequest(BaseModel):
    version_id: Optional[int] = None
    train_ratio: float = Field(0.8, gt=0, lt=1)
    val_ratio: Optional[float] = Field(None, gt=0, lt=1)
    seed: Optional[int] = None
    shuffle: bool = True
    overwrite: bool = True


class DatasetSplitSummary(BaseModel):
    dataset_id: int
    version_id: int
    version: int
    total_images: int
    train_count: int
    val_count: int
    train_ratio: float
    val_ratio: float
    seed: Optional[int] = None
    shuffle: Optional[bool] = None


class DatasetSplitResultOut(BaseModel):
    summary: DatasetSplitSummary
    items: list[DatasetImageOut]
    meta: PageMeta


class DatasetDetailOut(BaseModel):
    dataset: DatasetOut
    statistics: Optional[DatasetStatisticsOut] = None
    active_version: Optional[DatasetVersionOut] = None
    versions: list[DatasetVersionOut] = []
    events: list[DatasetEventOut] = []
