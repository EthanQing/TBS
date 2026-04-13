from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from train_platform.models.enums import DatasetSplit, DatasetType, DatasetVersionStatus
from train_platform.schemas.v2.common import PageMeta


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_type: DatasetType
    format: Optional[str] = Field("yolo", min_length=1, max_length=50)
    storage_path: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="Deprecated/ignored. Dataset files are stored under BASE_DATASETS_DIR/<dataset_id>/.",
    )
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    active_version_id: Optional[int] = None


class DatasetOut(BaseModel):
    dataset_id: int
    name: str
    dataset_type: DatasetType
    format: str = "yolo"
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
    train_ratio: float = Field(0.9, gt=0, lt=1)
    val_ratio: Optional[float] = Field(None, gt=0, lt=1)
    test_ratio: Optional[float] = Field(None, gt=0, lt=1)
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


class DatasetDetailOut(BaseModel):
    dataset: DatasetOut
    statistics: Optional[DatasetStatisticsOut] = None
    active_version: Optional[DatasetVersionOut] = None
    versions: list[DatasetVersionOut] = []
    events: list[DatasetEventOut] = []


# --- New schemas for enhanced list and view endpoints ---

class DatasetListStatistics(BaseModel):
    """Embedded statistics for dataset list items."""
    num_images: int = 0
    num_classes: int = 0
    size_mb: float = 0.0


class DatasetListOut(DatasetOut):
    """Extended dataset output with embedded statistics for list view."""
    statistics: Optional[DatasetListStatistics] = None


class CategoryInfo(BaseModel):
    """Category information for the view endpoint sidebar."""
    class_id: int
    name: str
    count: int  # Number of images containing this class


class ViewImageItem(BaseModel):
    """Image item for the view endpoint grid."""
    id: int
    name: str
    url: str
    thumbnail_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    object_count: int = 0
    classes: list[int] = Field(default_factory=list)  # Class IDs present in this image


class ViewMeta(BaseModel):
    """Pagination metadata for the view endpoint."""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    thumbnail_status: Optional[str] = None
    thumbnail_progress: Optional[int] = None
    view_index_status: Optional[str] = None


class DatasetViewOut(BaseModel):
    """Response model for the dataset view endpoint."""
    dataset_id: int
    version_id: int
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
    dataset_id: int
    version_id: int
    image_path: str
    image_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    object_count: int = 0
    boxes: list[DatasetImageAnnotationBoxOut] = Field(default_factory=list)


class DatasetIllegalConvertRequest(BaseModel):
    label_strategy: str = Field(..., description="full | leaf | root | level | mapping")
    label_level: Optional[int] = Field(None, ge=1)
    label_separator: str = Field("%", min_length=1, max_length=10)
    label_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Optional mapping from raw label to target label (used when label_strategy=mapping)",
    )
    # Slice / crop parameters (override service defaults when provided)
    slice_size: Optional[int] = Field(None, ge=64, le=8192, description="Slice width/height in pixels")
    overlap: Optional[float] = Field(None, ge=0, lt=1, description="Overlap ratio between adjacent slices")
    padding: Optional[int] = Field(None, ge=0, le=1024, description="Extra padding around annotation bboxes")
    min_area_ratio: Optional[float] = Field(None, ge=0, le=1, description="Min bbox area ratio inside slice")
    min_visibility: Optional[float] = Field(None, ge=0, le=1, description="Min bbox width/height visibility")
    min_pixel_size: Optional[int] = Field(None, ge=1, le=1000, description="Min bbox pixel size in slice")
    negative_ratio: Optional[float] = Field(None, ge=0, le=1, description="Ratio of negative to positive slices")
    empty_positive_action: Optional[str] = Field(None, description="Action for empty positive slices: discard | negative")
    # Split parameters
    split_enabled: Optional[bool] = Field(False)
    split_train_ratio: Optional[float] = Field(None, gt=0, lt=1)
    split_val_ratio: Optional[float] = Field(None, gt=0, lt=1)
    split_test_ratio: Optional[float] = Field(None, gt=0, lt=1)
    split_seed: Optional[int] = None
    split_shuffle: Optional[bool] = None
    split_overwrite: Optional[bool] = None

    @model_validator(mode="after")
    def _check_level(self):
        strategy = str(self.label_strategy or "").strip().lower()
        if strategy == "level":
            if self.label_level is None or int(self.label_level) < 1:
                raise ValueError("label_level is required when label_strategy=level")
        return self


class DatasetIllegalConvertOut(BaseModel):
    job_id: str
    status: str

class DatasetIllegalLabelsOut(BaseModel):
    labels: list[str] = Field(default_factory=list, description="List of unique raw labels found in the dataset")

class DatasetIllegalLabelsUpdate(BaseModel):
    label_mapping: Dict[str, str] = Field(..., description="Mapping from old raw label to new absolute label")


class IllegalLabelPresetDetectionRow(BaseModel):
    source_label: str = Field(..., min_length=1, description="Raw source label path")
    target_label: str = Field(..., min_length=1, description="Mapped target label")


class IllegalLabelPresetClassificationRow(BaseModel):
    category: str = Field(..., min_length=1, description="Classification category")
    source_label: str = Field(..., min_length=1, description="Raw source label path")
    target_label: Optional[str] = Field(
        None, description="Optional mapped target label, defaults to category when omitted"
    )


class IllegalLabelPresetUpdateIn(BaseModel):
    detection: list[IllegalLabelPresetDetectionRow] = Field(default_factory=list)
    classification: list[IllegalLabelPresetClassificationRow] = Field(default_factory=list)


class IllegalLabelPresetOut(IllegalLabelPresetUpdateIn):
    updated_at: Optional[str] = None


class DatasetRenameClassesRequest(BaseModel):
    """Rename class labels in a converted YOLO dataset.

    Keys are the current class names, values are the new names.
    Only classes that need renaming should be included.
    class_id (index) is preserved — only the display name changes.
    """
    rename_map: Dict[str, str] = Field(
        ...,
        description="Mapping from current class name to new class name",
        min_length=1,
    )


class DatasetRenameClassesOut(BaseModel):
    renamed: int = Field(..., description="Number of classes actually renamed")
    total_classes: int = Field(..., description="Total number of classes after rename")
    class_names: list[str] = Field(..., description="Updated class name list (ordered by class_id)")


class UploadSessionCreateIn(BaseModel):
    filename: str = Field(..., min_length=1, max_length=512)
    total_size: int = Field(..., gt=0)
    chunk_size: Optional[int] = Field(None, gt=0, description="Bytes per chunk. Defaults to server policy.")


class UploadSessionCreateOut(BaseModel):
    session_id: str
    dataset_id: int
    filename: str
    total_size: int
    chunk_size: int
    total_parts: int
    uploaded_parts: int = 0
    expires_at: datetime
    status: str


class UploadPartOut(BaseModel):
    session_id: str
    part_no: int
    uploaded_parts: int
    total_parts: int
    status: str


class UploadSessionStatusOut(BaseModel):
    session_id: str
    dataset_id: int
    filename: str
    total_size: int
    chunk_size: int
    total_parts: int
    uploaded_parts: int
    status: str
    expires_at: datetime
    job_id: Optional[str] = None


class UploadCompleteOut(BaseModel):
    session_id: str
    job_id: str
    status: str


class DatasetImportJobOut(BaseModel):
    job_id: str
    dataset_id: int
    session_id: str
    status: str
    phase: str
    progress: int = Field(0, ge=0, le=100)
    seq: int = 0
    updated_at: datetime
    output_version_id: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_hint: Optional[str] = None
