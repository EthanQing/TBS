from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DatasetUploadSessionCreate(BaseModel):
    filename: str
    total_size: int = Field(gt=0)
    chunk_size: int | None = Field(default=None, gt=0)
    mode: str = "upload"
    created_by: str | None = None
    message: str | None = None


class DatasetImportFromPathRequest(BaseModel):
    path: str
    root_id: str = "default"
    mode: str = "upload"
    storage_strategy: str = "copy"
    created_by: str | None = None
    message: str | None = None


class DatasetUploadSessionOut(BaseModel):
    session_id: str
    dataset_kind: str
    dataset_id: int
    mode: str
    filename: str
    total_size: int
    chunk_size: int
    total_parts: int
    uploaded_parts: list[int]
    status: str
    created_by: str | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    error_message: str | None = None

    model_config = {"from_attributes": True}


class DatasetUploadPartOut(BaseModel):
    session_id: str
    part_no: int
    size: int
    uploaded_parts: list[int]
    status: str


class DatasetUploadCompleteOut(BaseModel):
    task_id: str
    session_id: str | None = None
    status: str


class DatasetUploadTaskOut(BaseModel):
    task_id: str
    dataset_kind: str
    dataset_id: int
    session_id: str | None = None
    mode: str
    source_type: str
    status: str
    stage: str
    progress: int
    error_message: str | None = None
    created_by: str | None = None
    message: str | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}
