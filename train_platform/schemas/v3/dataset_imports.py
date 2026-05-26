from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetImportRootOut(BaseModel):
    root_id: str
    path: str
    label: str
    exists: bool
    readable: bool
    editable: bool = False


class DatasetImportRootCreate(BaseModel):
    path: str
    label: str | None = None


class DatasetImportRootDeleteOut(BaseModel):
    root_id: str
    deleted: bool = True


class DatasetImportFsEntryOut(BaseModel):
    name: str
    path: str
    is_dir: bool = True
    readable: bool = True


class DatasetImportFsEntriesOut(BaseModel):
    path: str
    parent_path: str | None = None
    entries: list[DatasetImportFsEntryOut]


class DatasetImportEntryOut(BaseModel):
    name: str
    path: str
    is_dir: bool
    is_dataset_candidate: bool = False
    image_count: int | None = None
    json_count: int | None = None
    label_count: int | None = None
    has_data_yaml: bool = False


class DatasetImportEntriesOut(BaseModel):
    root: DatasetImportRootOut
    path: str
    parent_path: str | None = None
    entries: list[DatasetImportEntryOut]


class DatasetImportInspectRequest(BaseModel):
    root_id: str = "default"
    path: str = ""


class DatasetImportInspectOut(BaseModel):
    root_id: str
    path: str
    resolved_path: str
    exists: bool
    is_dir: bool
    format: str
    image_count: int = 0
    json_count: int = 0
    label_count: int = 0
    has_data_yaml: bool = False
    warnings: list[str] = Field(default_factory=list)
