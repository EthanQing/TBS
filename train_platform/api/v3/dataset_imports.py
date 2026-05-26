from __future__ import annotations

from fastapi import APIRouter, Query

from train_platform.schemas.v3.dataset_imports import (
    DatasetImportEntriesOut,
    DatasetImportFsEntriesOut,
    DatasetImportInspectOut,
    DatasetImportInspectRequest,
    DatasetImportRootCreate,
    DatasetImportRootDeleteOut,
    DatasetImportRootOut,
)
from train_platform.services.v3.dataset_import_service import DatasetImportService


router = APIRouter(prefix="/dataset-imports", tags=["dataset-imports"])
svc = DatasetImportService()


@router.get("/roots", response_model=list[DatasetImportRootOut])
def list_dataset_import_roots():
    return svc.roots()


@router.post("/roots", response_model=DatasetImportRootOut, status_code=201)
def create_dataset_import_root(payload: DatasetImportRootCreate):
    return svc.add_root(payload)


@router.delete("/roots/{root_id}", response_model=DatasetImportRootDeleteOut)
def delete_dataset_import_root(root_id: str):
    return svc.delete_root(root_id)


@router.get("/fs/entries", response_model=DatasetImportFsEntriesOut)
def list_dataset_import_filesystem_entries(path: str = Query("")):
    return svc.browse_filesystem(path=path)


@router.get("/entries", response_model=DatasetImportEntriesOut)
def list_dataset_import_entries(
    root_id: str = Query("default"),
    path: str = Query(""),
):
    return svc.list_entries(root_id=root_id, path=path)


@router.post("/inspect", response_model=DatasetImportInspectOut)
def inspect_dataset_import_path(payload: DatasetImportInspectRequest):
    return svc.inspect(root_id=payload.root_id, path=payload.path)
