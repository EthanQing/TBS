from __future__ import annotations

import json
import os
import random
import shutil
import stat
import time
import uuid
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import yaml
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func
from sqlalchemy.orm import Session
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None
try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.dataset_event import DatasetEvent
from train_platform.models.dataset_image import DatasetImage
from train_platform.models.enums import DatasetSplit, DatasetType, DatasetVersionStatus
from train_platform.models.project import Project
from train_platform.repositories.dataset_event_repo import DatasetEventRepository
from train_platform.repositories.dataset_repo import DatasetRepository
from train_platform.repositories.dataset_version_repo import DatasetVersionRepository
from train_platform.services.file_service import FileService
from train_platform.services.project_service import ProjectService
from train_platform.services.thumbnail_service import ThumbnailService
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.image_exts import IMAGE_EXTS
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_dataset_path


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _validate_dataset_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        raise ValidationError("name is required")
    if any(sep in s for sep in ("/", "\\")) or s in (".", ".."):
        raise ValidationError("Invalid dataset name")
    return s


def _normalize_storage_token(storage_path: str) -> str:
    """
    Normalize dataset storage token/relative path (under BASE_DATASETS_DIR).
    """
    p = str(storage_path or "").strip().replace("\\", "/")
    marker = "/static/datasets/"
    if marker in p:
        p = p.split(marker, 1)[1]
    p = p.strip("/\\")
    if not p or p in (".", ".."):
        raise ValidationError("storage_path is required")
    rel = Path(p)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValidationError("storage_path must be a safe relative token under BASE_DATASETS_DIR")
    return rel.as_posix()


def _versions_token(ds: Dataset) -> str:
    """
    Token used for storing version manifests/snapshots under:
      BASE_DATASETS_DIR/.versions/<token>/v<version>/...

    Backwards-compatible layout:
    - Legacy datasets store versions under `.versions/<dataset_name>/...`
    - New datasets (this repo) store data under `<dataset_id>/`, so we store versions
      under `.versions/<dataset_id>/...` when dataset.storage_path == str(dataset_id).
    """
    name = str(getattr(ds, "name", "") or getattr(ds, "dataset_id", "dataset"))
    try:
        token = _normalize_storage_token(str(getattr(ds, "storage_path", "") or ""))
    except Exception:
        return name

    try:
        if token == str(int(getattr(ds, "dataset_id"))):
            return token
    except Exception:
        pass

    # Keep legacy layout for existing datasets.
    return name


class DatasetService:
    _illegal_preset_lock = threading.Lock()
    _illegal_preset_file_name = "illegal_label_mapping_presets.json"
    _artifact_jobs_lock = threading.Lock()
    _artifact_jobs: dict[str, dict[str, Any]] = {}

    def __init__(self) -> None:
        self.datasets = DatasetRepository()
        self.versions = DatasetVersionRepository()
        self.events = DatasetEventRepository()

    # --------------------
    # datasets
    # --------------------
    def list_datasets(self, db: Session, *, skip: int = 0, limit: int = 100, format: str | None = None) -> list[Dataset]:
        q = self.datasets.get_query(db)
        if format:
            q = q.filter(Dataset.format == str(format))
        return q.offset(skip).limit(limit).all()

    def list_datasets_with_stats(self, db: Session, *, skip: int = 0, limit: int = 100, format: str | None = None) -> list[dict]:
        """
        List datasets with embedded statistics for the list view.
        Returns dataset objects with statistics dict containing num_images, num_classes, size_mb.
        """
        datasets = self.list_datasets(db, skip=skip, limit=limit, format=format)
        results = []
        
        for ds in datasets:
            # Build base dict from dataset
            item = {
                "dataset_id": ds.dataset_id,
                "name": ds.name,
                "dataset_type": ds.dataset_type,
                "format": ds.format or "yolo",
                "storage_path": ds.storage_path,
                "description": ds.description,
                "active_version_id": ds.active_version_id,
                "created_at": ds.created_at,
                "updated_at": ds.updated_at,
                "statistics": None,
            }
            
            # Try to get statistics from active version
            if ds.active_version_id is not None:
                ver = (
                    db.query(DatasetVersion)
                    .filter(DatasetVersion.version_id == ds.active_version_id, DatasetVersion.dataset_id == ds.dataset_id)
                    .first()
                )
                if ver:
                    num_images = 0
                    num_classes = 0
                    size_mb = 0.0
                    
                    # Get size
                    size_bytes = int(ver.size_bytes or 0)
                    size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0
                    
                    # Get stats from meta
                    if isinstance(ver.meta, dict):
                        stats = ver.meta.get("stats")
                        if isinstance(stats, dict):
                            num_images = int(stats.get("total_images") or 0)
                        
                        yolo = ver.meta.get("yolo")
                        if isinstance(yolo, dict):
                            num_classes = int(yolo.get("nc") or 0)
                    
                    item["statistics"] = {
                        "num_images": num_images,
                        "num_classes": num_classes,
                        "size_mb": size_mb,
                    }
            
            results.append(item)
        
        return results

    def get_dataset(self, db: Session, dataset_id: int) -> Dataset:
        ds = self.datasets.get(db, int(dataset_id))
        if not ds:
            raise NotFoundError("Dataset not found")
        return ds

    def create_dataset(self, db: Session, *, obj: dict) -> Dataset:
        name = _validate_dataset_name(obj.get("name"))

        exists = self.datasets.get_by_name(db, name)
        if exists:
            raise ConflictError(f"Dataset '{name}' already exists")

        # Always use dataset_id as the dataset directory name (stable + unique).
        # Clients may still send `storage_path`; ignore it for consistency.
        tmp_token = f"_tmp_{uuid.uuid4().hex}"

        ds = self.datasets.create(
            db,
            obj_in={
                "name": name,
                "dataset_type": obj["dataset_type"],
                "format": obj.get("format", "yolo"),
                "storage_path": tmp_token,
                "description": obj.get("description"),
            },
        )

        storage_token = str(int(ds.dataset_id))
        ds.storage_path = storage_token
        db.flush()

        dataset_dir = resolve_dataset_path(storage_token)
        base_dir = settings.datasets_dir.resolve()
        if dataset_dir == base_dir or (base_dir not in dataset_dir.parents and dataset_dir != base_dir):
            raise ValidationError("storage_path must be under BASE_DATASETS_DIR")
        if dataset_dir.exists():
            if dataset_dir.is_file():
                raise ConflictError("Dataset path already exists as a file")
        else:
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create initial version only if there is already data in the dataset directory.
        has_files = False
        try:
            has_files = any(dataset_dir.iterdir())
        except Exception:
            has_files = False

        if has_files:
            self.create_version(db, ds.dataset_id, message="initial", created_by=obj.get("created_by"), create_snapshot=False)
            db.refresh(ds)
        else:
            db.commit()
            db.refresh(ds)

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ds.active_version_id) if ds.active_version_id is not None else None,
                    event_type="create_dataset",
                    message="Dataset created",
                    created_by=obj.get("created_by"),
                    data={"storage_path": ds.storage_path, "dataset_type": str(getattr(ds.dataset_type, "value", ds.dataset_type))},
                )
            )
            db.commit()
        except Exception:
            db.rollback()

        return ds

    def update_dataset(self, db: Session, dataset_id: int, *, patch: dict) -> Dataset:
        ds = self.get_dataset(db, dataset_id)

        if "name" in patch and patch["name"] is not None:
            new_name = _validate_dataset_name(patch["name"])
            exists = self.datasets.get_by_name(db, new_name)
            if exists and int(exists.dataset_id) != int(ds.dataset_id):
                raise ConflictError(f"Dataset '{new_name}' already exists")
            ds.name = new_name

        if "description" in patch:
            ds.description = patch["description"]

        if "active_version_id" in patch and patch["active_version_id"] is not None:
            ver_id = int(patch["active_version_id"])
            ver = db.query(DatasetVersion).filter(DatasetVersion.version_id == ver_id, DatasetVersion.dataset_id == ds.dataset_id).first()
            if not ver:
                raise NotFoundError("Dataset version not found")
            ds.active_version_id = ver.version_id

        db.commit()
        db.refresh(ds)
        return ds

    def delete_dataset(self, db: Session, dataset_id: int, *, delete_files: bool = False, force: bool = False) -> None:
        ds = self.get_dataset(db, dataset_id)
        ds_id = int(ds.dataset_id)
        ds_name = str(ds.name or "").strip()
        ds_storage_path = str(ds.storage_path or "").strip()
        versions_token = _versions_token(ds)

        projects = db.query(Project).filter(Project.dataset_id == ds.dataset_id).all()
        if projects and not force:
            raise ConflictError(f"Cannot delete dataset; {len(projects)} project(s) still reference it")

        if projects and force:
            svc = ProjectService()
            for p in projects:
                svc.delete_project(db, int(p.project_id), force=True)
            # Refresh dataset after project deletions/commits.
            ds = self.get_dataset(db, dataset_id)
            ds_id = int(ds.dataset_id)
            ds_name = str(ds.name or "").strip()
            ds_storage_path = str(ds.storage_path or "").strip()
            versions_token = _versions_token(ds)

        # Remove DB rows first.
        db.delete(ds)
        db.commit()

        if not delete_files:
            return

        base = settings.datasets_dir.resolve()
        thumb_base = settings.thumbnails_dir.resolve()
        versions_base = (base / ".versions").resolve(strict=False)

        def _under(parent: Path, child: Path) -> bool:
            try:
                c = child.resolve(strict=False)
            except Exception:
                return False
            return c == parent or parent in c.parents

        def _remove_path(path: Path, *, check_parent: Path) -> bool:
            p = Path(path).resolve(strict=False)
            if not _under(check_parent, p):
                return False

            def _onerror(func, path_str, _exc_info):
                try:
                    os.chmod(path_str, stat.S_IWRITE)
                    func(path_str)
                except Exception:
                    pass

            for attempt in range(6):
                try:
                    if not p.exists():
                        return True
                    if p.is_dir():
                        shutil.rmtree(p, onerror=_onerror)
                    else:
                        p.unlink(missing_ok=True)
                    if not p.exists():
                        return True
                except Exception:
                    pass
                time.sleep(0.2 * (attempt + 1))
            return not p.exists()

        # Try multiple candidates for compatibility with legacy storage_path layouts.
        dataset_candidates: list[Path] = []
        for cand in (
            resolve_dataset_path(ds_storage_path),
            (base / str(ds_id)),
            (base / ds_name) if ds_name else None,
        ):
            if cand is None:
                continue
            c = Path(cand).resolve(strict=False)
            if not _under(base, c) or c == base:
                continue
            if c not in dataset_candidates:
                dataset_candidates.append(c)
        # Delete deeper paths first.
        dataset_candidates.sort(key=lambda p: len(p.parts), reverse=True)
        for p in dataset_candidates:
            _remove_path(p, check_parent=base)

        # Also remove version snapshots/manifests for this dataset.
        try:
            version_candidates: list[Path] = []
            for token in (versions_token, str(ds_id), ds_name):
                t = str(token or "").strip()
                if not t:
                    continue
                p = (versions_base / t).resolve(strict=False)
                if p not in version_candidates and _under(versions_base, p) and p != versions_base:
                    version_candidates.append(p)
            for p in version_candidates:
                _remove_path(p, check_parent=versions_base)
        except Exception:
            pass

        # Remove thumbnail cache for this dataset.
        try:
            thumb_dir = (thumb_base / str(ds_id)).resolve(strict=False)
            _remove_path(thumb_dir, check_parent=thumb_base)
        except Exception:
            pass

    def upload_dataset_archive(
        self,
        db: Session,
        dataset_id: int,
        *,
        file,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_version: bool = True,
        activate: bool = True,
        split_enabled: bool = False,
        split_train_ratio: float | None = None,
        split_val_ratio: float | None = None,
        split_test_ratio: float | None = None,
        split_seed: int | None = None,
        split_shuffle: bool | None = None,
        split_overwrite: bool | None = None,
    ):
        ds = self.get_dataset(db, dataset_id)

        dataset_root = resolve_dataset_path(ds.storage_path)
        base = settings.datasets_dir.resolve()
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")

        _path, info = FileService().upload_dataset_into_existing(file, dataset_root, ds.dataset_type)
        
        detected_format = info.get("format")
        if detected_format in ("labelme", "unknown_json"):
            ds.format = "illegal"
            illegal_reason = info.get("illegal_reason") or "non_yolo_json"
            ver = self._create_illegal_version_row(
                db,
                ds,
                message=message or "illegal_upload",
                created_by=created_by,
                illegal_reason=illegal_reason,
            )
        else:
            # Update dataset format if detected
            if detected_format and detected_format != "yolo":
                ds.format = detected_format

            ver = None
            if create_version:
                ver = self._create_version_row(
                    db,
                    ds,
                    message=message or "upload",
                    created_by=created_by,
                    create_snapshot=False,
                )
                if activate:
                    ds.active_version_id = ver.version_id

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id) if ver is not None else None,
                    event_type="upload_archive",
                    message=message or "Dataset archive uploaded",
                    created_by=created_by,
                    data={"storage_path": ds.storage_path, "dataset_type": str(getattr(ds.dataset_type, "value", ds.dataset_type))},
                )
            )
        except Exception:
            pass

        db.commit()
        db.refresh(ds)
        if ver is not None:
            db.refresh(ver)
        if split_enabled:
            if not create_version:
                raise ValidationError("split_enabled requires create_version=true")
            if ds.dataset_type != DatasetType.DETECTION:
                raise ValidationError("Split is only supported for detection datasets")
            if str(getattr(ds, "format", "") or "") == "illegal":
                raise ValidationError("Cannot split illegal dataset; convert it first")
            if ver is None:
                raise ValidationError("Dataset version not created")
            try:
                self.split_dataset(
                    db,
                    int(ds.dataset_id),
                    version_id=int(ver.version_id),
                    train_ratio=float(split_train_ratio) if split_train_ratio is not None else 0.9,
                    val_ratio=split_val_ratio,
                    test_ratio=split_test_ratio,
                    seed=split_seed,
                    shuffle=bool(split_shuffle) if split_shuffle is not None else True,
                    overwrite=bool(split_overwrite) if split_overwrite is not None else True,
                    trigger="upload",
                )
            except Exception as e:
                try:
                    db.add(
                        DatasetEvent(
                            dataset_id=int(ds.dataset_id),
                            version_id=int(ver.version_id),
                            event_type="split_failed",
                            message="Dataset split failed",
                            data={"trigger": "upload", "error": str(e)},
                        )
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                if isinstance(e, ValidationError):
                    raise
                raise ValidationError("Failed to split dataset") from e
        return ds, ver

    async def upload_dataset_archive_async(
        self,
        db: Session,
        dataset_id: int,
        *,
        file,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_version: bool = True,
        activate: bool = True,
        split_enabled: bool = False,
        split_train_ratio: float | None = None,
        split_val_ratio: float | None = None,
        split_test_ratio: float | None = None,
        split_seed: int | None = None,
        split_shuffle: bool | None = None,
        split_overwrite: bool | None = None,
    ):
        """
        Async wrapper for large uploads.

        FastAPI async endpoints should not do heavy blocking I/O in the event loop;
        we offload archive extraction to a threadpool and keep DB work in the request
        thread.
        """
        ds = self.get_dataset(db, dataset_id)

        dataset_root = resolve_dataset_path(ds.storage_path)
        base = settings.datasets_dir.resolve()
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")

        _path, info = await run_in_threadpool(FileService().upload_dataset_into_existing, file, dataset_root, ds.dataset_type)

        detected_format = info.get("format")
        if detected_format in ("labelme", "unknown_json"):
            ds.format = "illegal"
            illegal_reason = info.get("illegal_reason") or "non_yolo_json"
            ver = self._create_illegal_version_row(
                db,
                ds,
                message=message or "illegal_upload",
                created_by=created_by,
                illegal_reason=illegal_reason,
            )
        else:
            # Update dataset format if detected
            if detected_format and detected_format != "yolo":
                ds.format = detected_format

            ver = None
            if create_version:
                ver = self._create_version_row(
                    db,
                    ds,
                    message=message or "upload",
                    created_by=created_by,
                    create_snapshot=False,
                )
                if activate:
                    ds.active_version_id = ver.version_id

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id) if ver is not None else None,
                    event_type="upload_archive",
                    message=message or "Dataset archive uploaded",
                    created_by=created_by,
                    data={"storage_path": ds.storage_path, "dataset_type": str(getattr(ds.dataset_type, "value", ds.dataset_type))},
                )
            )
        except Exception:
            pass

        db.commit()
        db.refresh(ds)
        if ver is not None:
            db.refresh(ver)
        if split_enabled:
            if not create_version:
                raise ValidationError("split_enabled requires create_version=true")
            if ds.dataset_type != DatasetType.DETECTION:
                raise ValidationError("Split is only supported for detection datasets")
            if str(getattr(ds, "format", "") or "") == "illegal":
                raise ValidationError("Cannot split illegal dataset; convert it first")
            if ver is None:
                raise ValidationError("Dataset version not created")
            try:
                self.split_dataset(
                    db,
                    int(ds.dataset_id),
                    version_id=int(ver.version_id),
                    train_ratio=float(split_train_ratio) if split_train_ratio is not None else 0.9,
                    val_ratio=split_val_ratio,
                    test_ratio=split_test_ratio,
                    seed=split_seed,
                    shuffle=bool(split_shuffle) if split_shuffle is not None else True,
                    overwrite=bool(split_overwrite) if split_overwrite is not None else True,
                    trigger="upload",
                )
            except Exception as e:
                try:
                    db.add(
                        DatasetEvent(
                            dataset_id=int(ds.dataset_id),
                            version_id=int(ver.version_id),
                            event_type="split_failed",
                            message="Dataset split failed",
                            data={"trigger": "upload", "error": str(e)},
                        )
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                if isinstance(e, ValidationError):
                    raise
                raise ValidationError("Failed to split dataset") from e
        return ds, ver

    def append_dataset_archive(
        self,
        db: Session,
        dataset_id: int,
        *,
        file,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_version: bool = True,
        activate: bool = True,
    ):
        """Append ZIP archive contents to an existing (possibly non-empty) dataset."""
        ds = self.get_dataset(db, dataset_id)

        dataset_root = resolve_dataset_path(ds.storage_path)
        base = settings.datasets_dir.resolve()
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")

        # Get class info from the append operation
        _, class_info = FileService().append_dataset_archive(file, dataset_root, ds.dataset_type)
        added_classes = class_info.get("added_classes", [])
        total_classes = class_info.get("total_classes", 0)

        ver = None
        if create_version:
            ver = self._create_version_row(
                db,
                ds,
                message=message or "append",
                created_by=created_by,
                create_snapshot=False,
            )
            if activate:
                ds.active_version_id = ver.version_id

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id) if ver is not None else None,
                    event_type="append_archive",
                    message=message or "Dataset archive appended",
                    created_by=created_by,
                    data={
                        "storage_path": ds.storage_path,
                        "dataset_type": str(getattr(ds.dataset_type, "value", ds.dataset_type)),
                        "added_classes": added_classes,
                        "total_classes": total_classes,
                    },
                )
            )
        except Exception:
            pass

        db.commit()
        db.refresh(ds)
        if ver is not None:
            db.refresh(ver)
        return ds, ver

    async def append_dataset_archive_async(
        self,
        db: Session,
        dataset_id: int,
        *,
        file,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_version: bool = True,
        activate: bool = True,
    ):
        ds = self.get_dataset(db, dataset_id)

        dataset_root = resolve_dataset_path(ds.storage_path)
        base = settings.datasets_dir.resolve()
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")

        # Get class info from the append operation (in threadpool).
        _path, class_info = await run_in_threadpool(FileService().append_dataset_archive, file, dataset_root, ds.dataset_type)
        added_classes = class_info.get("added_classes", [])
        total_classes = class_info.get("total_classes", 0)

        ver = None
        if create_version:
            ver = self._create_version_row(
                db,
                ds,
                message=message or "append",
                created_by=created_by,
                create_snapshot=False,
            )
            if activate:
                ds.active_version_id = ver.version_id

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id) if ver is not None else None,
                    event_type="append_archive",
                    message=message or "Dataset archive appended",
                    created_by=created_by,
                    data={
                        "storage_path": ds.storage_path,
                        "dataset_type": str(getattr(ds.dataset_type, "value", ds.dataset_type)),
                        "added_classes": added_classes,
                        "total_classes": total_classes,
                    },
                )
            )
        except Exception:
            pass

        db.commit()
        db.refresh(ds)
        if ver is not None:
            db.refresh(ver)
        return ds, ver

    @classmethod
    def _illegal_preset_file_path(cls) -> Path:
        return (settings.temp_dir / cls._illegal_preset_file_name).resolve(strict=False)

    @staticmethod
    def _leaf_label(label: str, separator: str = "%") -> str:
        parts = [p.strip() for p in str(label or "").split(separator) if p.strip()]
        return parts[-1] if parts else str(label or "").strip()

    @classmethod
    def _normalize_illegal_preset_payload(cls, payload: dict | None) -> dict:
        data = payload if isinstance(payload, dict) else {}

        detection_rows: list[dict[str, str]] = []
        raw_detection = data.get("detection")
        if isinstance(raw_detection, dict):
            for k, v in raw_detection.items():
                src = str(k or "").strip()
                if not src:
                    continue
                tgt = str(v or "").strip() or cls._leaf_label(src)
                detection_rows.append({"source_label": src, "target_label": tgt})
        elif isinstance(raw_detection, list):
            for item in raw_detection:
                if isinstance(item, str):
                    src = str(item).strip()
                    if not src:
                        continue
                    detection_rows.append({"source_label": src, "target_label": cls._leaf_label(src)})
                    continue
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source_label") or item.get("source") or item.get("raw_label") or "").strip()
                if not src:
                    continue
                tgt = (
                    str(item.get("target_label") or item.get("target") or item.get("mapped_label") or "").strip()
                    or cls._leaf_label(src)
                )
                detection_rows.append({"source_label": src, "target_label": tgt})

        dedup_detection: dict[str, str] = {}
        for row in detection_rows:
            dedup_detection[row["source_label"]] = row["target_label"]
        normalized_detection = [
            {"source_label": src, "target_label": tgt} for src, tgt in dedup_detection.items()
        ]

        classification_rows: list[dict[str, str]] = []
        raw_classification = data.get("classification")
        if isinstance(raw_classification, dict):
            for category, sources in raw_classification.items():
                cat = str(category or "").strip()
                if not cat:
                    continue
                if isinstance(sources, list):
                    for source in sources:
                        src = str(source or "").strip()
                        if not src:
                            continue
                        classification_rows.append(
                            {"category": cat, "source_label": src, "target_label": cat}
                        )
        elif isinstance(raw_classification, list):
            for item in raw_classification:
                if not isinstance(item, dict):
                    continue
                cat = str(item.get("category") or item.get("group") or "").strip()
                src = str(item.get("source_label") or item.get("source") or "").strip()
                if not cat or not src:
                    continue
                tgt = str(item.get("target_label") or item.get("target") or "").strip() or cat
                classification_rows.append({"category": cat, "source_label": src, "target_label": tgt})

        dedup_classification: dict[tuple[str, str], str] = {}
        for row in classification_rows:
            dedup_classification[(row["category"], row["source_label"])] = row["target_label"]
        normalized_classification = [
            {"category": cat, "source_label": src, "target_label": tgt}
            for (cat, src), tgt in dedup_classification.items()
        ]

        updated_at = str(data.get("updated_at") or "").strip() or None

        return {
            "detection": normalized_detection,
            "classification": normalized_classification,
            "updated_at": updated_at,
        }

    @classmethod
    def _load_illegal_preset_payload(cls) -> dict:
        path = cls._illegal_preset_file_path()
        if not path.exists() or not path.is_file():
            return cls._normalize_illegal_preset_payload({})
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return cls._normalize_illegal_preset_payload({})
        return cls._normalize_illegal_preset_payload(payload)

    @classmethod
    def _save_illegal_preset_payload(cls, payload: dict) -> None:
        path = cls._illegal_preset_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def get_illegal_label_presets(self) -> dict:
        with self._illegal_preset_lock:
            return self._load_illegal_preset_payload()

    def save_illegal_label_presets(self, payload: dict | None) -> dict:
        normalized = self._normalize_illegal_preset_payload(payload)
        normalized["updated_at"] = _utcnow().isoformat()
        with self._illegal_preset_lock:
            self._save_illegal_preset_payload(normalized)
        return normalized

    def get_illegal_labels(self, db: Session, dataset_id: int) -> list[str]:
        ds = self.get_dataset(db, dataset_id)
        if ds.dataset_type != DatasetType.DETECTION:
            raise ValidationError("Only detection datasets can have illegal labels extracted")

        ver = None
        if ds.active_version_id is not None:
            ver = self.versions.get(db, int(ds.active_version_id))
        if not ver:
            raise ValidationError("Active version not found")

        meta = ver.meta if isinstance(ver.meta, dict) else {}
        if not meta.get("illegal"):
            raise ValidationError("Dataset is not marked as illegal")

        from train_platform.services.dataset_illegal_convert_service import DatasetIllegalConvertService

        dataset_root = resolve_dataset_path(ds.storage_path)
        return DatasetIllegalConvertService().extract_dataset_labels(dataset_root)

    def update_illegal_labels(self, db: Session, dataset_id: int, label_mapping: dict) -> Dataset:
        ds = self.get_dataset(db, dataset_id)
        ver = None
        if ds.active_version_id is not None:
            ver = self.versions.get(db, int(ds.active_version_id))
        if not ver:
            raise ValidationError("Active version not found")

        meta = ver.meta if isinstance(ver.meta, dict) else {}
        if not meta.get("illegal"):
            raise ValidationError("Dataset is not marked as illegal")

        if not isinstance(label_mapping, dict) or not label_mapping:
            raise ValidationError("label_mapping is required")

        # Expand mapping to include identity entries for mapped targets.
        expanded_mapping: dict = {}
        for k, v in label_mapping.items():
            key = str(k).strip()
            if not key:
                continue
            expanded_mapping[key] = v
        for _k, v in list(expanded_mapping.items()):
            if v is None:
                continue
            v_str = str(v).strip()
            if v_str and v_str != "__DISCARD__":
                expanded_mapping.setdefault(v_str, v_str)

        from train_platform.services.dataset_illegal_convert_service import DatasetIllegalConvertService

        dataset_root = resolve_dataset_path(ds.storage_path)
        DatasetIllegalConvertService().apply_label_mapping(dataset_root, expanded_mapping)

        meta["illegal_label_mapping"] = expanded_mapping
        ver.meta = meta
        db.commit()
        return ds

    @staticmethod
    def _empty_conversion_progress() -> dict:
        return {
            "overall": {
                "processed_images": 0,
                "total_images": 0,
                "current_image_index": 0,
                "current_image_name": "",
                "percent": 0,
            },
            "image": {
                "phase": "scanning",
                "processed_slices": 0,
                "total_slices": 0,
                "percent": 0,
            },
        }

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _build_conversion_progress(self, event: dict | None) -> tuple[dict, str, str]:
        payload = dict(event or {})
        phase = str(payload.get("phase") or "scanning").strip().lower() or "scanning"
        message = str(payload.get("message") or "").strip()

        overall_total = max(0, self._safe_int(payload.get("overall_total_images"), 0))
        overall_processed = max(0, self._safe_int(payload.get("overall_processed_images"), 0))
        if overall_total > 0:
            overall_processed = min(overall_processed, overall_total)
            overall_percent = int(round((overall_processed / overall_total) * 100))
        else:
            overall_percent = 0

        current_image_index = max(0, self._safe_int(payload.get("current_image_index"), 0))
        current_image_name = str(payload.get("current_image_name") or "").strip()

        image_total = max(0, self._safe_int(payload.get("current_slice_total"), 0))
        image_processed = max(0, self._safe_int(payload.get("current_slice_processed"), 0))
        if image_total > 0:
            image_processed = min(image_processed, image_total)
            image_percent = int(round((image_processed / image_total) * 100))
        else:
            image_percent = 0

        progress = {
            "overall": {
                "processed_images": int(overall_processed),
                "total_images": int(overall_total),
                "current_image_index": int(current_image_index),
                "current_image_name": current_image_name,
                "percent": int(max(0, min(100, overall_percent))),
            },
            "image": {
                "phase": phase,
                "processed_slices": int(image_processed),
                "total_slices": int(image_total),
                "percent": int(max(0, min(100, image_percent))),
            },
        }
        signature = (
            f"{phase}|{overall_processed}|{overall_total}|{current_image_index}|{current_image_name}|"
            f"{image_processed}|{image_total}|{message}"
        )
        return progress, signature, message

    def convert_illegal_dataset(
        self,
        db: Session,
        dataset_id: int,
        *,
        label_strategy: str,
        label_level: Optional[int],
        label_separator: Optional[str],
        label_mapping: Optional[dict] = None,
        # Slice / crop parameters
        slice_size: int | None = None,
        overlap: float | None = None,
        padding: int | None = None,
        min_area_ratio: float | None = None,
        min_visibility: float | None = None,
        min_pixel_size: int | None = None,
        negative_ratio: float | None = None,
        empty_positive_action: str | None = None,
        # Split parameters
        split_enabled: bool | None = None,
        split_train_ratio: float | None = None,
        split_val_ratio: float | None = None,
        split_test_ratio: float | None = None,
        split_seed: int | None = None,
        split_shuffle: bool | None = None,
        split_overwrite: bool | None = None,
    ) -> dict:
        ds = self.get_dataset(db, dataset_id)
        if ds.dataset_type != DatasetType.DETECTION:
            raise ValidationError("Only detection datasets can be converted")

        ver: DatasetVersion | None = None
        if ds.active_version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )
        if not ver:
            raise ValidationError("Active version not found")

        meta = ver.meta if isinstance(ver.meta, dict) else {}
        if not meta.get("illegal"):
            raise ValidationError("Dataset is not marked as illegal")
        illegal_reason = str(meta.get("illegal_reason") or "")
        if illegal_reason != "labelme_json":
            raise ValidationError("Unsupported illegal format")

        strategy = str(label_strategy or "").strip().lower()
        if strategy not in ("full", "leaf", "root", "level", "mapping"):
            raise ValidationError("label_strategy must be one of: full, leaf, root, level, mapping")
        level = int(label_level) if label_level is not None else None
        if strategy == "level" and (level is None or level < 1):
            raise ValidationError("label_level is required when label_strategy=level")
        # Always persist incoming label_mapping so it is available to the
        # background thread (which reads from meta) regardless of strategy.
        incoming_mapping = label_mapping if isinstance(label_mapping, dict) and label_mapping else None
        if incoming_mapping is not None:
            meta["illegal_label_mapping"] = incoming_mapping

        if strategy == "mapping":
            stored_mapping = meta.get("illegal_label_mapping")
            mapping_to_use = incoming_mapping or stored_mapping
            if not isinstance(mapping_to_use, dict) or not mapping_to_use:
                raise ValidationError("label_mapping is required when label_strategy=mapping (save mapping first or pass label_mapping)")

        conv = dict(meta.get("conversion") or {})
        if str(conv.get("status")) in ("queued", "running"):
            raise ConflictError("Conversion is already in progress")

        job_id = uuid.uuid4().hex
        conv.update(
            {
                "status": "queued",
                "job_id": job_id,
                "label_strategy": strategy,
                "label_level": level,
                "label_separator": label_separator or "%",
                "seq": 0,
                "updated_at": _utcnow().isoformat(),
                "phase": "scanning",
                "progress": self._empty_conversion_progress(),
            }
        )
        conv.pop("error_message", None)
        conv.pop("output_version_id", None)
        conv["split"] = {
            "enabled": bool(split_enabled),
            "train_ratio": float(split_train_ratio) if split_train_ratio is not None else None,
            "val_ratio": float(split_val_ratio) if split_val_ratio is not None else None,
            "test_ratio": float(split_test_ratio) if split_test_ratio is not None else None,
            "seed": int(split_seed) if split_seed is not None else None,
            "shuffle": bool(split_shuffle) if split_shuffle is not None else None,
            "overwrite": bool(split_overwrite) if split_overwrite is not None else None,
        }
        # Store slice/crop overrides for the background thread
        slice_config = {}
        if slice_size is not None:
            slice_config["slice_size"] = int(slice_size)
        if overlap is not None:
            slice_config["overlap"] = float(overlap)
        if padding is not None:
            slice_config["padding"] = int(padding)
        if min_area_ratio is not None:
            slice_config["min_area_ratio"] = float(min_area_ratio)
        if min_visibility is not None:
            slice_config["min_visibility"] = float(min_visibility)
        if min_pixel_size is not None:
            slice_config["min_pixel_size"] = int(min_pixel_size)
        if negative_ratio is not None:
            slice_config["negative_ratio"] = float(negative_ratio)
        if empty_positive_action is not None:
            slice_config["empty_positive_action"] = str(empty_positive_action)
        if slice_config:
            conv["slice_config"] = slice_config
        meta["conversion"] = conv
        ver.meta = meta
        db.commit()

        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id),
                    event_type="conversion_queued",
                    message="Illegal dataset conversion queued",
                    data={"job_id": job_id},
                )
            )
            db.commit()
        except Exception:
            db.rollback()

        t = threading.Thread(
            target=self._run_illegal_conversion_job,
            args=(int(ds.dataset_id), int(ver.version_id), job_id, strategy, level, label_separator or "%"),
            daemon=True,
        )
        t.start()

        return {"job_id": job_id, "status": "queued"}

    def _run_illegal_conversion_job(
        self,
        dataset_id: int,
        version_id: int,
        job_id: str,
        label_strategy: str,
        label_level: Optional[int],
        label_separator: str,
    ) -> None:
        logger.info(f"[Conversion] Starting illegal conversion job {job_id} for dataset {dataset_id}")

        db = SessionLocal()
        try:
            ds = self.get_dataset(db, dataset_id)
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )
            if not ver:
                raise ValidationError("Illegal version not found")

            meta = ver.meta if isinstance(ver.meta, dict) else {}
            conv = dict(meta.get("conversion") or {})
            conv.update(
                {
                    "status": "running",
                    "job_id": job_id,
                    "phase": "scanning",
                    "updated_at": _utcnow().isoformat(),
                }
            )
            conv["progress"] = conv.get("progress") if isinstance(conv.get("progress"), dict) else self._empty_conversion_progress()
            conv["seq"] = self._safe_int(conv.get("seq"), 0)
            conv.pop("error_message", None)
            meta["conversion"] = conv
            ver.meta = meta
            db.commit()

            last_progress_commit_ts = 0.0
            last_progress_signature = ""
            last_phase = str(conv.get("phase") or "").strip().lower()

            def _persist_progress_event(event: dict, *, force: bool = False) -> None:
                nonlocal last_progress_commit_ts, last_progress_signature, last_phase, conv, meta

                progress, signature, message = self._build_conversion_progress(event)
                phase = str(progress.get("image", {}).get("phase") or "scanning").strip().lower()
                image_info = progress.get("image") or {}
                slice_processed = self._safe_int(image_info.get("processed_slices"), 0)
                slice_total = self._safe_int(image_info.get("total_slices"), 0)

                phase_changed = phase != last_phase
                first_progress = self._safe_int(conv.get("seq"), 0) <= 0
                stage_boundary = slice_total > 0 and slice_processed in (0, slice_total)
                should_force = bool(force or first_progress or phase_changed or stage_boundary)

                if signature == last_progress_signature and not should_force:
                    return

                now_ts = time.monotonic()
                if not should_force and (now_ts - last_progress_commit_ts) < 0.5:
                    return

                conv["progress"] = progress
                conv["phase"] = phase
                conv["updated_at"] = _utcnow().isoformat()
                conv["seq"] = self._safe_int(conv.get("seq"), 0) + 1
                if message:
                    conv["message"] = message
                meta["conversion"] = conv
                ver.meta = meta
                db.commit()

                last_progress_signature = signature
                last_progress_commit_ts = now_ts
                last_phase = phase

            def _on_progress(event: dict) -> None:
                try:
                    _persist_progress_event(event, force=False)
                except Exception as progress_err:
                    logger.warning(
                        "[Conversion] Failed to persist progress for job %s: %s",
                        job_id,
                        progress_err,
                    )

            try:
                db.add(
                    DatasetEvent(
                        dataset_id=int(ds.dataset_id),
                        version_id=int(ver.version_id),
                        event_type="conversion_running",
                        message="Illegal dataset conversion running",
                        data={"job_id": job_id},
                    )
                )
                db.commit()
            except Exception:
                db.rollback()

            dataset_root = resolve_dataset_path(ds.storage_path)

            from train_platform.services.dataset_illegal_convert_service import DatasetIllegalConvertService

            label_mapping = meta.get("illegal_label_mapping")
            slice_config = conv.get("slice_config") if isinstance(conv, dict) else None

            svc = DatasetIllegalConvertService()
            svc.convert_dataset(
                dataset_root,
                label_strategy=label_strategy,
                label_level=label_level,
                label_separator=label_separator,
                label_mapping=label_mapping,
                slice_config=slice_config,
                progress_cb=_on_progress,
            )

            ds.format = "yolo"
            new_ver = self._create_version_row(
                db,
                ds,
                message="conversion",
                created_by=None,
                create_snapshot=False,
            )
            # Clear the illegal flag on the newly created version so downstream
            # workflows (training, export, etc.) treat it as a normal YOLO dataset.
            new_meta = new_ver.meta if isinstance(new_ver.meta, dict) else {}
            new_meta.pop("illegal", None)
            new_meta.pop("illegal_reason", None)
            new_ver.meta = new_meta
            ds.active_version_id = new_ver.version_id

            try:
                conv_split = conv.get("split") if isinstance(conv, dict) else None
                if isinstance(conv_split, dict) and conv_split.get("enabled"):
                    self.split_dataset(
                        db,
                        int(ds.dataset_id),
                        version_id=int(new_ver.version_id),
                        train_ratio=float(conv_split.get("train_ratio") or 0.9),
                        val_ratio=conv_split.get("val_ratio"),
                        test_ratio=conv_split.get("test_ratio"),
                        seed=conv_split.get("seed"),
                        shuffle=conv_split.get("shuffle") if conv_split.get("shuffle") is not None else True,
                        overwrite=conv_split.get("overwrite") if conv_split.get("overwrite") is not None else True,
                        trigger="illegal_conversion",
                    )
                    conv_split["status"] = "completed"
                elif isinstance(conv_split, dict):
                    conv_split["status"] = "skipped"
            except Exception as e:
                if isinstance(conv_split, dict):
                    conv_split["status"] = "failed"
                    conv_split["error_message"] = str(e)
                try:
                    db.add(
                        DatasetEvent(
                            dataset_id=int(ds.dataset_id),
                            version_id=int(new_ver.version_id),
                            event_type="split_failed",
                            message="Dataset split failed",
                            data={"trigger": "illegal_conversion", "error": str(e)},
                        )
                    )
                    db.commit()
                except Exception:
                    db.rollback()

            overall = dict((conv.get("progress") or {}).get("overall") or {})
            image = dict((conv.get("progress") or {}).get("image") or {})
            overall_total = max(0, self._safe_int(overall.get("total_images"), 0))
            if overall_total > 0:
                overall["processed_images"] = overall_total
                overall["percent"] = 100
            image_total = max(0, self._safe_int(image.get("total_slices"), 0))
            if image_total > 0:
                image["processed_slices"] = image_total
                image["percent"] = 100
            image["phase"] = "done"
            conv["progress"] = {
                "overall": {
                    "processed_images": max(0, self._safe_int(overall.get("processed_images"), 0)),
                    "total_images": overall_total,
                    "current_image_index": max(0, self._safe_int(overall.get("current_image_index"), 0)),
                    "current_image_name": str(overall.get("current_image_name") or ""),
                    "percent": max(0, min(100, self._safe_int(overall.get("percent"), 0))),
                },
                "image": {
                    "phase": "done",
                    "processed_slices": max(0, self._safe_int(image.get("processed_slices"), 0)),
                    "total_slices": image_total,
                    "percent": max(0, min(100, self._safe_int(image.get("percent"), 0))),
                },
            }
            conv.update(
                {
                    "status": "completed",
                    "output_version_id": int(new_ver.version_id),
                    "phase": "done",
                    "updated_at": _utcnow().isoformat(),
                    "seq": self._safe_int(conv.get("seq"), 0) + 1,
                }
            )
            conv.pop("error_message", None)
            meta["conversion"] = conv
            ver.meta = meta
            db.commit()
            db.refresh(ds)

            try:
                db.add(
                    DatasetEvent(
                        dataset_id=int(ds.dataset_id),
                        version_id=int(new_ver.version_id),
                        event_type="conversion_completed",
                        message="Illegal dataset conversion completed",
                        data={"job_id": job_id},
                    )
                )
                db.commit()
            except Exception:
                db.rollback()
        except Exception as e:
            logger.error(f"Illegal dataset conversion failed for dataset {dataset_id}: {e}", exc_info=True)
            try:
                ver = (
                    db.query(DatasetVersion)
                    .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(dataset_id))
                    .first()
                )
                if ver:
                    meta = ver.meta if isinstance(ver.meta, dict) else {}
                    conv = dict(meta.get("conversion") or {})
                    progress = conv.get("progress") if isinstance(conv.get("progress"), dict) else self._empty_conversion_progress()
                    image = progress.get("image") if isinstance(progress.get("image"), dict) else {}
                    image = dict(image)
                    image["phase"] = "failed"
                    progress["image"] = image
                    conv.update(
                        {
                            "status": "failed",
                            "phase": "failed",
                            "error_message": str(e),
                            "updated_at": _utcnow().isoformat(),
                            "seq": self._safe_int(conv.get("seq"), 0) + 1,
                            "progress": progress,
                        }
                    )
                    meta["conversion"] = conv
                    ver.meta = meta
                    db.commit()
                try:
                    db.add(
                        DatasetEvent(
                            dataset_id=int(dataset_id),
                            version_id=int(version_id),
                            event_type="conversion_failed",
                            message="Illegal dataset conversion failed",
                            data={"job_id": job_id, "error": str(e)},
                        )
                    )
                    db.commit()
                except Exception:
                    db.rollback()
            except Exception:
                db.rollback()
        finally:
            db.close()

    # --------------------
    # rename class labels (YOLO dataset)
    # --------------------
    def rename_classes(
        self,
        db: Session,
        dataset_id: int,
        rename_map: Dict[str, str],
    ) -> dict:
        """Rename class names in classes.txt & data.yaml of a converted YOLO dataset.

        Only the display names are updated; class_id (line index in classes.txt /
        numeric id in label .txt files) stays unchanged.

        Args:
            rename_map: ``{old_name: new_name}`` — only classes that need
                renaming should be listed.

        Returns:
            dict with ``renamed``, ``total_classes``, ``class_names``.
        """
        if not isinstance(rename_map, dict) or not rename_map:
            raise ValidationError("rename_map must be a non-empty dict")

        ds = self.get_dataset(db, dataset_id)
        fmt = str(getattr(ds, "format", "") or "").lower()
        if fmt not in ("yolo",):
            raise ValidationError("Only YOLO-format datasets support class renaming")

        dataset_root = resolve_dataset_path(ds.storage_path)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ValidationError("Dataset directory not found on disk")

        # --- locate classes.txt ---
        classes_path = dataset_root / "classes.txt"
        if not classes_path.exists():
            # Try alternative names
            for alt in ("class_names.txt", "obj.names", "names.txt"):
                alt_p = dataset_root / alt
                if alt_p.exists():
                    classes_path = alt_p
                    break
        if not classes_path.exists():
            raise ValidationError("classes.txt not found in dataset directory")

        # Read current class list (one name per line, preserve order = class_id)
        lines = classes_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        class_names: list[str] = []
        for line in lines:
            s = line.strip()
            if s and not s.startswith("#"):
                class_names.append(s)

        if not class_names:
            raise ValidationError("classes.txt is empty")

        # Validate rename_map keys exist in current class list
        current_set = set(class_names)
        unknown = [k for k in rename_map if k not in current_set]
        if unknown:
            sample = ", ".join(unknown[:10])
            raise ValidationError(f"Unknown class names (not in classes.txt): {sample}")

        # Check for duplicate target names
        new_names = list(class_names)  # copy
        renamed_count = 0
        for i, name in enumerate(class_names):
            if name in rename_map:
                target = str(rename_map[name]).strip()
                if not target:
                    raise ValidationError(f"New name for '{name}' must not be empty")
                new_names[i] = target
                renamed_count += 1

        # Check uniqueness after rename
        seen: set[str] = set()
        for n in new_names:
            if n in seen:
                raise ValidationError(f"Duplicate class name after rename: '{n}'")
            seen.add(n)

        if renamed_count == 0:
            return {
                "renamed": 0,
                "total_classes": len(new_names),
                "class_names": new_names,
            }

        # --- write classes.txt ---
        with open(classes_path, "w", encoding="utf-8") as f:
            for n in new_names:
                f.write(n + "\n")

        # --- update data.yaml ---
        yaml_path = dataset_root / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["names"] = new_names
                cfg["nc"] = len(new_names)
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            except Exception:
                # If data.yaml is corrupt, regenerate from scratch
                FileService()._create_yolo_data_yaml(dataset_root, yaml_path)

        # --- record event ---
        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ds.active_version_id) if ds.active_version_id is not None else None,
                    event_type="rename_classes",
                    message=f"Renamed {renamed_count} class(es)",
                    data={"rename_map": rename_map, "class_names": new_names},
                )
            )
            db.commit()
        except Exception:
            db.rollback()

        return {
            "renamed": renamed_count,
            "total_classes": len(new_names),
            "class_names": new_names,
        }

    # --------------------
    # dataset uploads / events
    # --------------------
    def list_events(
        self,
        db: Session,
        dataset_id: int,
        *,
        skip: int = 0,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> list[DatasetEvent]:
        self.get_dataset(db, dataset_id)
        return self.events.list_by_dataset(db, int(dataset_id), event_type=event_type, skip=skip, limit=limit)

    def upload_images(
        self,
        db: Session,
        dataset_id: int,
        *,
        files: list,
        relative_dir: str = "images",
        labels: list | None = None,
        labels_relative_dir: str | None = None,
        require_labels: bool = True,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_version: bool = True,
        create_snapshot: bool = False,
        activate: bool = True,
        max_return_files: int = 200,
    ) -> dict:
        """
        Append images (+ YOLO labels for detection datasets) to an existing dataset directory and optionally create a dataset version.

        For DETECTION datasets:
        - Each uploaded image must have a matching label file `<stem>.txt`.
        - New class ids are NOT allowed (label class_id must be < dataset's current `nc` from data.yaml).

        `relative_dir` is a path under the dataset root (e.g. `images/train`).
        """
        ds = self.get_dataset(db, dataset_id)

        base = settings.datasets_dir.resolve()
        # Uploads always modify the dataset root (not a historical snapshot).
        root_token = str(ds.storage_path)
        dataset_root = resolve_dataset_path(root_token).resolve(strict=False)
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ValidationError(f"Dataset path does not exist: {dataset_root}")

        if not files:
            raise ValidationError("files is required")

        labels = list(labels or [])
        use_labels = ds.dataset_type == DatasetType.DETECTION
        if use_labels:
            require_labels = True

        def _sanitize_relative_dir(raw: str | None, *, field: str) -> tuple[str, Path]:
            s = str(raw or "").strip().replace("\\", "/").strip("/\\")
            if s.startswith(".versions"):
                raise ValidationError(f"{field} cannot start with '.versions'")
            p = Path(s) if s else Path()
            if p.is_absolute() or ".." in p.parts:
                raise ValidationError(f"Invalid {field}")
            return s, p

        rel_dir, rel_dir_path = _sanitize_relative_dir(relative_dir, field="relative_dir")

        dest_dir = (dataset_root / rel_dir_path).resolve(strict=False)
        if dataset_root != dest_dir and dataset_root not in dest_dir.parents:
            raise ValidationError("Invalid relative_dir (outside dataset root)")
        dest_dir.mkdir(parents=True, exist_ok=True)

        labels_rel_dir = ""
        labels_dest_dir: Path | None = None
        if use_labels:
            if require_labels and not labels:
                raise ValidationError("labels are required for detection datasets")

            def _derive_labels_dir(images_rel_dir: str) -> str:
                parts = list(Path(images_rel_dir).parts) if images_rel_dir else []
                for i, seg in enumerate(parts):
                    if str(seg).lower() == "images":
                        parts[i] = "labels"
                        return Path(*parts).as_posix()
                return "labels"

            labels_rel_dir = _derive_labels_dir(rel_dir) if labels_relative_dir is None else str(labels_relative_dir)
            labels_rel_dir, labels_rel_dir_path = _sanitize_relative_dir(labels_rel_dir, field="labels_relative_dir")
            labels_dest_dir = (dataset_root / labels_rel_dir_path).resolve(strict=False)
            if dataset_root != labels_dest_dir and dataset_root not in labels_dest_dir.parents:
                raise ValidationError("Invalid labels_relative_dir (outside dataset root)")
            labels_dest_dir.mkdir(parents=True, exist_ok=True)
        else:
            if labels:
                raise ValidationError("labels upload is only supported for DETECTION datasets")

        allowed_exts = IMAGE_EXTS

        # Pre-validate all filenames/extensions to avoid partial uploads.
        planned_image_names: list[str] = []
        for f in files:
            filename = getattr(f, "filename", None) or ""
            safe_name = Path(str(filename)).name.strip()
            ext = Path(safe_name).suffix.lower()
            if not ext:
                ext = ".jpg"
                safe_name = f"{uuid.uuid4().hex}{ext}"
            if ext not in allowed_exts:
                raise ValidationError(f"Unsupported image extension: {ext}")
            planned_image_names.append(safe_name)

        # Reject duplicate names in the same request (case-insensitive to match Windows semantics).
        seen: dict[str, str] = {}
        dup_keys: set[str] = set()
        for n in planned_image_names:
            k = str(n).casefold()
            if k in seen:
                dup_keys.add(k)
            else:
                seen[k] = n
        if dup_keys:
            dups = [seen[k] for k in sorted(dup_keys)]
            raise ConflictError(f"Duplicate filenames in upload: {', '.join(dups[:20])}")

        # If uploading labels, we must also reject duplicate stems (a.jpg + a.png would collide on a.txt).
        if use_labels:
            seen_stem: dict[str, str] = {}
            dup_stem: set[str] = set()
            for n in planned_image_names:
                k = Path(n).stem.casefold()
                if k in seen_stem:
                    dup_stem.add(k)
                else:
                    seen_stem[k] = n
            if dup_stem:
                dups = [seen_stem[k] for k in sorted(dup_stem)]
                raise ConflictError(f"Duplicate image stems in upload (labels would conflict): {', '.join(dups[:20])}")

        # Reject collisions with existing files in destination dir.
        try:
            existing_cf = {p.name.casefold() for p in dest_dir.iterdir() if p.is_file()}
        except Exception:
            existing_cf = set()
        conflicts = [n for n in planned_image_names if n.casefold() in existing_cf]
        if conflicts:
            raise ConflictError(f"File(s) already exist in dataset: {', '.join(conflicts[:20])}")

        planned_label_names: list[str] = []
        label_by_name_cf: dict[str, object] = {}
        expected_label_names: list[str] = []
        expected_label_cf: set[str] = set()

        if use_labels:
            # Pre-validate label filenames/extensions.
            for f in labels:
                filename = getattr(f, "filename", None) or ""
                safe_name = Path(str(filename)).name.strip()
                if not safe_name:
                    raise ValidationError("Label filename is required")
                ext = Path(safe_name).suffix.lower()
                if ext != ".txt":
                    raise ValidationError(f"Unsupported label extension: {ext}")
                planned_label_names.append(safe_name)

            # Reject duplicate label names in the same request (case-insensitive).
            seen_l: dict[str, str] = {}
            dup_l: set[str] = set()
            for n in planned_label_names:
                k = str(n).casefold()
                if k in seen_l:
                    dup_l.add(k)
                else:
                    seen_l[k] = n
            if dup_l:
                dups = [seen_l[k] for k in sorted(dup_l)]
                raise ConflictError(f"Duplicate label filenames in upload: {', '.join(dups[:20])}")

            if labels_dest_dir is None:
                raise ValidationError("Internal error: labels_dest_dir is not set")

            # Reject collisions with existing label files.
            try:
                existing_labels_cf = {p.name.casefold() for p in labels_dest_dir.iterdir() if p.is_file()}
            except Exception:
                existing_labels_cf = set()
            label_conflicts = [n for n in planned_label_names if n.casefold() in existing_labels_cf]
            if label_conflicts:
                raise ConflictError(f"Label file(s) already exist in dataset: {', '.join(label_conflicts[:20])}")

            for f, n in zip(labels, planned_label_names):
                label_by_name_cf[str(n).casefold()] = f

            # Compute expected label names based on the uploaded images.
            for img_name in planned_image_names:
                expected = f"{Path(img_name).stem}.txt"
                expected_label_names.append(expected)
                expected_label_cf.add(expected.casefold())

            if labels:
                missing = [n for n in expected_label_names if n.casefold() not in label_by_name_cf]
                if missing:
                    raise ValidationError(f"Missing label files for uploaded images: {', '.join(missing[:20])}")

                extra = [n for n in planned_label_names if n.casefold() not in expected_label_cf]
                if extra:
                    raise ValidationError(f"Unexpected label files (no matching image in this upload): {', '.join(extra[:20])}")

        max_class_id: int | None = None
        nc_before: int | None = None

        # Validate label class ids BEFORE writing any files.
        if use_labels:
            if labels_dest_dir is None:
                raise ValidationError("Internal error: labels_dest_dir is not set")

            data_yaml_path = find_yolo_dataset_yaml(dataset_root, dataset_name=str(getattr(ds, "name", "") or "") or None)
            if data_yaml_path is None or not data_yaml_path.exists():
                # We need an existing schema (nc/names) to validate "no new classes".
                # If the dataset doesn't ship with a yaml, create a minimal data.yaml.
                data_yaml_path = (dataset_root / "data.yaml").resolve(strict=False)
                if not data_yaml_path.exists():
                    try:
                        from train_platform.services.file_service import FileService

                        FileService()._create_yolo_data_yaml(dataset_root, data_yaml_path)
                    except Exception:
                        raise ValidationError("Dataset YAML not found; cannot validate labels/classes")

            try:
                cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}

            base_names = self._normalize_yolo_names(cfg.get("names"), cfg.get("nc"))
            nc_before = int(len(base_names))
            if nc_before <= 0:
                raise ValidationError("Dataset YAML missing valid 'names'/'nc'; cannot validate labels/classes")

            max_class_id = self._scan_yolo_label_max_class_id_from_uploads(labels)
            if max_class_id is not None and int(max_class_id) >= int(nc_before):
                raise ValidationError(
                    f"New class ids are not allowed in this dataset (max_class_id={int(max_class_id)}, nc={int(nc_before)}). "
                    "Please create a new dataset."
                )

        saved_abs: list[Path] = []
        saved_images_rel: list[str] = []
        saved_labels_rel: list[str] = []
        total_bytes = 0
        created_version_dir: Path | None = None

        try:
            for f, safe_name in zip(files, planned_image_names):
                out_path = (dest_dir / safe_name).resolve(strict=False)
                if dest_dir != out_path and dest_dir not in out_path.parents:
                    raise ValidationError("Unsafe upload path")
                if out_path.exists():
                    # Shouldn't happen due to pre-check, but keep a hard guard.
                    raise ConflictError(f"File already exists: {safe_name}")
                with open(out_path, "wb") as dst:
                    src = getattr(f, "file", None)
                    if src is None:
                        raise ValidationError("Invalid upload file object (missing .file)")
                    try:
                        src.seek(0)
                    except Exception:
                        pass
                    shutil.copyfileobj(src, dst)

                try:
                    st = out_path.stat()
                    total_bytes += int(st.st_size)
                except Exception:
                    pass

                saved_abs.append(out_path)
                try:
                    saved_images_rel.append(out_path.relative_to(dataset_root).as_posix())
                except Exception:
                    saved_images_rel.append(out_path.name)

            if use_labels and labels_dest_dir is not None and labels:
                for expected in expected_label_names:
                    f = label_by_name_cf.get(expected.casefold())
                    if f is None:
                        # Shouldn't happen due to pre-check.
                        raise ValidationError(f"Missing label file: {expected}")

                    out_path = (labels_dest_dir / expected).resolve(strict=False)
                    if labels_dest_dir != out_path and labels_dest_dir not in out_path.parents:
                        raise ValidationError("Unsafe label upload path")
                    if out_path.exists():
                        raise ConflictError(f"Label file already exists: {expected}")

                    with open(out_path, "wb") as dst:
                        src = getattr(f, "file", None)
                        if src is None:
                            raise ValidationError("Invalid label upload file object (missing .file)")
                        try:
                            src.seek(0)
                        except Exception:
                            pass
                        shutil.copyfileobj(src, dst)

                    try:
                        st = out_path.stat()
                        total_bytes += int(st.st_size)
                    except Exception:
                        pass

                    saved_abs.append(out_path)
                    try:
                        saved_labels_rel.append(out_path.relative_to(dataset_root).as_posix())
                    except Exception:
                        saved_labels_rel.append(out_path.name)

            version_row: DatasetVersion | None = None
            ver_msg: str | None = None
            if create_version:
                ver_msg = (message or "").strip() or f"upload_images +{len(saved_images_rel)}"
                version_row = self._create_version_row(
                    db,
                    ds,
                    message=ver_msg,
                    created_by=created_by,
                    create_snapshot=bool(create_snapshot),
                )
                created_version_dir = settings.datasets_dir.resolve() / ".versions" / _versions_token(ds) / f"v{int(version_row.version)}"
                if activate:
                    ds.active_version_id = version_row.version_id

            ev = DatasetEvent(
                dataset_id=int(ds.dataset_id),
                version_id=int(version_row.version_id) if version_row else None,
                event_type="upload_images",
                message=(message or "").strip() or ver_msg,
                data={
                    "relative_dir": rel_dir or "",
                    "saved_count": int(len(saved_images_rel)),
                    "saved_files": saved_images_rel[: int(max_return_files)],
                    "truncated": bool(len(saved_images_rel) > int(max_return_files)),
                    "total_bytes": int(total_bytes),
                    "labels_relative_dir": labels_rel_dir or None,
                    "saved_label_count": int(len(saved_labels_rel)),
                    "saved_label_files": saved_labels_rel[: int(max_return_files)],
                    "truncated_labels": bool(len(saved_labels_rel) > int(max_return_files)),
                    "max_class_id": int(max_class_id) if max_class_id is not None else None,
                    "nc_before": int(nc_before) if nc_before is not None else None,
                    "nc_after": int(nc_before) if nc_before is not None else None,
                    "added_class_ids": [],
                    "class_names_updated": False,
                    "create_version": bool(create_version),
                    "create_snapshot": bool(create_snapshot),
                    "activate": bool(activate),
                },
                created_by=created_by,
            )
            db.add(ev)
            db.commit()

            db.refresh(ev)
            if version_row:
                db.refresh(version_row)
            db.refresh(ds)

            return {
                "dataset_id": int(ds.dataset_id),
                "event_id": int(ev.event_id),
                "version_id": int(version_row.version_id) if version_row else None,
                "version": int(version_row.version) if version_row else None,
                "active_version_id": int(ds.active_version_id) if ds.active_version_id is not None else None,
                "relative_dir": rel_dir or "",
                "saved_count": int(len(saved_images_rel)),
                "saved_files": saved_images_rel[: int(max_return_files)],
                "truncated": bool(
                    len(saved_images_rel) > int(max_return_files) or len(saved_labels_rel) > int(max_return_files)
                ),
                "total_bytes": int(total_bytes),
                "labels_relative_dir": labels_rel_dir or None,
                "saved_label_count": int(len(saved_labels_rel)),
                "saved_label_files": saved_labels_rel[: int(max_return_files)],
                "max_class_id": int(max_class_id) if max_class_id is not None else None,
                "nc_before": int(nc_before) if nc_before is not None else None,
                "nc_after": int(nc_before) if nc_before is not None else None,
                "added_class_ids": [],
                "class_names_updated": False,
                "created_at": ev.created_at,
            }
        except Exception:
            db.rollback()
            # Best-effort cleanup of uploaded files. (If DB write fails, we do not want to leave partial files.)
            for p in saved_abs:
                try:
                    if p.exists() and p.is_file():
                        p.unlink()
                except Exception:
                    pass
            if created_version_dir is not None:
                try:
                    if created_version_dir.exists() and created_version_dir.is_dir():
                        shutil.rmtree(created_version_dir, ignore_errors=True)
                except Exception:
                    pass
            raise

    def _scan_yolo_label_max_class_id_from_uploads(self, label_files: list) -> int | None:
        """
        Return max class id from YOLO label uploads. Validates that each non-empty line starts with an int >= 0.
        """
        max_id: int | None = None
        for f in label_files:
            src = getattr(f, "file", None)
            if src is None:
                raise ValidationError("Invalid label upload file object (missing .file)")
            try:
                src.seek(0)
            except Exception:
                pass
            try:
                for ln, raw_line in enumerate(src, start=1):
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="ignore")
                    else:
                        line = str(raw_line)
                    s = line.strip()
                    if not s:
                        continue
                    head = s.split()[0]
                    try:
                        cid = int(head)
                    except Exception:
                        name = Path(str(getattr(f, "filename", "") or "label.txt")).name
                        raise ValidationError(f"Invalid label format in {name} at line {ln}: class_id must be int")
                    if cid < 0:
                        name = Path(str(getattr(f, "filename", "") or "label.txt")).name
                        raise ValidationError(f"Invalid label format in {name} at line {ln}: class_id must be >= 0")
                    if max_id is None or cid > max_id:
                        max_id = cid
            finally:
                try:
                    src.seek(0)
                except Exception:
                    pass
        return max_id

    def _normalize_yolo_names(self, names_obj, nc_obj=None) -> list[str]:
        if isinstance(names_obj, list):
            return [str(x) for x in names_obj]
        if isinstance(names_obj, dict):
            items: list[tuple[int, str]] = []
            for k, v in names_obj.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                items.append((idx, str(v)))
            if not items:
                return []
            max_idx = max(i for i, _ in items)
            names = [f"class_{i}" for i in range(max_idx + 1)]
            for i, v in items:
                if 0 <= i < len(names):
                    names[i] = v
            return names
        try:
            nc = int(nc_obj) if nc_obj is not None else 0
        except Exception:
            nc = 0
        return [f"class_{i}" for i in range(max(0, nc))]

    def _unique_path(self, dest_dir: Path, filename: str) -> Path:
        """
        Avoid overwriting existing files by suffixing a short random token.
        """
        safe_name = Path(str(filename or "")).name.strip() or f"{uuid.uuid4().hex}.jpg"
        base = Path(safe_name).stem
        ext = Path(safe_name).suffix or ".jpg"

        out = dest_dir / f"{base}{ext}"
        if not out.exists():
            return out

        for _ in range(50):
            token = uuid.uuid4().hex[:8]
            cand = dest_dir / f"{base}_{token}{ext}"
            if not cand.exists():
                return cand

        # Fallback
        return dest_dir / f"{uuid.uuid4().hex}{ext}"

    # --------------------
    # dataset versions
    # --------------------
    def list_versions(self, db: Session, dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[DatasetVersion]:
        self.get_dataset(db, dataset_id)
        return self.versions.list_by_dataset(db, int(dataset_id), skip=skip, limit=limit)

    def get_detail(self, db: Session, dataset_id: int, *, versions_limit: int = 20, events_limit: int = 20) -> dict:
        ds = self.get_dataset(db, dataset_id)

        active_ver = None
        if ds.active_version_id is not None:
            active_ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )

        stats = None
        try:
            stats = self.get_statistics(db, int(ds.dataset_id))
        except Exception:
            stats = None

        versions = []
        if versions_limit and versions_limit > 0:
            versions = self.versions.list_by_dataset(db, int(ds.dataset_id), skip=0, limit=int(versions_limit))

        events = []
        if events_limit and events_limit > 0:
            events = self.events.list_by_dataset(db, int(ds.dataset_id), skip=0, limit=int(events_limit))

        return {
            "dataset": ds,
            "statistics": stats,
            "active_version": active_ver,
            "versions": versions,
            "events": events,
        }

    def get_view(
        self,
        db: Session,
        dataset_id: int,
        *,
        version_id: int | None = None,
        class_id: int | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        """
        Get dataset view with category statistics and paginated image list.
        """
        import math

        ds = self.get_dataset(db, dataset_id)
        ver = self._resolve_dataset_version(db, ds, version_id)
        root_token = str(ver.snapshot_path or ds.storage_path)
        dataset_path = resolve_dataset_path(root_token).resolve(strict=False)
        url_prefix = self._dataset_static_prefix(root_token)
        class_names = self._get_version_class_names(ver)
        image_rels = self._collect_manifest_image_rels(ver.manifest_path) if ver.manifest_path else []
        index_payload = self._load_view_index(ver.manifest_path) if ver.manifest_path else None

        if index_payload is None and ver.manifest_path:
            # Filtering by class requires the complete index; build it on demand.
            if class_id is not None and ds.dataset_type == DatasetType.DETECTION:
                index_payload = self._build_and_store_view_index(ds, ver, dataset_root=dataset_path)
            else:
                self._trigger_view_index_generation(ds, ver, dataset_root=dataset_path)

        if index_payload is not None:
            items_payload = list(index_payload.get("images") or [])
            if class_id is not None:
                items_payload = [
                    item for item in items_payload
                    if int(class_id) in {int(cid) for cid in list(item.get("classes") or [])}
                ]

            total_items = len(items_payload)
            total_pages = max(1, math.ceil(total_items / page_size))
            page = max(1, min(page, total_pages))
            start = (page - 1) * page_size
            end = start + page_size
            page_payload = items_payload[start:end]

            categories = []
            class_image_count = dict(index_payload.get("class_image_count") or {})
            for raw_cid, raw_count in sorted(class_image_count.items(), key=lambda kv: int(kv[0])):
                cid = int(raw_cid)
                name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                categories.append({"class_id": cid, "name": name, "count": int(raw_count or 0)})

            items = [
                self._make_view_item(
                    dataset_id=int(ds.dataset_id),
                    version=ver,
                    url_prefix=url_prefix,
                    rel=str(item.get("rel") or ""),
                    index=int(start + idx + 1),
                    object_count=int(item.get("object_count") or 0),
                    classes=[int(cid) for cid in list(item.get("classes") or [])],
                )
                for idx, item in enumerate(page_payload)
                if str(item.get("rel") or "").strip()
            ]
            view_index_status = "ready"
        else:
            total_items = len(image_rels)
            total_pages = max(1, math.ceil(total_items / page_size))
            page = max(1, min(page, total_pages))
            start = (page - 1) * page_size
            end = start + page_size
            page_rels = image_rels[start:end]
            fallback_stats = (
                self._load_detection_stats_for_image_rels(
                    dataset_path=dataset_path,
                    image_rels=page_rels,
                    max_workers=max(1, settings.view_index_max_workers),
                )
                if ds.dataset_type == DatasetType.DETECTION
                else {}
            )
            items = []
            for idx, rel in enumerate(page_rels):
                stat = fallback_stats.get(str(rel), {})
                items.append(
                    self._make_view_item(
                        dataset_id=int(ds.dataset_id),
                        version=ver,
                        url_prefix=url_prefix,
                        rel=rel,
                        index=int(start + idx + 1),
                        object_count=int(stat.get("object_count") or 0),
                        classes=[int(cid) for cid in list(stat.get("classes") or [])],
                    )
                )
            categories = []
            view_index_status = self._get_artifact_job_summary(
                self._artifact_job_key("view_index", int(ds.dataset_id), int(ver.version_id))
            ).get("state", "building")

        thumb_job = self._get_artifact_job_summary(
            self._artifact_job_key(
                "thumbnails",
                int(ds.dataset_id),
                int(ver.version_id) if ver.snapshot_path else None,
            )
        )

        return {
            "dataset_id": ds.dataset_id,
            "version_id": ver.version_id,
            "categories": categories,
            "items": items,
            "meta": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "thumbnail_status": thumb_job.get("state", "ready"),
                "thumbnail_progress": thumb_job.get("progress"),
                "view_index_status": view_index_status,
            },
        }

    def get_image_annotations(
        self,
        db: Session,
        dataset_id: int,
        *,
        image_path: str,
        version_id: int | None = None,
    ) -> dict:
        ds = self.get_dataset(db, dataset_id)
        ver = self._resolve_dataset_version(db, ds, version_id)

        root_token = str(ver.snapshot_path or ds.storage_path)
        dataset_path = resolve_dataset_path(root_token).resolve(strict=False)
        dataset_root = dataset_path.resolve(strict=False)

        image_rel = Path(str(image_path or "").strip().replace("\\", "/").lstrip("/"))
        if not image_rel.as_posix() or image_rel.is_absolute() or ".." in image_rel.parts:
            raise ValidationError("image_path must be a safe relative path")

        image_abs = (dataset_root / image_rel).resolve(strict=False)
        if dataset_root not in image_abs.parents and image_abs != dataset_root:
            raise ValidationError("Unsafe image_path")
        if not image_abs.exists() or not image_abs.is_file():
            raise NotFoundError("Image file not found")

        width, height = self._read_image_size(image_abs)
        class_names = self._get_version_class_names(ver)
        label_rel = self._guess_label_rel_path_from_image(image_rel.as_posix())
        label_abs = (dataset_root / label_rel).resolve(strict=False)

        boxes = []
        object_count = 0
        if dataset_root in label_abs.parents and label_abs.exists() and label_abs.is_file():
            object_count = self._count_yolo_objects(label_abs)
            boxes = self._parse_yolo_boxes(label_abs, width=width, height=height, class_names=class_names)

        url_prefix = self._dataset_static_prefix(root_token)
        image_url = f"{url_prefix}/{image_rel.as_posix()}" if url_prefix else image_rel.as_posix()

        return {
            "dataset_id": int(ds.dataset_id),
            "version_id": int(ver.version_id),
            "image_path": image_rel.as_posix(),
            "image_url": image_url,
            "width": width,
            "height": height,
            "object_count": int(object_count),
            "boxes": boxes,
        }

    def create_version(
        self,
        db: Session,
        dataset_id: int,
        *,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        create_snapshot: bool = False,
    ) -> DatasetVersion:
        ds = self.get_dataset(db, dataset_id)
        row = self._create_version_row(
            db,
            ds,
            message=message,
            created_by=created_by,
            create_snapshot=create_snapshot,
        )
        db.commit()
        db.refresh(row)
        return row

    def create_version_from_directory(
        self,
        db: Session,
        dataset_id: int,
        *,
        source_dir: Path | str,
        message: Optional[str] = None,
        created_by: Optional[str] = None,
        activate: bool = False,
    ) -> tuple[Dataset, DatasetVersion]:
        """
        Create a new dataset version from an arbitrary source directory without
        touching the dataset's main storage directory.

        The source content is copied into `.versions/<token>/vN/snapshot/` and
        that snapshot becomes the new version's `snapshot_path`.
        """
        ds = self.get_dataset(db, dataset_id)
        src = Path(source_dir).expanduser().resolve(strict=False)
        if not src.exists() or not src.is_dir():
            raise ValidationError("source_dir does not exist")

        latest = self.versions.get_latest(db, ds.dataset_id)
        next_version = int(getattr(latest, "version", 0) or 0) + 1
        parent_version_id = getattr(latest, "version_id", None) if latest else None

        manifest_rel, file_count, size_bytes, stats = self._write_manifest(
            dataset_token=_versions_token(ds),
            dataset_id=int(ds.dataset_id),
            dataset_name=str(ds.name),
            dataset_path=src,
            version=next_version,
            dataset_type=ds.dataset_type,
        )
        snapshot_rel = self._create_snapshot(dataset_token=_versions_token(ds), dataset_path=src, version=next_version)

        meta: dict | None = None
        try:
            meta_obj: dict = {}
            if isinstance(stats, dict) and stats:
                meta_obj["stats"] = stats
            if ds.dataset_type == DatasetType.DETECTION:
                yolo_yaml = find_yolo_dataset_yaml(src, dataset_name=str(getattr(ds, "name", "") or "") or None)
                if yolo_yaml is not None and yolo_yaml.exists() and yolo_yaml.is_file():
                    try:
                        cfg = yaml.safe_load(yolo_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
                    except Exception:
                        cfg = {}
                    if isinstance(cfg, dict):
                        names = self._normalize_yolo_names(cfg.get("names"), cfg.get("nc"))
                        if names:
                            meta_obj["yolo"] = {"nc": int(len(names)), "names": names}
            meta = meta_obj or None
        except Exception:
            meta = {"stats": stats} if isinstance(stats, dict) and stats else None

        row = DatasetVersion(
            dataset_id=ds.dataset_id,
            version=next_version,
            parent_version_id=parent_version_id,
            status=DatasetVersionStatus.FINALIZED,
            message=message,
            manifest_path=manifest_rel,
            snapshot_path=snapshot_rel,
            file_count=file_count,
            size_bytes=size_bytes,
            created_by=created_by,
            meta=meta,
        )
        db.add(row)
        db.flush()

        if activate or ds.active_version_id is None:
            ds.active_version_id = row.version_id

        self._ensure_images_indexed(db, ds, row)
        self._schedule_version_artifacts(ds, row)

        db.commit()
        db.refresh(ds)
        db.refresh(row)
        return ds, row

    def _create_version_row(
        self,
        db: Session,
        ds: Dataset,
        *,
        message: Optional[str],
        created_by: Optional[str],
        create_snapshot: bool,
    ) -> DatasetVersion:
        latest = self.versions.get_latest(db, ds.dataset_id)
        next_version = int(getattr(latest, "version", 0) or 0) + 1
        parent_version_id = getattr(latest, "version_id", None) if latest else None

        dataset_path = resolve_dataset_path(ds.storage_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise ValidationError(f"Dataset path does not exist: {dataset_path}")

        manifest_rel, file_count, size_bytes, stats = self._write_manifest(
            dataset_token=_versions_token(ds),
            dataset_id=int(ds.dataset_id),
            dataset_name=str(ds.name),
            dataset_path=dataset_path,
            version=next_version,
            dataset_type=ds.dataset_type,
        )

        snapshot_rel = None
        if create_snapshot:
            snapshot_rel = self._create_snapshot(dataset_token=_versions_token(ds), dataset_path=dataset_path, version=next_version)

        meta: dict | None = None
        try:
            meta_obj: dict = {}
            if isinstance(stats, dict) and stats:
                meta_obj["stats"] = stats
            if ds.dataset_type == DatasetType.DETECTION:
                yolo_yaml = find_yolo_dataset_yaml(dataset_path, dataset_name=str(getattr(ds, "name", "") or "") or None)
                if yolo_yaml is not None and yolo_yaml.exists() and yolo_yaml.is_file():
                    try:
                        cfg = yaml.safe_load(yolo_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
                    except Exception:
                        cfg = {}
                    if isinstance(cfg, dict):
                        names = self._normalize_yolo_names(cfg.get("names"), cfg.get("nc"))
                        if names:
                            meta_obj["yolo"] = {"nc": int(len(names)), "names": names}
            meta = meta_obj or None
        except Exception:
            meta = {"stats": stats} if isinstance(stats, dict) and stats else None

        row = DatasetVersion(
            dataset_id=ds.dataset_id,
            version=next_version,
            parent_version_id=parent_version_id,
            status=DatasetVersionStatus.FINALIZED,
            message=message,
            manifest_path=manifest_rel,
            snapshot_path=snapshot_rel,
            file_count=file_count,
            size_bytes=size_bytes,
            created_by=created_by,
            meta=meta,
        )
        db.add(row)
        db.flush()

        # If dataset has no active version yet, set it.
        if ds.active_version_id is None:
            ds.active_version_id = row.version_id

        # Index images for this version so every image has a stable ID.
        self._ensure_images_indexed(db, ds, row)
        self._schedule_version_artifacts(ds, row)

        return row

    def _create_illegal_version_row(
        self,
        db: Session,
        ds: Dataset,
        *,
        message: Optional[str],
        created_by: Optional[str],
        illegal_reason: str,
    ) -> DatasetVersion:
        latest = self.versions.get_latest(db, ds.dataset_id)
        next_version = int(getattr(latest, "version", 0) or 0) + 1
        parent_version_id = getattr(latest, "version_id", None) if latest else None

        dataset_path = resolve_dataset_path(ds.storage_path)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise ValidationError(f"Dataset path does not exist: {dataset_path}")

        manifest_rel, file_count, size_bytes, stats = self._write_manifest(
            dataset_token=_versions_token(ds),
            dataset_id=int(ds.dataset_id),
            dataset_name=str(ds.name),
            dataset_path=dataset_path,
            version=next_version,
            dataset_type=ds.dataset_type,
        )

        reason = str(illegal_reason or "non_yolo_json")
        meta: dict = {
            "stats": stats,
            "illegal": True,
            "illegal_reason": reason,
            "conversion": {"status": "pending", "supported": reason == "labelme_json"},
        }

        row = DatasetVersion(
            dataset_id=ds.dataset_id,
            version=next_version,
            parent_version_id=parent_version_id,
            status=DatasetVersionStatus.FAILED,
            message=message,
            manifest_path=manifest_rel,
            snapshot_path=None,
            file_count=file_count,
            size_bytes=size_bytes,
            created_by=created_by,
            meta=meta,
        )
        db.add(row)
        db.flush()

        ds.active_version_id = row.version_id

        # Index images for preview
        self._ensure_images_indexed(db, ds, row)
        self._schedule_version_artifacts(ds, row)

        return row

    def activate_version(self, db: Session, dataset_id: int, version_id: int) -> Dataset:
        ds = self.get_dataset(db, dataset_id)
        ver = db.query(DatasetVersion).filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == ds.dataset_id).first()
        if not ver:
            raise NotFoundError("Dataset version not found")
        ds.active_version_id = ver.version_id
        db.commit()
        db.refresh(ds)
        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(ds.dataset_id),
                    version_id=int(ver.version_id),
                    event_type="activate_version",
                    message=f"Activated version v{int(ver.version)}",
                )
            )
            db.commit()
        except Exception:
            db.rollback()
        return ds

    def get_statistics(self, db: Session, dataset_id: int, *, version_id: int | None = None) -> dict:
        ds = self.get_dataset(db, dataset_id)

        ver: DatasetVersion | None = None
        if version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )
        else:
            if ds.active_version_id is not None:
                ver = (
                    db.query(DatasetVersion)
                    .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                    .first()
                )

        if not ver:
            raise NotFoundError("Dataset version not found (no active version?)")

        stats: dict | None = None
        if isinstance(ver.meta, dict):
            stats = ver.meta.get("stats")
        if not isinstance(stats, dict):
            stats = self._compute_stats_from_manifest(ver.manifest_path)
            # Persist best-effort stats for later calls.
            try:
                ver.meta = dict(ver.meta or {})
                ver.meta["stats"] = stats
                db.commit()
            except Exception:
                db.rollback()

        size_bytes = int(ver.size_bytes or 0)
        size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0

        return {
            "dataset_id": int(ds.dataset_id),
            "version_id": int(ver.version_id),
            "version": int(ver.version),
            "total_files": int(ver.file_count or 0),
            "total_size_bytes": size_bytes,
            "total_size_mb": size_mb,
            "total_images": int(stats.get("total_images") or 0),
            "annotations_count": stats.get("annotations_count"),
        }

    def diff_versions(
        self,
        db: Session,
        dataset_id: int,
        version_id: int,
        *,
        base_version_id: int | None = None,
        limit: int = 200,
    ) -> dict:
        """
        Diff two dataset versions using the stored NDJSON manifests.

        If `base_version_id` is omitted, uses version.parent_version_id.
        """
        ds = self.get_dataset(db, dataset_id)
        limit = max(1, min(int(limit), 2000))

        ver = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
            .first()
        )
        if not ver:
            raise NotFoundError("Dataset version not found")
        if not ver.manifest_path:
            raise ValidationError("Dataset version has no manifest_path")

        base_id = int(base_version_id) if base_version_id is not None else ver.parent_version_id
        if not base_id:
            raise ValidationError("base_version_id is required for the first version (no parent)")

        base_ver = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.version_id == int(base_id), DatasetVersion.dataset_id == int(ds.dataset_id))
            .first()
        )
        if not base_ver:
            raise NotFoundError("Base dataset version not found")
        if not base_ver.manifest_path:
            raise ValidationError("Base dataset version has no manifest_path")

        a = self._load_manifest_map(base_ver.manifest_path)
        b = self._load_manifest_map(ver.manifest_path)

        added = [p for p in b.keys() if p not in a]
        removed = [p for p in a.keys() if p not in b]

        modified: list[str] = []
        for p, cur in b.items():
            prev = a.get(p)
            if not prev:
                continue
            if cur.get("size_bytes") != prev.get("size_bytes") or cur.get("mtime") != prev.get("mtime"):
                modified.append(p)

        def _clip(xs: list[str]) -> list[str]:
            return xs[:limit]

        return {
            "dataset_id": int(ds.dataset_id),
            "base_version_id": int(base_ver.version_id),
            "base_version": int(base_ver.version),
            "version_id": int(ver.version_id),
            "version": int(ver.version),
            "summary": {
                "added": len(added),
                "removed": len(removed),
                "modified": len(modified),
            },
            "added": _clip(sorted(added)),
            "removed": _clip(sorted(removed)),
            "modified": _clip(sorted(modified)),
            "truncated": {
                "added": len(added) > limit,
                "removed": len(removed) > limit,
                "modified": len(modified) > limit,
            },
        }

    def _write_manifest(
        self,
        *,
        dataset_token: str,
        dataset_id: int,
        dataset_name: str,
        dataset_path: Path,
        version: int,
        dataset_type: DatasetType,
    ) -> tuple[str, int, int, dict]:
        """
        Stores a portable NDJSON manifest under:
        BASE_DATASETS_DIR/.versions/<dataset_token>/v<version>/manifest.ndjson
        """
        base = settings.datasets_dir.resolve()
        out_dir = base / ".versions" / str(dataset_token) / f"v{int(version)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manifest.ndjson"

        file_count = 0
        size_bytes = 0
        image_count = 0
        annotations_count = 0

        with out_path.open("w", encoding="utf-8") as f:
            header = {
                "dataset_id": int(dataset_id),
                "dataset_name": str(dataset_name),
                "dataset_token": str(dataset_token),
                "version": int(version),
                "dataset_type": str(getattr(dataset_type, "value", dataset_type)),
                "generated_at": _utcnow().isoformat(),
                "format": "ndjson",
            }
            f.write(json.dumps({"_header": header}, ensure_ascii=False) + "\n")

            image_exts = IMAGE_EXTS
            for root, _, files in os.walk(dataset_path):
                for fn in files:
                    p = Path(root) / fn
                    try:
                        st = p.stat()
                    except Exception:
                        continue
                    try:
                        rel = p.relative_to(dataset_path).as_posix()
                    except Exception:
                        rel = p.name

                    file_count += 1
                    size_bytes += int(getattr(st, "st_size", 0) or 0)
                    if p.suffix.lower() in image_exts:
                        image_count += 1

                    # Best-effort label line count (YOLO detection labels).
                    # We only count lines under a `labels/` directory to avoid random .txt files.
                    if dataset_type == DatasetType.DETECTION and p.suffix.lower() == ".txt":
                        try:
                            parts = Path(rel).parts
                            if "labels" in parts:
                                with p.open("r", encoding="utf-8", errors="ignore") as lf:
                                    for line in lf:
                                        if line.strip():
                                            annotations_count += 1
                        except Exception:
                            pass

                    f.write(
                        json.dumps(
                            {"path": rel, "size_bytes": int(st.st_size), "mtime": float(st.st_mtime)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        rel_manifest = out_path.relative_to(base).as_posix()
        stats = {
            "total_files": int(file_count),
            "total_size_bytes": int(size_bytes),
            "total_images": int(image_count),
            "annotations_count": int(annotations_count) if dataset_type == DatasetType.DETECTION else None,
        }
        return rel_manifest, file_count, size_bytes, stats

    def _create_snapshot(self, *, dataset_token: str, dataset_path: Path, version: int) -> str:
        """
        Creates a filesystem snapshot copy (heavy for large datasets).

        Stored under:
        BASE_DATASETS_DIR/.versions/<dataset_token>/v<version>/snapshot/
        """
        base = settings.datasets_dir.resolve()
        out_dir = base / ".versions" / str(dataset_token) / f"v{int(version)}" / "snapshot"
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        shutil.copytree(dataset_path, out_dir)
        return out_dir.relative_to(base).as_posix()

    def _compute_stats_from_manifest(self, manifest_rel: str | None) -> dict:
        if not manifest_rel:
            return {"total_images": 0, "annotations_count": None}

        image_exts = IMAGE_EXTS
        images = 0

        base = settings.datasets_dir.resolve()
        path = (base / str(manifest_rel)).resolve(strict=False)
        try:
            if base not in path.parents and path != base:
                return {"total_images": 0, "annotations_count": None}
            if not path.exists() or not path.is_file():
                return {"total_images": 0, "annotations_count": None}
        except Exception:
            return {"total_images": 0, "annotations_count": None}

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('{"_header"'):
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rel = str(obj.get("path") or "")
                    if Path(rel).suffix.lower() in image_exts:
                        images += 1
        except Exception:
            pass

        return {"total_images": int(images), "annotations_count": None}

    def _load_manifest_map(self, manifest_rel: str) -> dict:
        """
        Load NDJSON manifest into a mapping: rel_path -> {size_bytes, mtime}.
        """
        base = settings.datasets_dir.resolve()
        path = (base / str(manifest_rel)).resolve(strict=False)
        if base not in path.parents and path != base:
            raise ValidationError("Unsafe manifest path")
        if not path.exists() or not path.is_file():
            raise NotFoundError("Manifest file not found")

        out: dict = {}
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{"_header"'):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rel = str(obj.get("path") or "")
                if not rel:
                    continue
                out[rel] = {"size_bytes": obj.get("size_bytes"), "mtime": obj.get("mtime")}
        return out

    def list_files(
        self,
        db: Session,
        dataset_id: int,
        *,
        version_id: int | None = None,
        kind: str = "image",
        prefix: str | None = None,
        q: str | None = None,
        skip: int = 0,
        limit: int = 100,
        include_missing: bool = False,
    ) -> tuple[list[dict], int]:
        """
        List files by dataset version using the stored manifest (no per-file DB rows).

        Returns (items, total) where total is the total number of matched files.
        """
        ds = self.get_dataset(db, dataset_id)

        ver: DatasetVersion | None = None
        if version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )
        else:
            if ds.active_version_id is not None:
                ver = (
                    db.query(DatasetVersion)
                    .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                    .first()
                )

        if not ver:
            raise NotFoundError("Dataset version not found (no active version?)")
        if not ver.manifest_path:
            raise ValidationError("Dataset version has no manifest_path")

        # Prefer snapshot for reproducible "old version" browsing if present.
        root_token = str(ver.snapshot_path or ds.storage_path)
        dataset_root = resolve_dataset_path(root_token).resolve(strict=False)

        kind = (kind or "image").strip().lower()
        if kind not in ("image", "label", "all"):
            raise ValidationError("kind must be one of: image, label, all")

        prefix_norm = (prefix or "").strip().replace("\\", "/").lstrip("/")
        q_norm = (q or "").strip().lower()

        image_exts = IMAGE_EXTS

        def _match(rel: str) -> bool:
            if prefix_norm and not rel.startswith(prefix_norm):
                return False
            if q_norm and q_norm not in rel.lower():
                return False
            if kind == "all":
                return True
            ext = Path(rel).suffix.lower()
            if kind == "image":
                return ext in image_exts
            # label
            if ext != ".txt":
                return False
            parts = Path(rel).parts
            return "labels" in parts

        items: list[dict] = []
        total = 0

        start = max(0, int(skip))
        end = start + max(0, int(limit))

        url_prefix = self._dataset_static_prefix(root_token)

        for rel, size_bytes, mtime in self._iter_manifest_entries(ver.manifest_path):
            rel = str(rel or "").strip().replace("\\", "/")
            if not rel:
                continue
            if not _match(rel):
                continue

            abs_path = (dataset_root / rel).resolve(strict=False)
            exists = False
            try:
                exists = abs_path.exists() and abs_path.is_file()
            except Exception:
                exists = False

            if not include_missing and not exists:
                # If old versions are manifest-only and files were deleted later, skip by default.
                continue

            idx = total
            total += 1

            if idx < start or idx >= end:
                continue

            url = None
            if url_prefix:
                url = f"{url_prefix}/{rel}"

            items.append(
                {
                    "path": rel,
                    "size_bytes": int(size_bytes or 0),
                    "mtime": float(mtime or 0.0),
                    "url": url,
                    "exists": bool(exists),
                }
            )

        return items, int(total)

    # --------------------
    # dataset images / splits
    # --------------------
    def _artifact_job_key(self, kind: str, dataset_id: int, version_id: int | None = None) -> str:
        suffix = f":v{int(version_id)}" if version_id is not None else ":live"
        return f"{str(kind).strip().lower()}:{int(dataset_id)}{suffix}"

    def _set_artifact_job_summary(self, key: str, **fields: Any) -> dict[str, Any]:
        with self._artifact_jobs_lock:
            current = dict(self._artifact_jobs.get(key) or {})
            current.update(fields)
            current["updated_at"] = _utcnow().isoformat()
            self._artifact_jobs[key] = current
            return dict(current)

    def _get_artifact_job_summary(self, key: str) -> dict[str, Any]:
        with self._artifact_jobs_lock:
            return dict(self._artifact_jobs.get(key) or {})

    def _thumbnail_cache_prefix_for_version(self, ver: DatasetVersion) -> str | None:
        if getattr(ver, "snapshot_path", None):
            return f"v{int(ver.version_id)}"
        return None

    def _thumbnail_static_url(self, *, dataset_id: int, version: DatasetVersion, rel: str) -> str:
        rel_webp = Path(str(rel or "")).with_suffix(".webp").as_posix().lstrip("/")
        cache_prefix = self._thumbnail_cache_prefix_for_version(version)
        if cache_prefix:
            return f"/static/thumbnails/{int(dataset_id)}/{cache_prefix}/{rel_webp}"
        return f"/static/thumbnails/{int(dataset_id)}/{rel_webp}"

    def _make_view_item(
        self,
        *,
        dataset_id: int,
        version: DatasetVersion,
        url_prefix: str | None,
        rel: str,
        index: int,
        object_count: int,
        classes: list[int],
    ) -> dict[str, Any]:
        rel_clean = str(rel or "").strip().replace("\\", "/").lstrip("/")
        url = f"{url_prefix}/{rel_clean}" if url_prefix else rel_clean
        return {
            "id": int(index),
            "name": rel_clean,
            "url": url,
            "thumbnail_url": self._thumbnail_static_url(dataset_id=int(dataset_id), version=version, rel=rel_clean),
            "width": None,
            "height": None,
            "object_count": int(object_count or 0),
            "classes": sorted({int(cid) for cid in list(classes or []) if int(cid) >= 0}),
        }

    def _view_index_path(self, manifest_rel: str | None) -> Path | None:
        if not manifest_rel:
            return None
        base = settings.datasets_dir.resolve()
        manifest_path = (base / str(manifest_rel)).resolve(strict=False)
        if base not in manifest_path.parents and manifest_path != base:
            raise ValidationError("Unsafe manifest path")
        return manifest_path.with_name("view_index.json")

    def _collect_manifest_image_rels(self, manifest_rel: str | None, *, limit: int | None = None) -> list[str]:
        image_rels: list[str] = []
        if not manifest_rel:
            return image_rels
        max_items = int(limit) if limit is not None else None
        for rel, _size_bytes, _mtime in self._iter_manifest_entries(manifest_rel):
            rel_str = str(rel or "").strip().replace("\\", "/")
            if not rel_str:
                continue
            if Path(rel_str).suffix.lower() not in IMAGE_EXTS:
                continue
            image_rels.append(rel_str)
            if max_items is not None and len(image_rels) >= max_items:
                break
        return image_rels

    def _read_json_file(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or not path.is_file():
                return None
            raw = path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw or "{}")
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _load_view_index(self, manifest_rel: str | None) -> dict[str, Any] | None:
        path = self._view_index_path(manifest_rel)
        if path is None:
            return None
        data = self._read_json_file(path)
        if not isinstance(data, dict):
            return None
        images_raw = list(data.get("images") or [])
        class_image_count_raw = data.get("class_image_count") or {}
        images: list[dict[str, Any]] = []
        for item in images_raw:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("rel") or "").strip().replace("\\", "/")
            if not rel:
                continue
            images.append(
                {
                    "rel": rel,
                    "key": self._canonical_image_key_from_rel(rel),
                    "object_count": int(item.get("object_count") or 0),
                    "classes": sorted({int(cid) for cid in list(item.get("classes") or []) if int(cid) >= 0}),
                }
            )
        class_image_count: dict[str, int] = {}
        if isinstance(class_image_count_raw, dict):
            for raw_k, raw_v in class_image_count_raw.items():
                try:
                    class_image_count[str(int(raw_k))] = int(raw_v or 0)
                except Exception:
                    continue
        return {
            "generated_at": data.get("generated_at"),
            "images": images,
            "class_image_count": class_image_count,
        }

    def _write_view_index(self, manifest_rel: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._view_index_path(manifest_rel)
        if path is None:
            return payload
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
        return payload

    def _summarize_yolo_label(self, label_path: Path) -> tuple[list[int], int]:
        if not label_path.exists() or not label_path.is_file():
            return [], 0
        class_ids: set[int] = set()
        object_count = 0
        try:
            with label_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                    except Exception:
                        continue
                    if cid < 0:
                        continue
                    class_ids.add(int(cid))
                    object_count += 1
        except Exception:
            return [], 0
        return sorted(class_ids), int(object_count)

    def _load_detection_stats_for_image_rels(
        self,
        *,
        dataset_path: Path,
        image_rels: list[str],
        max_workers: int,
    ) -> dict[str, dict[str, Any]]:
        if not image_rels:
            return {}

        dataset_root = Path(dataset_path).resolve(strict=False)
        results: dict[str, dict[str, Any]] = {}

        def _one(rel: str) -> tuple[str, list[int], int]:
            rel_str = str(rel or "").strip().replace("\\", "/").lstrip("/")
            if not rel_str:
                return "", [], 0
            label_rel = self._guess_label_rel_path_from_image(rel_str)
            label_abs = (dataset_root / label_rel).resolve(strict=False)
            if dataset_root not in label_abs.parents and label_abs != dataset_root:
                return rel_str, [], 0
            classes, object_count = self._summarize_yolo_label(label_abs)
            return rel_str, classes, object_count

        worker_count = max(1, min(int(max_workers or 1), max(1, len(image_rels))))
        if worker_count <= 1 or len(image_rels) <= 1:
            for rel in image_rels:
                rel_str, classes, object_count = _one(rel)
                if rel_str:
                    results[rel_str] = {"classes": classes, "object_count": int(object_count)}
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_one, rel) for rel in image_rels]
            for future in as_completed(futures):
                try:
                    rel_str, classes, object_count = future.result()
                except Exception:
                    continue
                if rel_str:
                    results[rel_str] = {"classes": classes, "object_count": int(object_count)}
        return results

    def _build_view_index_payload(
        self,
        *,
        dataset_path: Path,
        image_rels: list[str],
        dataset_type: DatasetType,
    ) -> dict[str, Any]:
        images: list[dict[str, Any]] = []
        class_image_count: dict[str, int] = {}

        detection_stats = (
            self._load_detection_stats_for_image_rels(
                dataset_path=dataset_path,
                image_rels=image_rels,
                max_workers=max(1, settings.view_index_max_workers),
            )
            if dataset_type == DatasetType.DETECTION
            else {}
        )

        for rel in image_rels:
            rel_str = str(rel or "").strip().replace("\\", "/")
            if not rel_str:
                continue
            stat = detection_stats.get(rel_str, {})
            classes = sorted({int(cid) for cid in list(stat.get("classes") or []) if int(cid) >= 0})
            object_count = int(stat.get("object_count") or 0)
            images.append(
                {
                    "rel": rel_str,
                    "key": self._canonical_image_key_from_rel(rel_str),
                    "object_count": object_count,
                    "classes": classes,
                }
            )
            for cid in classes:
                key = str(int(cid))
                class_image_count[key] = int(class_image_count.get(key, 0) + 1)

        return {
            "generated_at": _utcnow().isoformat(),
            "images": images,
            "class_image_count": class_image_count,
        }

    def _build_and_store_view_index(
        self,
        ds: Dataset,
        ver: DatasetVersion,
        *,
        dataset_root: Path | None = None,
    ) -> dict[str, Any]:
        if not ver.manifest_path:
            return {"generated_at": _utcnow().isoformat(), "images": [], "class_image_count": {}}
        dataset_path = Path(dataset_root or resolve_dataset_path(ver.snapshot_path or ds.storage_path)).resolve(strict=False)
        image_rels = self._collect_manifest_image_rels(ver.manifest_path)
        payload = self._build_view_index_payload(
            dataset_path=dataset_path,
            image_rels=image_rels,
            dataset_type=ds.dataset_type,
        )
        self._write_view_index(ver.manifest_path, payload)
        return payload

    def _trigger_view_index_generation(
        self,
        ds: Dataset,
        ver: DatasetVersion,
        *,
        dataset_root: Path | None = None,
    ) -> None:
        if not ver.manifest_path:
            return
        if self._load_view_index(ver.manifest_path) is not None:
            self._set_artifact_job_summary(
                self._artifact_job_key("view_index", int(ds.dataset_id), int(ver.version_id)),
                state="ready",
                progress=100,
            )
            return

        job_key = self._artifact_job_key("view_index", int(ds.dataset_id), int(ver.version_id))
        current = self._get_artifact_job_summary(job_key)
        if str(current.get("state") or "").lower() in {"queued", "running"}:
            return

        dataset_path = Path(dataset_root or resolve_dataset_path(ver.snapshot_path or ds.storage_path)).resolve(strict=False)
        self._set_artifact_job_summary(job_key, state="queued", progress=0)

        def _generate():
            self._set_artifact_job_summary(job_key, state="running", progress=1)
            try:
                payload = self._build_and_store_view_index(ds, ver, dataset_root=dataset_path)
                total_images = len(list(payload.get("images") or []))
                self._set_artifact_job_summary(
                    job_key,
                    state="ready",
                    progress=100,
                    total_images=int(total_images),
                )
            except Exception as e:
                self._set_artifact_job_summary(job_key, state="failed", progress=0, error=str(e))
                logger.error(
                    "View index generation failed for dataset %s version %s: %s",
                    getattr(ds, "dataset_id", "?"),
                    getattr(ver, "version_id", "?"),
                    e,
                )

        threading.Thread(target=_generate, daemon=True).start()

    def _schedule_version_artifacts(self, ds: Dataset, ver: DatasetVersion) -> None:
        root_token = str(ver.snapshot_path or ds.storage_path)
        dataset_path = resolve_dataset_path(root_token).resolve(strict=False)

        try:
            prewarm_limit = max(0, int(settings.thumbnail_first_page_prewarm or 0))
        except Exception:
            prewarm_limit = 0

        if prewarm_limit > 0 and ver.manifest_path:
            try:
                priority_rels = self._collect_manifest_image_rels(ver.manifest_path, limit=prewarm_limit)
                if priority_rels:
                    ThumbnailService().pregenerate_for_dataset(
                        dataset_id=int(ds.dataset_id),
                        dataset_root=dataset_path,
                        size=max(16, int(settings.thumbnail_size or 200)),
                        max_workers=max(1, min(int(settings.thumbnail_max_workers or 1), len(priority_rels))),
                        cache_prefix=self._thumbnail_cache_prefix_for_version(ver),
                        rel_paths=priority_rels,
                    )
            except Exception as e:
                logger.warning(
                    "Thumbnail prewarm failed for dataset %s version %s: %s",
                    getattr(ds, "dataset_id", "?"),
                    getattr(ver, "version_id", "?"),
                    e,
                )

        self._trigger_thumbnail_pregeneration(
            dataset_id=int(ds.dataset_id),
            version_id=int(ver.version_id),
            dataset_path=dataset_path,
            cache_prefix=self._thumbnail_cache_prefix_for_version(ver),
        )
        self._trigger_view_index_generation(ds, ver, dataset_root=dataset_path)

    def _get_version_class_names(self, ver: DatasetVersion) -> list[str]:
        class_names: list[str] = []
        if isinstance(ver.meta, dict):
            yolo = ver.meta.get("yolo")
            if isinstance(yolo, dict):
                names = yolo.get("names")
                class_names = self._normalize_yolo_names(names, yolo.get("nc"))
        return class_names

    def _canonical_image_key_from_label_path(self, label_file: Path, dataset_path: Path) -> str:
        try:
            rel = label_file.resolve(strict=False).relative_to((dataset_path / "labels").resolve(strict=False))
        except Exception:
            rel = label_file
        return rel.with_suffix("").as_posix().lstrip("./")

    def _canonical_image_key_from_rel(self, rel_path: str) -> str:
        s = str(rel_path or "").strip().replace("\\", "/")
        if s.startswith("images/"):
            s = s[len("images/") :]
        return Path(s).with_suffix("").as_posix().lstrip("./")

    def _guess_label_rel_path_from_image(self, image_rel: str) -> str:
        rel = Path(str(image_rel or "").replace("\\", "/").lstrip("/"))
        parts = list(rel.parts)
        for idx, part in enumerate(parts):
            if str(part).lower() == "images":
                parts[idx] = "labels"
                return Path(*parts).with_suffix(".txt").as_posix()
        return (Path("labels") / rel).with_suffix(".txt").as_posix()

    def _read_image_size(self, image_path: Path) -> tuple[int | None, int | None]:
        try:
            if Image is not None:
                with Image.open(image_path) as img:
                    w, h = img.size
                    if w and h:
                        return int(w), int(h)
        except Exception:
            pass

        try:
            if rasterio is not None:
                with rasterio.open(str(image_path)) as ds:
                    if ds.width and ds.height:
                        return int(ds.width), int(ds.height)
        except Exception:
            pass

        return None, None

    def _parse_yolo_boxes(
        self,
        label_path: Path,
        *,
        width: int | None,
        height: int | None,
        class_names: list[str],
    ) -> list[dict]:
        if not width or not height:
            return []
        if not label_path.exists() or not label_path.is_file():
            return []

        boxes: list[dict] = []
        try:
            text = label_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        img_w = float(width)
        img_h = float(height)
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w_n = float(parts[3])
                h_n = float(parts[4])
            except Exception:
                continue
            if cid < 0:
                continue

            box_w = max(0.0, w_n * img_w)
            box_h = max(0.0, h_n * img_h)
            x1 = max(0.0, x_c * img_w - box_w / 2.0)
            y1 = max(0.0, y_c * img_h - box_h / 2.0)
            x2 = min(img_w, x1 + box_w)
            y2 = min(img_h, y1 + box_h)
            class_name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

            boxes.append(
                {
                    "class_id": int(cid),
                    "class_name": str(class_name),
                    "x1": round(x1, 3),
                    "y1": round(y1, 3),
                    "x2": round(x2, 3),
                    "y2": round(y2, 3),
                }
            )
        return boxes

    def _count_yolo_objects(self, label_path: Path) -> int:
        _classes, total = self._summarize_yolo_label(label_path)
        return int(total)

    def _resolve_dataset_version(self, db: Session, ds: Dataset, version_id: int | None) -> DatasetVersion:
        if version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
                .first()
            )
            if not ver:
                raise NotFoundError("Dataset version not found")
            return ver

        if ds.active_version_id is None:
            raise NotFoundError("Dataset has no active version")

        ver = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(ds.dataset_id))
            .first()
        )
        if not ver:
            raise NotFoundError("Dataset version not found")
        return ver

    def _ensure_images_indexed(self, db: Session, ds: Dataset, ver: DatasetVersion) -> int:
        if not ver.manifest_path:
            raise ValidationError("Dataset version has no manifest_path")

        # Skip expensive work if we've already indexed this version.
        existing_any = (
            db.query(DatasetImage.image_id)
            .filter(DatasetImage.dataset_id == int(ds.dataset_id), DatasetImage.dataset_version_id == int(ver.version_id))
            .limit(1)
            .first()
        )
        existing_paths: set[str] = set()
        if existing_any:
            existing_paths = {
                str(r[0])
                for r in db.query(DatasetImage.path)
                .filter(DatasetImage.dataset_id == int(ds.dataset_id), DatasetImage.dataset_version_id == int(ver.version_id))
                .all()
            }

        image_exts = IMAGE_EXTS
        inserted = 0
        batch: list[DatasetImage] = []

        for rel, _size_bytes, _mtime in self._iter_manifest_entries(ver.manifest_path):
            rel_str = str(rel or "").strip().replace("\\", "/")
            if not rel_str:
                continue
            if Path(rel_str).suffix.lower() not in image_exts:
                continue
            if existing_paths and rel_str in existing_paths:
                continue
            batch.append(
                DatasetImage(
                    dataset_id=int(ds.dataset_id),
                    dataset_version_id=int(ver.version_id),
                    path=rel_str,
                    split=None,
                )
            )
            if len(batch) >= 1000:
                db.add_all(batch)
                db.flush()
                inserted += len(batch)
                batch = []

        if batch:
            db.add_all(batch)
            db.flush()
            inserted += len(batch)

        return int(inserted)

    def _normalize_split_ratios(
        self,
        train_ratio: float,
        val_ratio: float | None,
        test_ratio: float | None,
    ) -> tuple[float, float, float]:
        try:
            tr = float(train_ratio)
        except Exception:
            raise ValidationError("train_ratio must be a number")
        if tr <= 0 or tr >= 1:
            raise ValidationError("train_ratio must be between 0 and 1")

        vr = val_ratio
        ter = test_ratio
        if vr is None and ter is None:
            remainder = 1.0 - tr
            if remainder <= 0:
                raise ValidationError("train_ratio must be less than 1 to compute val/test ratios")
            vr = remainder * 0.7
            ter = remainder * 0.3
        elif vr is None:
            try:
                ter = float(ter)
            except Exception:
                raise ValidationError("test_ratio must be a number")
            vr = 1.0 - tr - float(ter)
        elif ter is None:
            try:
                vr = float(vr)
            except Exception:
                raise ValidationError("val_ratio must be a number")
            ter = 1.0 - tr - float(vr)
        else:
            try:
                vr = float(vr)
            except Exception:
                raise ValidationError("val_ratio must be a number")
            try:
                ter = float(ter)
            except Exception:
                raise ValidationError("test_ratio must be a number")

        if vr <= 0 or vr >= 1:
            raise ValidationError("val_ratio must be between 0 and 1")
        if ter <= 0 or ter >= 1:
            raise ValidationError("test_ratio must be between 0 and 1")
        if abs((tr + vr + ter) - 1.0) > 1e-6:
            raise ValidationError("train_ratio + val_ratio + test_ratio must equal 1")

        return float(tr), float(vr), float(ter)

    def split_dataset(
        self,
        db: Session,
        dataset_id: int,
        *,
        version_id: int | None = None,
        train_ratio: float = 0.9,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int | None = None,
        shuffle: bool = True,
        overwrite: bool = True,
        trigger: str | None = None,
    ) -> dict:
        ds = self.get_dataset(db, dataset_id)
        if ds.dataset_type != DatasetType.DETECTION:
            raise ValidationError("split_dataset is only supported for detection datasets")
        ver = self._resolve_dataset_version(db, ds, version_id)
        self._ensure_images_indexed(db, ds, ver)

        train_ratio, val_ratio, test_ratio = self._normalize_split_ratios(train_ratio, val_ratio, test_ratio)

        q = db.query(DatasetImage.image_id).filter(
            DatasetImage.dataset_id == int(ds.dataset_id),
            DatasetImage.dataset_version_id == int(ver.version_id),
        )
        if not overwrite:
            q = q.filter(DatasetImage.split.is_(None))

        ids = [int(r[0]) for r in q.order_by(DatasetImage.image_id).all()]
        total = len(ids)
        if total <= 0:
            raise ValidationError("No images available for split")

        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(ids)

        train_count = int(total * float(train_ratio))
        val_count = int(total * float(val_ratio))
        test_count = total - train_count - val_count
        if test_count < 0:
            # Best-effort correction if rounding makes us overshoot.
            deficit = -int(test_count)
            if val_count >= deficit:
                val_count -= deficit
                test_count = 0
            elif train_count >= (deficit - val_count):
                train_count -= (deficit - val_count)
                val_count = 0
                test_count = 0
            else:
                raise ValidationError("Invalid split ratios for dataset size")

        train_ids = ids[:train_count]
        val_ids = ids[train_count : train_count + val_count]
        test_ids = ids[train_count + val_count :]

        def _chunked(seq: list[int], size: int = 1000):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

        for chunk in _chunked(train_ids):
            db.query(DatasetImage).filter(DatasetImage.image_id.in_(chunk)).update(
                {DatasetImage.split: DatasetSplit.TRAIN, DatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )
        for chunk in _chunked(val_ids):
            db.query(DatasetImage).filter(DatasetImage.image_id.in_(chunk)).update(
                {DatasetImage.split: DatasetSplit.VAL, DatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )
        for chunk in _chunked(test_ids):
            db.query(DatasetImage).filter(DatasetImage.image_id.in_(chunk)).update(
                {DatasetImage.split: DatasetSplit.TEST, DatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )

        export_meta = self._export_split_files_and_update_yaml(db, ds, ver)

        total_images = int(
            db.query(DatasetImage.image_id)
            .filter(DatasetImage.dataset_id == int(ds.dataset_id), DatasetImage.dataset_version_id == int(ver.version_id))
            .count()
        )
        train_total = int(
            db.query(DatasetImage.image_id)
            .filter(
                DatasetImage.dataset_id == int(ds.dataset_id),
                DatasetImage.dataset_version_id == int(ver.version_id),
                DatasetImage.split == DatasetSplit.TRAIN,
            )
            .count()
        )
        val_total = int(
            db.query(DatasetImage.image_id)
            .filter(
                DatasetImage.dataset_id == int(ds.dataset_id),
                DatasetImage.dataset_version_id == int(ver.version_id),
                DatasetImage.split == DatasetSplit.VAL,
            )
            .count()
        )
        test_total = int(
            db.query(DatasetImage.image_id)
            .filter(
                DatasetImage.dataset_id == int(ds.dataset_id),
                DatasetImage.dataset_version_id == int(ver.version_id),
                DatasetImage.split == DatasetSplit.TEST,
            )
            .count()
        )

        # Best-effort event log.
        db.add(
            DatasetEvent(
                dataset_id=int(ds.dataset_id),
                version_id=int(ver.version_id),
                event_type="split_dataset",
                message="Split dataset into train/val",
                data={
                    "train_ratio": float(train_ratio),
                    "val_ratio": float(val_ratio),
                    "test_ratio": float(test_ratio),
                    "seed": int(seed) if seed is not None else None,
                    "shuffle": bool(shuffle),
                    "overwrite": bool(overwrite),
                    "total_images": int(total_images),
                    "train_count": int(train_total),
                    "val_count": int(val_total),
                    "test_count": int(test_total),
                    "train_file": export_meta.get("train_file"),
                    "val_file": export_meta.get("val_file"),
                    "test_file": export_meta.get("test_file"),
                    "yaml_updated": bool(export_meta.get("yaml_updated")),
                    "trigger": str(trigger) if trigger else None,
                },
            )
        )
        db.commit()

        return {
            "dataset_id": int(ds.dataset_id),
            "version_id": int(ver.version_id),
            "version": int(ver.version),
            "total_images": int(total_images),
            "train_count": int(train_total),
            "val_count": int(val_total),
            "test_count": int(test_total),
            "train_ratio": round((train_total / total_images), 6) if total_images else 0.0,
            "val_ratio": round((val_total / total_images), 6) if total_images else 0.0,
            "test_ratio": round((test_total / total_images), 6) if total_images else 0.0,
            "seed": int(seed) if seed is not None else None,
            "shuffle": bool(shuffle),
        }

    def _export_split_files_and_update_yaml(self, db: Session, ds: Dataset, ver: DatasetVersion) -> dict:
        dataset_root = resolve_dataset_path(ver.snapshot_path or ds.storage_path)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ValidationError(f"Dataset path does not exist: {dataset_root}")

        train_file = "train.txt"
        val_file = "val.txt"
        test_file = "test.txt"
        train_path = (dataset_root / train_file).resolve(strict=False)
        val_path = (dataset_root / val_file).resolve(strict=False)
        test_path = (dataset_root / test_file).resolve(strict=False)

        def _write_list(out_path: Path, split_value: DatasetSplit) -> int:
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            count = 0
            try:
                with tmp_path.open("w", encoding="utf-8") as f:
                    q = (
                        db.query(DatasetImage.path)
                        .filter(
                            DatasetImage.dataset_id == int(ds.dataset_id),
                            DatasetImage.dataset_version_id == int(ver.version_id),
                            DatasetImage.split == split_value,
                        )
                        .order_by(DatasetImage.image_id)
                        .yield_per(1000)
                    )
                    for row in q:
                        rel = str(row[0] or "").strip().replace("\\", "/").lstrip("/")
                        if not rel:
                            continue
                        rel_path = Path(rel)
                        if rel_path.is_absolute():
                            abs_path = rel_path
                        else:
                            abs_path = (dataset_root / rel_path).resolve(strict=False)
                        # Safety guard: ensure we only emit paths under dataset_root.
                        if abs_path != dataset_root and dataset_root not in abs_path.parents:
                            continue
                        f.write(abs_path.as_posix() + "\n")
                        count += 1
                tmp_path.replace(out_path)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
            return int(count)

        train_count = _write_list(train_path, DatasetSplit.TRAIN)
        val_count = _write_list(val_path, DatasetSplit.VAL)
        test_count = _write_list(test_path, DatasetSplit.TEST)

        yaml_updated = False
        data_yaml = dataset_root / "data.yaml"
        if not data_yaml.exists():
            if ds.dataset_type in (DatasetType.DETECTION, DatasetType.SEGMENTATION):
                try:
                    FileService()._create_yolo_data_yaml(dataset_root, data_yaml)
                except Exception:
                    raise ValidationError("data.yaml not found and could not be created")

        if data_yaml.exists():
            try:
                cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}
            cfg["train"] = train_file
            cfg["val"] = val_file
            cfg["test"] = test_file
            with open(data_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            yaml_updated = True

        return {
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "train_count": int(train_count),
            "val_count": int(val_count),
            "test_count": int(test_count),
            "yaml_updated": bool(yaml_updated),
        }

    def get_split_result(
        self,
        db: Session,
        dataset_id: int,
        *,
        version_id: int | None = None,
        split: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[DatasetImage], dict, int]:
        ds = self.get_dataset(db, dataset_id)
        ver = self._resolve_dataset_version(db, ds, version_id)
        self._ensure_images_indexed(db, ds, ver)

        base_q = db.query(DatasetImage).filter(
            DatasetImage.dataset_id == int(ds.dataset_id),
            DatasetImage.dataset_version_id == int(ver.version_id),
        )

        total_images = int(base_q.count())
        train_count = int(base_q.filter(DatasetImage.split == DatasetSplit.TRAIN).count())
        val_count = int(base_q.filter(DatasetImage.split == DatasetSplit.VAL).count())
        test_count = int(base_q.filter(DatasetImage.split == DatasetSplit.TEST).count())

        split_norm = (split or "").strip().lower()
        if split_norm:
            if split_norm == "train":
                base_q = base_q.filter(DatasetImage.split == DatasetSplit.TRAIN)
            elif split_norm == "val":
                base_q = base_q.filter(DatasetImage.split == DatasetSplit.VAL)
            elif split_norm == "test":
                base_q = base_q.filter(DatasetImage.split == DatasetSplit.TEST)
            elif split_norm in ("none", "null", "unassigned", "unsplit"):
                base_q = base_q.filter(DatasetImage.split.is_(None))
            else:
                raise ValidationError("split must be one of: train, val, test, unassigned")

        total = int(base_q.count())
        items = (
            base_q.order_by(DatasetImage.image_id)
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )

        summary = {
            "dataset_id": int(ds.dataset_id),
            "version_id": int(ver.version_id),
            "version": int(ver.version),
            "total_images": int(total_images),
            "train_count": int(train_count),
            "val_count": int(val_count),
            "test_count": int(test_count),
            "train_ratio": round((train_count / total_images), 6) if total_images else 0.0,
            "val_ratio": round((val_count / total_images), 6) if total_images else 0.0,
            "test_ratio": round((test_count / total_images), 6) if total_images else 0.0,
        }

        return items, summary, int(total)

    def _dataset_static_prefix(self, storage_path: str) -> str | None:
        """
        Convert dataset.storage_path to a /static/datasets/... URL prefix.

        Returns None if storage_path is not a safe relative token.
        """
        token = str(storage_path or "").strip().replace("\\", "/")
        marker = "/static/datasets/"
        if marker in token:
            token = token.split(marker, 1)[1]
        token = token.strip("/\\")
        if not token:
            return None
        p = Path(token)
        if p.is_absolute() or ".." in p.parts:
            return None
        return f"/static/datasets/{token}"

    def _iter_manifest_entries(self, manifest_rel: str):
        """
        Yield (path, size_bytes, mtime) from a NDJSON manifest.
        """
        base = settings.datasets_dir.resolve()
        path = (base / str(manifest_rel)).resolve(strict=False)
        if base not in path.parents and path != base:
            raise ValidationError("Unsafe manifest path")
        if not path.exists() or not path.is_file():
            raise NotFoundError("Manifest file not found")

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('{"_header"'):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield obj.get("path"), obj.get("size_bytes"), obj.get("mtime")

    def _trigger_thumbnail_pregeneration(
        self,
        *,
        dataset_id: int,
        version_id: int,
        dataset_path: Path,
        cache_prefix: str | None = None,
    ) -> None:
        """
        Start thumbnail pre-generation in a background thread.

        This allows the main request to return quickly while thumbnails are
        generated asynchronously. The frontend will use static file paths
        and gracefully handle missing thumbnails via onerror fallback.
        """
        job_key = self._artifact_job_key("thumbnails", int(dataset_id), int(version_id) if cache_prefix else None)
        current = self._get_artifact_job_summary(job_key)
        if str(current.get("state") or "").lower() in {"queued", "running"}:
            return

        self._set_artifact_job_summary(job_key, state="queued", progress=0)

        def _generate():
            self._set_artifact_job_summary(job_key, state="running", progress=1)
            try:
                svc = ThumbnailService()
                result = svc.pregenerate_for_dataset(
                    dataset_id=int(dataset_id),
                    dataset_root=dataset_path,
                    size=max(16, int(settings.thumbnail_size or 200)),
                    max_workers=max(1, int(settings.thumbnail_max_workers or 1)),
                    cache_prefix=cache_prefix,
                )
                total = int(result.get("total") or 0)
                processed = int(result.get("generated") or 0) + int(result.get("skipped") or 0) + int(result.get("failed") or 0)
                progress = 100 if total <= 0 else int(min(100, round((processed / max(1, total)) * 100)))
                self._set_artifact_job_summary(
                    job_key,
                    state="ready",
                    progress=progress,
                    total=total,
                    generated=int(result.get("generated") or 0),
                    skipped=int(result.get("skipped") or 0),
                    failed=int(result.get("failed") or 0),
                )
                logger.info(
                    "Thumbnail pre-generation complete for dataset %s version %s: %s",
                    dataset_id,
                    version_id,
                    result,
                )
            except Exception as e:
                self._set_artifact_job_summary(job_key, state="failed", progress=0, error=str(e))
                logger.error(
                    "Thumbnail pre-generation failed for dataset %s version %s: %s",
                    dataset_id,
                    version_id,
                    e,
                )

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

