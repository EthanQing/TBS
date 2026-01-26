from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.core.config import settings
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
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_dataset_path


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
    def __init__(self) -> None:
        self.datasets = DatasetRepository()
        self.versions = DatasetVersionRepository()
        self.events = DatasetEventRepository()

    # --------------------
    # datasets
    # --------------------
    def list_datasets(self, db: Session, *, skip: int = 0, limit: int = 100) -> list[Dataset]:
        return self.datasets.get_multi(db, skip=skip, limit=limit)

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

        projects = db.query(Project).filter(Project.dataset_id == ds.dataset_id).all()
        if projects and not force:
            raise ConflictError(f"Cannot delete dataset; {len(projects)} project(s) still reference it")

        if projects and force:
            svc = ProjectService()
            for p in projects:
                svc.delete_project(db, int(p.project_id), force=True)
            # Refresh dataset after project deletions/commits.
            ds = self.get_dataset(db, dataset_id)

        # Remove DB rows first.
        db.delete(ds)
        db.commit()

        if not delete_files:
            return

        base = settings.datasets_dir.resolve()
        abs_path = resolve_dataset_path(ds.storage_path).resolve(strict=False)
        if base != abs_path and base not in abs_path.parents:
            raise ConflictError("Refusing to delete dataset outside BASE_DATASETS_DIR")

        try:
            if abs_path.exists():
                if abs_path.is_dir():
                    shutil.rmtree(abs_path, ignore_errors=True)
                else:
                    abs_path.unlink(missing_ok=True)
        except Exception:
            # Best-effort.
            pass

        # Also remove version snapshots/manifests for this dataset.
        try:
            versions_dir = (base / ".versions" / _versions_token(ds)).resolve(strict=False)
            if base == versions_dir or base in versions_dir.parents:
                if versions_dir.exists() and versions_dir.is_dir():
                    shutil.rmtree(versions_dir, ignore_errors=True)
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
    ):
        ds = self.get_dataset(db, dataset_id)

        dataset_root = resolve_dataset_path(ds.storage_path)
        base = settings.datasets_dir.resolve()
        if dataset_root == base or (base not in dataset_root.parents and dataset_root != base):
            raise ValidationError("Invalid dataset storage_path (must be under BASE_DATASETS_DIR)")

        FileService().upload_dataset_into_existing(file, dataset_root, ds.dataset_type)

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

        await run_in_threadpool(FileService().upload_dataset_into_existing, file, dataset_root, ds.dataset_type)

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

        allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

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

            data_yaml_path = (dataset_root / "data.yaml").resolve(strict=False)
            if not data_yaml_path.exists():
                # We need an existing schema (nc/names) to validate "no new classes".
                try:
                    from train_platform.services.file_service import FileService

                    FileService()._create_yolo_data_yaml(dataset_root, data_yaml_path)
                except Exception:
                    raise ValidationError("data.yaml not found; cannot validate labels/classes")

            try:
                cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}

            base_names = self._normalize_yolo_names(cfg.get("names"), cfg.get("nc"))
            nc_before = int(len(base_names))
            if nc_before <= 0:
                raise ValidationError("data.yaml missing valid 'names'/'nc'; cannot validate labels/classes")

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
                yolo_yaml = dataset_path / "data.yaml"
                if yolo_yaml.exists() and yolo_yaml.is_file():
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

            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
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

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
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

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

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

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
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

    def split_dataset(
        self,
        db: Session,
        dataset_id: int,
        *,
        version_id: int | None = None,
        train_ratio: float = 0.8,
        val_ratio: float | None = None,
        seed: int | None = None,
        shuffle: bool = True,
        overwrite: bool = True,
    ) -> dict:
        ds = self.get_dataset(db, dataset_id)
        ver = self._resolve_dataset_version(db, ds, version_id)
        self._ensure_images_indexed(db, ds, ver)

        if train_ratio <= 0 or train_ratio >= 1:
            raise ValidationError("train_ratio must be between 0 and 1")
        if val_ratio is None:
            val_ratio = 1.0 - float(train_ratio)
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValidationError("val_ratio must be between 0 and 1")
        if abs((float(train_ratio) + float(val_ratio)) - 1.0) > 1e-6:
            raise ValidationError("train_ratio + val_ratio must equal 1")

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
        train_count = max(0, min(train_count, total))
        val_count = total - train_count

        train_ids = ids[:train_count]
        val_ids = ids[train_count:]

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
                    "seed": int(seed) if seed is not None else None,
                    "shuffle": bool(shuffle),
                    "overwrite": bool(overwrite),
                    "total_images": int(total_images),
                    "train_count": int(train_total),
                    "val_count": int(val_total),
                    "train_file": export_meta.get("train_file"),
                    "val_file": export_meta.get("val_file"),
                    "yaml_updated": bool(export_meta.get("yaml_updated")),
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
            "train_ratio": round((train_total / total_images), 6) if total_images else 0.0,
            "val_ratio": round((val_total / total_images), 6) if total_images else 0.0,
            "seed": int(seed) if seed is not None else None,
            "shuffle": bool(shuffle),
        }

    def _export_split_files_and_update_yaml(self, db: Session, ds: Dataset, ver: DatasetVersion) -> dict:
        dataset_root = resolve_dataset_path(ver.snapshot_path or ds.storage_path)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ValidationError(f"Dataset path does not exist: {dataset_root}")

        train_file = "train.txt"
        val_file = "val.txt"
        train_path = (dataset_root / train_file).resolve(strict=False)
        val_path = (dataset_root / val_file).resolve(strict=False)

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
            with open(data_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            yaml_updated = True

        return {
            "train_file": train_file,
            "val_file": val_file,
            "train_count": int(train_count),
            "val_count": int(val_count),
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

        split_norm = (split or "").strip().lower()
        if split_norm:
            if split_norm == "train":
                base_q = base_q.filter(DatasetImage.split == DatasetSplit.TRAIN)
            elif split_norm == "val":
                base_q = base_q.filter(DatasetImage.split == DatasetSplit.VAL)
            elif split_norm in ("none", "null", "unassigned", "unsplit"):
                base_q = base_q.filter(DatasetImage.split.is_(None))
            else:
                raise ValidationError("split must be one of: train, val, unassigned")

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
            "train_ratio": round((train_count / total_images), 6) if total_images else 0.0,
            "val_ratio": round((val_count / total_images), 6) if total_images else 0.0,
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
