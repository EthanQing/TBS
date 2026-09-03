from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.datasets import yolo
from train_platform.domains.datasets.storage.files import count_tree, iter_image_files
from train_platform.domains.datasets.storage.mounted import load_mounted_manifest
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.platform.filesystem import clear_directory, copy_tree, extract_archive, remove_tree
from train_platform.utils.exceptions import ConflictError, ValidationError

from .events import add_event
from .service import StandardDatasetService


def _resolve_uploaded_yolo_root(source_root: Path) -> Path | None:
    root = Path(source_root)
    if (root / "images").exists() and (root / "labels").exists():
        return root
    if any((root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")):
        return root
    return yolo.find_yolo_export_root(root)


def image_count(db: Session, dataset: StandardDataset) -> int:
    return int(
        db.query(StandardDatasetImage)
        .filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id))
        .count()
    )


def index_images(db: Session, dataset: StandardDataset, root: Path | None = None) -> None:
    dataset_root = Path(root or StandardDatasetService().dataset_root(dataset))
    db.query(StandardDatasetImage).filter(
        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
    ).delete()
    for image_path in iter_image_files(dataset_root):
        rel = image_path.relative_to(dataset_root).as_posix()
        db.add(
            StandardDatasetImage(
                standard_dataset_id=int(dataset.standard_dataset_id),
                path=rel,
                split=yolo.detect_split_from_relpath(rel),
            )
        )
    db.flush()


def index_mounted_images(db: Session, dataset: StandardDataset, root: Path | None = None) -> None:
    dataset_root = Path(root or StandardDatasetService().dataset_root(dataset))
    manifest = load_mounted_manifest(dataset_root) or {}
    rel_paths = manifest.get("image_paths") if isinstance(manifest, dict) else []
    if not isinstance(rel_paths, list):
        rel_paths = []
    db.query(StandardDatasetImage).filter(
        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
    ).delete()
    for raw_rel in rel_paths:
        rel = str(raw_rel or "").strip().replace("\\", "/").strip("/")
        if rel:
            db.add(
                StandardDatasetImage(
                    standard_dataset_id=int(dataset.standard_dataset_id),
                    path=rel,
                    split=yolo.detect_split_from_relpath(rel),
                )
            )
    db.flush()


def ensure_image_index(db: Session, dataset: StandardDataset, *, commit: bool = False) -> bool:
    if image_count(db, dataset) > 0:
        return False
    root = StandardDatasetService().dataset_root(dataset)
    if not root.exists() or not root.is_dir() or not any(iter_image_files(root)):
        return False
    index_images(db, dataset, root)
    if commit:
        db.commit()
    return True


def _install_yolo_tree(
    db: Session,
    dataset: StandardDataset,
    source_root: Path,
    *,
    event_type: str,
    event_message: str,
    created_by: str | None = None,
    filename: str | None = None,
    event_data: dict[str, Any] | None = None,
    cleanup_root_on_error: bool = False,
    commit: bool,
) -> StandardDataset:
    root = StandardDatasetService().dataset_root(dataset)
    materialized: Path | None = None
    try:
        yolo_root = _resolve_uploaded_yolo_root(Path(source_root))
        if yolo_root is None:
            raise ValidationError("Standard dataset upload only supports YOLO format")
        staging_parent = settings.dataset_staging_dir / "standard"
        staging_parent.mkdir(parents=True, exist_ok=True)
        materialized = staging_parent / f"{int(dataset.standard_dataset_id)}-materialized-{uuid.uuid4().hex}"
        copy_tree(yolo_root, materialized)
        if not any((materialized / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")):
            yolo.create_yolo_data_yaml(materialized, materialized / "data.yaml")
        index_images(db, dataset, materialized)
        clear_directory(root)
        root.mkdir(parents=True, exist_ok=True)
        for item in materialized.iterdir():
            shutil.move(str(item), str(root / item.name))
        from .queries import refresh_statistics, refresh_view_index

        refresh_statistics(db, dataset, commit=False)
        refresh_view_index(db, dataset, commit=False)
        add_event(
            db,
            int(dataset.standard_dataset_id),
            event_type,
            message=event_message,
            created_by=created_by,
            data=event_data if event_data is not None else {"filename": str(filename or "")},
        )
        if commit:
            db.commit()
            db.refresh(dataset)
        else:
            db.flush()
        return dataset
    except Exception:
        if cleanup_root_on_error or not commit:
            remove_tree(root, ignore_errors=True)
        raise
    finally:
        if materialized is not None:
            remove_tree(materialized, ignore_errors=True)


def import_archive_file(
    db: Session,
    standard_dataset_id: int,
    archive_path: Path,
    *,
    created_by: str | None = None,
    filename: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> StandardDataset:
    staging = settings.dataset_staging_dir / "standard" / f"{int(standard_dataset_id)}-{uuid.uuid4().hex}"
    extracted_dir = staging / "extracted"
    try:
        extracted_root = extract_archive(Path(archive_path), extracted_dir, progress_callback=progress_callback)
        return import_source_tree(
            db,
            standard_dataset_id,
            extracted_root,
            created_by=created_by,
            filename=filename or Path(archive_path).name,
        )
    finally:
        remove_tree(staging, ignore_errors=True)


def import_source_tree(
    db: Session,
    standard_dataset_id: int,
    source_root: Path,
    *,
    created_by: str | None = None,
    filename: str | None = None,
) -> StandardDataset:
    service = StandardDatasetService()
    dataset = service.get_dataset(db, standard_dataset_id)
    root = service.dataset_root(dataset)
    existing_files, _ = count_tree(root)
    if existing_files > 0:
        raise ConflictError("Standard dataset content is immutable after upload")
    return _install_yolo_tree(
        db,
        dataset,
        source_root,
        event_type="uploaded",
        event_message="Standard dataset archive uploaded",
        created_by=created_by,
        filename=filename,
        event_data={"filename": str(filename or "")},
        commit=True,
    )


def materialize_from_source_tree(
    db: Session,
    *,
    name: str,
    dataset_type: Any,
    source_root: Path,
    description: str | None = None,
    source_type: str | None = None,
    publish_config: dict[str, Any] | None = None,
    created_by: str | None = None,
    commit: bool = True,
) -> StandardDataset:
    dataset = StandardDatasetService().create_dataset(
        db,
        obj={
            "name": name,
            "dataset_type": dataset_type,
            "format": "yolo",
            "description": description,
            "source_type": source_type,
            "publish_config": publish_config,
        },
        commit=False,
    )
    return _install_yolo_tree(
        db,
        dataset,
        source_root,
        event_type="published",
        event_message="Standard dataset materialized from source tree",
        created_by=created_by,
        event_data={"source_type": source_type},
        cleanup_root_on_error=True,
        commit=commit,
    )


__all__ = [
    "ensure_image_index",
    "image_count",
    "import_archive_file",
    "import_source_tree",
    "index_images",
    "index_mounted_images",
    "materialize_from_source_tree",
]
