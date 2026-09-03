from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from sqlalchemy.orm import Session

from train_platform.domains.datasets.illegal import labels, versions
from train_platform.domains.datasets.illegal.cas import (
    illegal_dataset_temp_root,
    load_version_manifest,
    materialize_manifest_to_dir,
)
from train_platform.domains.datasets.illegal.events import add_event
from train_platform.domains.datasets.illegal.publishing.converter import convert_dataset
from train_platform.domains.datasets.storage.mounted import validate_mounted_source_root
from train_platform.platform.filesystem import remove_tree
from train_platform.utils.exceptions import NotFoundError, ValidationError


def _mounted_publish_source_root(version: Any) -> Path | None:
    meta = version.meta if isinstance(version.meta, dict) else {}
    if str(meta.get("source_type") or "") != "mounted_dir_link":
        return None
    if str(meta.get("format") or "").lower().strip() != "json":
        return None
    raw_source_root = str(meta.get("source_root") or "").strip()
    if not raw_source_root:
        raise ValidationError("Mounted illegal dataset source root is missing")
    source_root = Path(raw_source_root).expanduser().resolve(strict=False)
    if not source_root.exists() or not source_root.is_dir():
        raise NotFoundError("Mounted illegal dataset source directory is no longer available")
    validate_mounted_source_root(source_root)
    return source_root


def _publish_config(
    db: Session,
    dataset: Any,
    version: Any,
    obj: dict[str, Any],
    *,
    publish_job_id: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    mapping_snapshot = labels.mapping_snapshot(
        db,
        int(dataset.illegal_dataset_id),
        overrides=obj.get("label_mapping_overrides") if isinstance(obj.get("label_mapping_overrides"), dict) else None,
    )
    label_filters = [str(item) for item in (obj.get("label_filters") or []) if str(item).strip()]
    publish_config = {
        "source_illegal_dataset_id": int(dataset.illegal_dataset_id),
        "source_illegal_dataset_name": str(dataset.name),
        "source_illegal_version_id": int(version.version_id),
        "source_version": int(version.version),
        "publish_job_id": str(publish_job_id) if publish_job_id else None,
        "idempotency_key": str(idempotency_key) if idempotency_key else None,
        "label_mappings": mapping_snapshot,
        "label_filters": label_filters,
        "split": obj.get("split") or {},
        **(obj.get("publish_config") or {}),
    }
    if publish_config.get("publish_job_id") is None:
        publish_config.pop("publish_job_id", None)
    if publish_config.get("idempotency_key") is None:
        publish_config.pop("idempotency_key", None)
    return mapping_snapshot, label_filters, publish_config


def prepare_publish_snapshot(
    db: Session,
    illegal_dataset_id: int,
    *,
    obj: dict[str, Any],
    publish_job_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    from train_platform.domains.datasets.illegal.service import IllegalDatasetService

    dataset = IllegalDatasetService().get_dataset(db, illegal_dataset_id)
    version = versions.selected_version(db, dataset, version_id=obj.get("version_id"))
    mapping_snapshot, label_filters, publish_config = _publish_config(
        db,
        dataset,
        version,
        obj,
        publish_job_id=publish_job_id,
        idempotency_key=idempotency_key,
    )
    return {
        "illegal_dataset_id": int(dataset.illegal_dataset_id),
        "dataset_type": dataset.dataset_type,
        "version_id": int(version.version_id),
        "version": int(version.version),
        "version_manifest_path": str(version.manifest_path or ""),
        "version_meta": dict(version.meta or {}) if isinstance(version.meta, dict) else {},
        "mapping_snapshot": mapping_snapshot,
        "label_filters": label_filters,
        "publish_config": publish_config,
    }


def _snapshot_version(snapshot: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        version_id=int(snapshot["version_id"]),
        version=int(snapshot["version"]),
        manifest_path=str(snapshot.get("version_manifest_path") or ""),
        meta=dict(snapshot.get("version_meta") or {}),
    )


def _conversion_config(publish_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairs_total": int(publish_result.get("pairs_total", 0)),
        "pairs_processed": int(publish_result.get("pairs_processed", 0)),
        "pairs_skipped": int(publish_result.get("pairs_skipped", 0)),
        "skipped_details": publish_result.get("skipped_details") or [],
        "warnings": publish_result.get("warnings") or [],
        "class_names": publish_result.get("class_names") or [],
        "stats": publish_result.get("stats") or {},
        "split_summary": publish_result.get("split_summary"),
        "normalized_slice_config": publish_result.get("normalized_slice_config") or {},
    }


def materialize_publish_snapshot(
    snapshot: dict[str, Any],
    *,
    obj: dict[str, Any],
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(dir=illegal_dataset_temp_root()))
    source_root = temp_dir / "illegal_source"
    processed_root = temp_dir / "standard_publish"
    version = _snapshot_version(snapshot)
    mapping_snapshot = dict(snapshot.get("mapping_snapshot") or {})
    label_filters = list(snapshot.get("label_filters") or [])
    publish_config = dict(snapshot.get("publish_config") or {})
    try:
        if callable(progress_callback):
            progress_callback("materializing", {"message": f"正在准备原始数据集版本 v{int(version.version)}"})
        mounted_root = _mounted_publish_source_root(version)
        if mounted_root is not None:
            source_root = mounted_root
        else:
            materialize_manifest_to_dir(load_version_manifest(version), source_root, replace=True)
        if callable(progress_callback):
            progress_callback("converting", {"message": "原始数据已准备完成，开始执行格式转换"})
        result = convert_dataset(
            source_root,
            processed_root,
            label_mapping=mapping_snapshot,
            label_filters=label_filters,
            publish_config=obj.get("publish_config") or {},
            split_config=obj.get("split") or {},
            progress_callback=progress_callback,
        )
        publish_config["conversion_result"] = _conversion_config(result)
        return {
            "temp_dir": str(temp_dir),
            "processed_root": str(processed_root),
            "publish_result": result,
            "publish_config": publish_config,
        }
    except Exception:
        remove_tree(temp_dir, ignore_errors=True)
        raise


def finalize_publish_snapshot(
    db: Session,
    snapshot: dict[str, Any],
    materialized: dict[str, Any],
    *,
    obj: dict[str, Any],
) -> dict[str, Any]:
    from train_platform.services.v3.standard_dataset_service import StandardDatasetService

    temp_dir_value = str(materialized.get("temp_dir") or "").strip()
    if not temp_dir_value:
        raise ValidationError("Publish materialization is missing its temporary directory")
    temp_dir = Path(temp_dir_value)
    result = materialized.get("publish_result") if isinstance(materialized.get("publish_result"), dict) else {}
    publish_config = materialized.get("publish_config") if isinstance(materialized.get("publish_config"), dict) else {}
    try:
        processed_root_value = str(materialized.get("processed_root") or "").strip()
        if not processed_root_value:
            raise ValidationError("Publish materialization is missing its processed source root")
        processed_root = Path(processed_root_value)
        standard = StandardDatasetService().materialize_from_source_tree(
            db,
            name=str(obj.get("name") or "").strip(),
            dataset_type=snapshot["dataset_type"],
            source_root=processed_root,
            description=obj.get("description"),
            source_type="illegal_publish",
            publish_config=publish_config,
            commit=False,
        )
        add_event(
            db,
            int(snapshot["illegal_dataset_id"]),
            "published",
            version_id=int(snapshot["version_id"]),
            message=f"Published standard dataset {standard.name}",
            data={
                "standard_dataset_id": int(standard.standard_dataset_id),
                "pairs_processed": int(result.get("pairs_processed", 0)),
                "pairs_skipped": int(result.get("pairs_skipped", 0)),
            },
        )
        db.commit()
        db.refresh(standard)
        return {
            "standard_dataset_id": int(standard.standard_dataset_id),
            "name": standard.name,
            "source_illegal_dataset_id": int(snapshot["illegal_dataset_id"]),
            "source_illegal_version_id": int(snapshot["version_id"]),
            "publish_config": publish_config,
        }
    finally:
        remove_tree(temp_dir, ignore_errors=True)


def publish_standard_dataset(
    db: Session,
    illegal_dataset_id: int,
    *,
    obj: dict[str, Any],
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    publish_job_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    snapshot = prepare_publish_snapshot(
        db,
        illegal_dataset_id,
        obj=obj,
        publish_job_id=publish_job_id,
        idempotency_key=idempotency_key,
    )
    materialized = materialize_publish_snapshot(
        snapshot,
        obj=obj,
        progress_callback=progress_callback,
    )
    return finalize_publish_snapshot(db, snapshot, materialized, obj=obj)


def cleanup_materialized_publish(materialized: dict[str, Any] | None) -> None:
    if not isinstance(materialized, dict):
        return
    temp_dir = str(materialized.get("temp_dir") or "").strip()
    if temp_dir:
        remove_tree(Path(temp_dir), ignore_errors=True)
