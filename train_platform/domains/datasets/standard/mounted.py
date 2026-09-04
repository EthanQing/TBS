from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.datasets import yolo
from train_platform.domains.datasets.labelme import (
    bbox_to_yolo,
    choose_image_link,
    collect_image_json_pairs,
    image_rel_for_source,
    label_rel_for_image,
    parse_annotations,
)
from train_platform.domains.datasets.storage.files import count_tree
from train_platform.domains.datasets.storage.mounted import validate_mounted_source_root, write_mounted_manifest
from train_platform.models.v3.enums import DatasetSplit
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.platform.filesystem import clear_directory, remove_path, remove_tree
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.domains.datasets.images import IMAGE_EXTS

from .service import StandardDatasetService


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_directory_link(source: Path, target: Path) -> str:
    src = Path(source).resolve(strict=False)
    if not src.exists() or not src.is_dir():
        raise NotFoundError("Mounted source directory not found")
    dst = Path(target).resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove_path(dst)
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
        return "symlink"
    except OSError as symlink_exc:
        if os.name != "nt":
            raise ValidationError(f"Failed to create directory symlink: {symlink_exc}") from symlink_exc
        try:
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            raise ValidationError(f"Failed to create Windows junction: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or str(symlink_exc)).strip()
            raise ValidationError(f"Failed to create Windows junction: {detail}")
        return "junction"


def _write_class_files(root: Path, class_names: list[str]) -> list[str]:
    names = [str(name).strip() for name in class_names if str(name).strip()]
    if not names:
        names = ["class_0"]
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def _infer_class_names_from_labels(labels_root: Path) -> list[str]:
    class_names: list[str] = []
    if not labels_root.exists():
        return class_names
    for label_file in sorted(labels_root.rglob("*.txt")):
        if label_file.name.lower() in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
            continue
        try:
            lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = [part for part in line.split() if part]
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            while len(class_names) <= class_id:
                class_names.append(f"class_{len(class_names)}")
    return class_names


def write_yolo_yaml(root: Path, class_names: list[str], image_rels: list[str]) -> None:
    root = Path(root)
    names = _write_class_files(root, class_names)
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rel in sorted(image_rels):
        split = yolo.detect_split_from_relpath(rel)
        key = split.value if isinstance(split, DatasetSplit) else "train"
        buckets.setdefault(key, []).append(rel)
    if not buckets["train"] and (buckets["val"] or buckets["test"]):
        buckets["train"] = buckets["val"] or buckets["test"]
    for split_name, rels in buckets.items():
        if not rels and split_name != "val":
            continue
        if split_name == "val" and not rels:
            rels = buckets["train"]
        (root / f"{split_name}.txt").write_text("\n".join(rels) + ("\n" if rels else ""), encoding="utf-8")
    payload: dict[str, Any] = {
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(names),
        "names": names,
    }
    if buckets["test"]:
        payload["test"] = "test.txt"
    with (root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _copy_source_label(source_root: Path, image_path: Path, image_rel: str, target_root: Path) -> bool:
    source_root = Path(source_root).resolve(strict=False)
    rel_to_source = Path(image_path).resolve(strict=False).relative_to(source_root)
    candidates = [
        source_root / label_rel_for_image(rel_to_source),
        source_root / label_rel_for_image(image_rel.replace("images/source/", "", 1)),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            destination = target_root / label_rel_for_image(image_rel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            return True
    return False


def link_yolo_source_tree(target_root: Path, source_root: Path) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = Path(source_root).resolve(strict=False)
    source_image_root, image_rel_prefix = choose_image_link(source_root)
    link_type = create_directory_link(source_image_root, target_root / image_rel_prefix)
    image_rels: list[str] = []
    copied_labels = 0
    for image_path in sorted(
        path for path in source_image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ):
        image_rel = image_rel_for_source(source_root, source_image_root, image_path, image_rel_prefix)
        image_rels.append(image_rel)
        if _copy_source_label(source_root, image_path, image_rel, target_root):
            copied_labels += 1
        else:
            label_path = target_root / label_rel_for_image(image_rel)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")
    class_names = yolo.read_class_names(source_root) or _infer_class_names_from_labels(target_root / "labels")
    write_yolo_yaml(target_root, class_names, image_rels)
    manifest = {
        "schema_version": 1,
        "source_type": "mounted_dir_link",
        "format": "yolo",
        "source_root": str(source_root),
        "source_image_root": str(source_image_root),
        "image_rel_prefix": image_rel_prefix,
        "link_type": link_type,
        "created_at": _utcnow_iso(),
        "image_count": len(image_rels),
        "image_paths": sorted(image_rels),
        "label_count": copied_labels,
    }
    write_mounted_manifest(target_root, manifest)
    return manifest


def link_json_source_tree(target_root: Path, source_root: Path) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = Path(source_root).resolve(strict=False)
    source_image_root, image_rel_prefix = choose_image_link(source_root)
    link_type = create_directory_link(source_image_root, target_root / image_rel_prefix)
    pairs, warnings = collect_image_json_pairs(source_root)
    if not pairs:
        raise ValidationError("No image/json pairs found in mounted directory")
    label_map: dict[str, int] = {}
    image_rels: list[str] = []
    skipped: list[str] = []
    object_count = 0
    base_cfg = {
        "label_map": label_map,
        "label_strategy": "full",
        "label_separator": "%",
        "min_probability": 0.0,
        "skip_hidden": True,
        "skip_outside": True,
    }
    for image_path, json_path in pairs:
        try:
            image_rel = image_rel_for_source(source_root, source_image_root, image_path, image_rel_prefix)
        except Exception:
            skipped.append(f"{image_path.name}: image is outside linked image root")
            continue
        width, height = yolo.image_size(image_path)
        if not width or not height:
            skipped.append(f"{image_path.name}: cannot read image size")
            continue
        try:
            bboxes, label_map = parse_annotations({**base_cfg, "label_map": label_map, "annotation_path": str(json_path), "image_height": int(height)})
        except Exception as exc:
            skipped.append(f"{json_path.name}: {exc}")
            continue
        lines: list[str] = []
        for bbox in bboxes:
            bbox.x_min = max(0.0, min(float(width), float(bbox.x_min)))
            bbox.y_min = max(0.0, min(float(height), float(bbox.y_min)))
            bbox.x_max = max(0.0, min(float(width), float(bbox.x_max)))
            bbox.y_max = max(0.0, min(float(height), float(bbox.y_max)))
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            lines.append(bbox_to_yolo(bbox, int(width), int(height)))
        label_path = target_root / label_rel_for_image(image_rel)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        image_rels.append(image_rel)
        object_count += len(lines)
    if not image_rels:
        detail = "; ".join(skipped[:10])
        raise ValidationError(f"No valid image/json pairs imported. {detail}")
    class_names = [name for name, _cid in sorted(label_map.items(), key=lambda item: item[1])]
    write_yolo_yaml(target_root, class_names, image_rels)
    manifest = {
        "schema_version": 1,
        "source_type": "mounted_dir_link",
        "format": "json",
        "source_root": str(source_root),
        "source_image_root": str(source_image_root),
        "image_rel_prefix": image_rel_prefix,
        "link_type": link_type,
        "created_at": _utcnow_iso(),
        "image_count": len(image_rels),
        "image_paths": sorted(image_rels),
        "json_count": len(pairs),
        "object_count": object_count,
        "warnings": warnings + skipped[:50],
    }
    write_mounted_manifest(target_root, manifest)
    return manifest


def link_source_tree(target_root: Path, source_root: Path, *, prefer_yolo: bool = True) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = validate_mounted_source_root(Path(source_root))
    if not source_root.exists() or not source_root.is_dir():
        raise NotFoundError("Mounted source directory not found")
    target_root.mkdir(parents=True, exist_ok=True)
    has_yolo = (source_root / "labels").exists() or any(
        (source_root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")
    )
    if prefer_yolo and has_yolo:
        return link_yolo_source_tree(target_root, source_root)
    return link_json_source_tree(target_root, source_root)


def import_mounted_source_tree(
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
    source = validate_mounted_source_root(Path(source_root))
    from .content import _resolve_uploaded_yolo_root

    detected_yolo_root = _resolve_uploaded_yolo_root(source)
    yolo_root = detected_yolo_root or source
    staging_parent = settings.dataset_staging_dir / "standard"
    staging_parent.mkdir(parents=True, exist_ok=True)
    materialized = staging_parent / f"{int(dataset.standard_dataset_id)}-linked-{uuid.uuid4().hex}"
    try:
        materialized.mkdir(parents=True, exist_ok=True)
        manifest = link_source_tree(materialized, yolo_root, prefer_yolo=detected_yolo_root is not None)
        clear_directory(root)
        for item in materialized.iterdir():
            shutil.move(str(item), str(root / item.name))
        dataset.source_type = "mounted_dir_link"
        dataset.publish_config = {
            **(dataset.publish_config or {}),
            "mounted_import": {
                "source_root": str(source.resolve(strict=False)),
                "linked_source": str(yolo_root.resolve(strict=False)),
                "format": manifest.get("format"),
                "link_type": manifest.get("link_type"),
                "image_count": int(manifest.get("image_count") or 0),
                "filename": str(filename or ""),
            },
        }
        from .content import index_mounted_images
        from .queries import refresh_statistics, refresh_view_index

        index_mounted_images(db, dataset, root)
        refresh_statistics(db, dataset, commit=False)
        refresh_view_index(db, dataset, commit=False)
        from .events import add_event

        add_event(
            db,
            int(dataset.standard_dataset_id),
            "mounted_imported",
            message="Standard dataset imported from mounted directory",
            created_by=created_by,
            data=dataset.publish_config.get("mounted_import"),
        )
        db.commit()
        db.refresh(dataset)
        return dataset
    finally:
        remove_tree(materialized, ignore_errors=True)


__all__ = [
    "create_directory_link",
    "import_mounted_source_tree",
    "link_json_source_tree",
    "link_source_tree",
    "link_yolo_source_tree",
    "write_yolo_yaml",
]
