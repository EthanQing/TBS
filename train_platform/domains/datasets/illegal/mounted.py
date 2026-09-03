from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from train_platform.core.config import settings
from train_platform.domains.datasets.labelme import (
    choose_image_link,
    collect_image_json_pairs,
    image_rel_for_source,
    iter_regular_files,
)
from train_platform.domains.datasets.storage.mounted import mounted_file_entry, validate_mounted_source_root
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_shapes(path: Path) -> tuple[list[Any], list[str]]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        try:
            with Path(path).open("r", encoding="gbk", errors="ignore") as f:
                data = json.load(f)
        except Exception as exc:
            return [], [f"{Path(path).name}: {exc}"]
    if isinstance(data, dict) and isinstance(data.get("shapes"), list):
        return data["shapes"], []
    if isinstance(data, list):
        return data, []
    return [], []


def _emit_manifest_progress(
    progress_callback: Callable[[int, str, dict[str, Any]], None] | None,
    progress: int,
    stage: str,
    *,
    processed_count: int | None = None,
    total_count: int | None = None,
    current_item: str | None = None,
    detail_message: str | None = None,
) -> None:
    if not progress_callback:
        return
    payload = {
        "processed_count": processed_count,
        "total_count": total_count,
        "current_item": current_item,
        "detail_message": detail_message,
    }
    try:
        progress_callback(int(progress), str(stage), payload)
    except TypeError:
        progress_callback(int(progress), str(stage))  # type: ignore[misc]


def build_illegal_mounted_manifest(
    source_root: Path,
    *,
    prefer_yolo: bool = True,
    progress_callback: Callable[[int, str, dict[str, Any]], None] | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Build a lightweight illegal-dataset manifest without converting labels."""
    source_root = validate_mounted_source_root(Path(source_root))
    if not source_root.exists() or not source_root.is_dir():
        raise NotFoundError("Mounted source directory not found")
    worker_count = max(1, int(max_workers or settings.dataset_import_max_workers or 1))

    has_yolo = (source_root / "labels").exists() or any((source_root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"))
    if prefer_yolo and has_yolo:
        _emit_manifest_progress(progress_callback, 5, "scanning", detail_message="Scanning mounted YOLO source")
        files: dict[str, dict[str, Any]] = {}
        image_paths: list[str] = []
        label_count = 0
        for path in sorted(iter_regular_files(source_root), key=lambda p: p.relative_to(source_root).as_posix()):
            rel = path.relative_to(source_root).as_posix()
            ext = path.suffix.lower()
            lower_name = path.name.lower()
            if ext in IMAGE_EXTS:
                image_paths.append(rel)
            if ext == ".txt" and lower_name not in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
                label_count += 1
            files[rel] = mounted_file_entry(path)
        _emit_manifest_progress(progress_callback, 95, "finalizing", processed_count=len(image_paths), total_count=len(image_paths), detail_message="Mounted YOLO source indexed")
        return {
            "schema_version": 1,
            "source_type": "mounted_dir_link",
            "format": "yolo",
            "source_root": str(source_root),
            "created_at": _utcnow_iso(),
            "image_count": len(image_paths),
            "image_paths": sorted(image_paths),
            "label_count": label_count,
            "files": files,
            "warnings": [],
        }

    source_image_root, image_rel_prefix = choose_image_link(source_root)
    _emit_manifest_progress(progress_callback, 5, "scanning", detail_message="Scanning mounted LabelMe/JSON source")
    pairs, warnings = collect_image_json_pairs(source_root)
    if not pairs:
        raise ValidationError("No image/json pairs found in mounted directory")
    _emit_manifest_progress(
        progress_callback,
        20,
        "pairing",
        processed_count=0,
        total_count=len(pairs),
        detail_message=f"Paired {len(pairs)} image/json items",
    )

    files: dict[str, dict[str, Any]] = {}
    image_rels: list[str] = []
    raw_labels: set[str] = set()
    object_count = 0
    batch_size = max(1, min(256, worker_count * 8))

    def _parse_pair(image_path: Path, json_path: Path) -> dict[str, Any]:
        try:
            image_rel = image_rel_for_source(source_root, source_image_root, image_path, image_rel_prefix)
        except Exception:
            return {
                "warnings": [f"{image_path.name}: image is outside linked image root"],
            }
        json_rel = json_path.relative_to(source_root).as_posix()
        image_entry = mounted_file_entry(image_path)
        json_entry = mounted_file_entry(json_path)
        shapes, shape_warnings = _read_json_shapes(json_path)
        labels: list[str] = []
        count = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label") or "").strip()
            if label:
                labels.append(label)
                count += 1
        return {
            "image_rel": image_rel,
            "json_rel": json_rel,
            "image_entry": image_entry,
            "json_entry": json_entry,
            "labels": labels,
            "object_count": count,
            "warnings": shape_warnings,
            "current_item": json_rel,
        }

    if worker_count <= 1 or len(pairs) < 2:
        results = []
        total = len(pairs)
        for completed, (image_path, json_path) in enumerate(pairs, start=1):
            result = _parse_pair(image_path, json_path)
            results.append(result)
            if completed % batch_size == 0 or completed == total:
                _emit_manifest_progress(
                    progress_callback,
                    20 + int(round(65 * completed / max(1, total))),
                    "parsing",
                    processed_count=completed,
                    total_count=total,
                    current_item=str(result.get("current_item") or ""),
                    detail_message=f"Parsed {completed}/{total} JSON files",
                )
    else:
        results = []
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mounted-import") as executor:
            futures = {
                executor.submit(_parse_pair, image_path, json_path): (image_path, json_path)
                for image_path, json_path in pairs
            }
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    image_path, json_path = futures[future]
                    result = {
                        "warnings": [f"{json_path.name}: {exc}"],
                        "current_item": json_path.relative_to(source_root).as_posix() if json_path.exists() else image_path.name,
                    }
                results.append(result)
                if completed % batch_size == 0 or completed == total:
                    _emit_manifest_progress(
                        progress_callback,
                        20 + int(round(65 * completed / max(1, total))),
                        "parsing",
                        processed_count=completed,
                        total_count=total,
                        current_item=str(result.get("current_item") or ""),
                        detail_message=f"Parsed {completed}/{total} JSON files",
                    )

    total_pairs = len(results)
    def _result_sort_key(result: dict[str, Any]) -> str:
        return str(result.get("image_rel") or result.get("json_rel") or result.get("current_item") or "")

    for result in sorted(results, key=_result_sort_key):
        warnings.extend(result.get("warnings") or [])
        image_rel = str(result.get("image_rel") or "").strip()
        json_rel = str(result.get("json_rel") or "").strip()
        if not image_rel or not json_rel:
            continue
        files[image_rel] = result["image_entry"]
        files[json_rel] = result["json_entry"]
        image_rels.append(image_rel)
        for label in result.get("labels") or []:
            label_text = str(label).strip()
            if label_text:
                raw_labels.add(label_text)
        object_count += int(result.get("object_count") or 0)
    _emit_manifest_progress(
        progress_callback,
        90,
        "indexing",
        processed_count=total_pairs,
        total_count=total_pairs,
        current_item=image_rels[-1] if image_rels else None,
        detail_message=f"Indexed {len(image_rels)} mounted images",
    )
    if not image_rels:
        raise ValidationError("No valid image/json pairs imported")
    _emit_manifest_progress(
        progress_callback,
        98,
        "finalizing",
        processed_count=len(image_rels),
        total_count=len(pairs),
        detail_message="Mounted manifest ready",
    )
    return {
        "schema_version": 1,
        "source_type": "mounted_dir_link",
        "format": "json",
        "source_root": str(source_root),
        "source_image_root": str(source_image_root),
        "image_rel_prefix": image_rel_prefix,
        "created_at": _utcnow_iso(),
        "image_count": len(image_rels),
        "image_paths": sorted(image_rels),
        "json_count": len(pairs),
        "object_count": object_count,
        "raw_labels": sorted(raw_labels),
        "files": files,
        "warnings": warnings[:50],
    }
