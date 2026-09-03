from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.datasets import yolo
from train_platform.domains.datasets.thumbnails import prewarm_dataset_thumbnails as prewarm_thumbnails
from train_platform.domains.datasets.storage.files import count_tree, iter_files, iter_image_files
from train_platform.domains.datasets.storage.mounted import resolve_dataset_file
from train_platform.domains.datasets.storage.paths import ensure_dataset_relative_path
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.platform import filesystem as _filesystem
from train_platform.utils.image_exts import IMAGE_EXTS


_JSON_CACHE_LOCK = threading.Lock()
_JSON_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_cached_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        return None
    key = str(path.resolve(strict=False))
    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(key)
        if cached and float(cached[0]) == mtime:
            return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[key] = (mtime, data)
    return data


def _write_cached_json(path: Path, data: dict[str, Any]) -> Path:
    _filesystem.atomic_write_json(path, data)
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[str(path.resolve(strict=False))] = (mtime, data)
    return path


def dataset_statistics_cache_path(root: Path) -> Path:
    return Path(root) / ".dataset_stats.json"


def load_cached_statistics(root: Path) -> dict[str, Any] | None:
    return _load_cached_json(dataset_statistics_cache_path(root))


def write_cached_statistics(root: Path, stats: dict[str, Any]) -> Path:
    return _write_cached_json(dataset_statistics_cache_path(root), stats)


def dataset_view_index_cache_path(root: Path) -> Path:
    return Path(root) / ".dataset_view_index.json"


def load_cached_view_index(root: Path) -> dict[str, Any] | None:
    return _load_cached_json(dataset_view_index_cache_path(root))


def write_cached_view_index(root: Path, view_index: dict[str, Any]) -> Path:
    return _write_cached_json(dataset_view_index_cache_path(root), view_index)


def static_dataset_url(storage_token: str, rel_path: str | Path) -> str:
    rel = ensure_dataset_relative_path(rel_path).as_posix()
    base = ensure_dataset_relative_path(storage_token).as_posix()
    return f"/static/datasets/{base}/{rel}"


def dataset_thumbnail_url(
    dataset_kind: str,
    dataset_id: int,
    rel_path: str | Path,
    *,
    version_id: int | None = None,
    size: int | None = None,
) -> str:
    rel = ensure_dataset_relative_path(rel_path).as_posix()
    encoded_rel = quote(rel, safe="/")
    base = f"/api/v3/thumbnails/{quote(str(dataset_kind).strip(), safe='')}/{int(dataset_id)}/{encoded_rel}"
    params: dict[str, str] = {}
    if version_id is not None:
        params["version_id"] = str(int(version_id))
    if size is not None:
        params["size"] = str(int(size))
    query = urlencode(params)
    return f"{base}?{query}" if query else base


def build_yolo_view_index(
    root: Path,
    image_rows: list[Any],
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve(strict=False)
    class_names = yolo.read_class_names(root)
    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(image_rows, start=1):
        if isinstance(row, dict):
            rel_path = str(row.get("path") or "")
            item_id = int(row.get("id") or row.get("image_id") or idx)
        else:
            rel_path = str(getattr(row, "path", "") or "")
            item_id = int(getattr(row, "image_id", idx) or idx)
        if rel_path:
            entries.append({"id": item_id, "path": rel_path, "name": Path(rel_path).name})

    def process(entry: dict[str, Any]) -> dict[str, Any]:
        width, height, object_count, classes = yolo.read_yolo_box_summary(root, entry["path"], class_names)
        return {
            **entry,
            "width": width,
            "height": height,
            "object_count": int(object_count),
            "classes": [int(value) for value in classes],
        }

    workers = max(1, int(max_workers or settings.view_index_max_workers or 1))
    if len(entries) <= 1 or workers <= 1:
        items = [process(entry) for entry in entries]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(entries)))) as executor:
            items = list(executor.map(process, entries))

    category_counts: dict[int, int] = {}
    for item in items:
        for class_id in item.get("classes", []):
            category_counts[int(class_id)] = int(category_counts.get(int(class_id), 0)) + 1
    categories = [
        {
            "class_id": int(class_id),
            "name": class_names[class_id] if 0 <= int(class_id) < len(class_names) else str(class_id),
            "count": int(count),
        }
        for class_id, count in sorted(category_counts.items())
    ]
    return {
        "schema_version": 1,
        "generated_at": utcnow().isoformat(),
        "class_names": class_names,
        "total_items": len(items),
        "categories": categories,
        "items": items,
    }


def build_view_payload_from_index(
    view_index: dict[str, Any],
    *,
    page: int,
    page_size: int,
    file_url_builder: Callable[[str], str],
    thumbnail_url_builder: Callable[[str], str],
    class_id: int | None = None,
) -> dict[str, Any]:
    raw_items = view_index.get("items") if isinstance(view_index, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []
    class_id_value = int(class_id) if class_id is not None else None
    filtered_items = (
        raw_items
        if class_id_value is None
        else [
            item
            for item in raw_items
            if isinstance(item, dict) and class_id_value in (item.get("classes") or [])
        ]
    )
    start = max(0, (int(page) - 1) * int(page_size))
    page_items = filtered_items[start : start + int(page_size)]
    items: list[dict[str, Any]] = []
    for idx, item in enumerate(page_items, start=start + 1):
        rel_path = str(item.get("path") or "")
        classes = [int(value) for value in (item.get("classes") or [])]
        items.append(
            {
                "id": int(item.get("id") or idx),
                "name": str(item.get("name") or Path(rel_path).name),
                "path": rel_path,
                "url": file_url_builder(rel_path),
                "thumbnail_url": thumbnail_url_builder(rel_path),
                "width": item.get("width"),
                "height": item.get("height"),
                "object_count": int(item.get("object_count") or 0),
                "classes": classes,
            }
        )
    categories = view_index.get("categories") if isinstance(view_index, dict) else []
    if not isinstance(categories, list):
        categories = []
    total_items = len(filtered_items)
    total_pages = math.ceil(total_items / int(page_size)) if int(page_size) else 1
    return {
        "categories": categories,
        "items": items,
        "meta": {
            "page": int(page),
            "page_size": int(page_size),
            "total_items": int(total_items),
            "total_pages": max(1, int(total_pages)),
        },
    }


def build_annotations_payload(root: Path, storage_token: str, image_rel_path: str) -> dict[str, Any]:
    class_names = yolo.read_class_names(root)
    width, height, boxes = yolo.read_yolo_boxes(root, image_rel_path, class_names)
    return {
        "image_path": str(image_rel_path),
        "image_url": static_dataset_url(storage_token, image_rel_path),
        "width": width,
        "height": height,
        "object_count": len(boxes),
        "boxes": boxes,
    }


def build_file_listing(
    root: Path,
    storage_token: str,
    *,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    files = sorted(iter_files(root), key=lambda path: path.relative_to(root).as_posix())
    total = len(files)
    start = max(0, (page - 1) * page_size)
    items: list[dict[str, Any]] = []
    for path in files[start : start + page_size]:
        rel = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
            size = int(stat.st_size)
            mtime = float(stat.st_mtime)
        except Exception:
            size = 0
            mtime = 0.0
        url = static_dataset_url(storage_token, rel) if path.suffix.lower() in IMAGE_EXTS else None
        items.append({"path": rel, "size_bytes": size, "mtime": mtime, "url": url, "exists": True})
    return items, total


def _scan_yolo_annotation_summary(root: Path) -> tuple[int, set[int]]:
    annotation_count = 0
    class_ids: set[int] = set()
    if not root.exists():
        return annotation_count, class_ids
    for label in root.rglob("*.txt"):
        if label.name.lower() in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
            continue
        try:
            lines = label.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = [part for part in line.strip().split() if part]
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            annotation_count += 1
            class_ids.add(class_id)
    return annotation_count, class_ids


def build_statistics(
    root: Path,
    *,
    image_count: int | None = None,
    total_files: int | None = None,
    total_size_bytes: int | None = None,
) -> dict[str, Any]:
    if total_files is None or total_size_bytes is None:
        counted_files, counted_size = count_tree(root)
        total_files = counted_files if total_files is None else total_files
        total_size_bytes = counted_size if total_size_bytes is None else total_size_bytes
    annotation_count, used_class_ids = _scan_yolo_annotation_summary(root)
    try:
        declared_class_count = len(yolo.read_class_names(root))
    except Exception:
        declared_class_count = 0
    class_count = declared_class_count if declared_class_count > 0 else len(used_class_ids)
    total_images = int(image_count) if image_count is not None else len(iter_image_files(root))
    size_mb = round(float(total_size_bytes or 0) / (1024 * 1024), 2)
    target_count = int(annotation_count)
    return {
        "total_files": int(total_files or 0),
        "total_size_bytes": int(total_size_bytes or 0),
        "total_size_mb": size_mb,
        "size_mb": size_mb,
        "dataset_size_mb": size_mb,
        "total_images": total_images,
        "num_images": total_images,
        "image_count": total_images,
        "annotations_count": target_count,
        "target_count": target_count,
        "total_targets": target_count,
        "object_count": target_count,
        "total_objects": target_count,
        "num_classes": int(class_count),
        "class_count": int(class_count),
        "declared_class_count": int(declared_class_count),
        "used_class_count": int(len(used_class_ids)),
    }


def refresh_statistics(db: Session, dataset: StandardDataset, *, commit: bool = False) -> dict[str, Any]:
    from .content import ensure_image_index, image_count

    root = dataset_root(dataset)
    ensure_image_index(db, dataset, commit=commit)
    stats = build_statistics(root, image_count=image_count(db, dataset))
    write_cached_statistics(root, stats)
    return stats


def get_statistics(db: Session, dataset: StandardDataset) -> dict[str, Any]:
    from .content import ensure_image_index, image_count

    root = dataset_root(dataset)
    cached = load_cached_statistics(root)
    if isinstance(cached, dict):
        cached_images = int(cached.get("num_images") or cached.get("total_images") or cached.get("image_count") or 0)
        if cached_images > 0:
            return cached
        if image_count(db, dataset) > 0 or ensure_image_index(db, dataset, commit=True):
            return refresh_statistics(db, dataset, commit=False)
        return cached
    return refresh_statistics(db, dataset, commit=True)


def refresh_view_index(db: Session, dataset: StandardDataset, *, commit: bool = False) -> dict[str, Any]:
    from .content import ensure_image_index

    root = dataset_root(dataset)
    ensure_image_index(db, dataset, commit=commit)
    image_rows = (
        db.query(StandardDatasetImage)
        .filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id))
        .order_by(StandardDatasetImage.path.asc())
        .all()
    )
    view_index = build_yolo_view_index(root, image_rows, max_workers=settings.view_index_max_workers)
    write_cached_view_index(root, view_index)
    prewarm_dataset_thumbnails(root, int(dataset.standard_dataset_id), view_index)
    return view_index


def get_view_index(db: Session, dataset: StandardDataset) -> dict[str, Any]:
    cached = load_cached_view_index(dataset_root(dataset))
    if isinstance(cached, dict) and int(cached.get("total_items") or 0) > 0:
        return cached
    return refresh_view_index(db, dataset, commit=True)


def prewarm_dataset_thumbnails(root: Path, dataset_id: int, view_index: dict[str, Any]) -> None:
    limit = max(0, int(settings.thumbnail_first_page_prewarm or 0))
    items = view_index.get("items") if isinstance(view_index, dict) else []
    if limit <= 0 or not isinstance(items, list) or not items:
        return
    entries: list[tuple[Path, str]] = []
    for item in items[:limit]:
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        try:
            source_path = resolve_dataset_file(root, rel_path)
        except Exception:
            continue
        entries.append((source_path, rel_path))
    if not entries:
        return
    try:
        prewarm_thumbnails(
            dataset_id=int(dataset_id),
            entries=entries,
            size=int(settings.thumbnail_size or 200),
            max_workers=int(settings.thumbnail_max_workers or 4),
            dataset_namespace="standard",
        )
    except Exception:
        pass


def dataset_root(dataset: StandardDataset) -> Path:
    from train_platform.domains.datasets.storage.paths import resolve_storage_token

    return resolve_storage_token(dataset.storage_path)


def dataset_base_payload(dataset: StandardDataset) -> dict[str, Any]:
    return {
        "standard_dataset_id": int(dataset.standard_dataset_id),
        "name": dataset.name,
        "dataset_type": dataset.dataset_type,
        "format": dataset.format,
        "storage_path": dataset.storage_path,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "publish_config": dataset.publish_config,
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
    }


def first_image_preview_url(
    db: Session,
    dataset: StandardDataset,
    *,
    statistics: dict[str, Any] | None = None,
) -> str | None:
    if statistics is not None and int(
        statistics.get("num_images") or statistics.get("total_images") or statistics.get("image_count") or 0
    ) <= 0:
        return None
    row = (
        db.query(StandardDatasetImage)
        .filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id))
        .order_by(StandardDatasetImage.image_id.asc())
        .first()
    )
    rel_path = str(getattr(row, "path", "") or "").strip() if row else ""
    if not rel_path:
        return None
    return dataset_thumbnail_url("standard", int(dataset.standard_dataset_id), rel_path, size=int(settings.thumbnail_size or 200))


def dataset_with_statistics(
    db: Session,
    dataset: StandardDataset,
    *,
    include_statistics: bool = True,
) -> dict[str, Any]:
    payload = dataset_base_payload(dataset)
    if not include_statistics:
        payload.update({"statistics": None, "preview_image_url": None})
        return payload
    statistics = get_statistics(db, dataset)
    payload.update(
        {
            "statistics": statistics,
            "preview_image_url": first_image_preview_url(db, dataset, statistics=statistics),
        }
    )
    return payload


def get_view(
    db: Session,
    dataset: StandardDataset,
    *,
    page: int = 1,
    page_size: int = 50,
    class_id: int | None = None,
) -> dict[str, Any]:
    view_index = get_view_index(db, dataset)
    return build_view_payload_from_index(
        view_index,
        page=page,
        page_size=page_size,
        class_id=class_id,
        file_url_builder=lambda rel_path: (
            f"/api/v3/standard-datasets/{int(dataset.standard_dataset_id)}/file/"
            f"{quote(str(rel_path).replace(chr(92), '/'), safe='/')}"
        ),
        thumbnail_url_builder=lambda rel_path: dataset_thumbnail_url(
            "standard", int(dataset.standard_dataset_id), rel_path, size=320
        ),
    )


def get_image_annotations(dataset: StandardDataset, *, image_path: str) -> dict[str, Any]:
    return build_annotations_payload(dataset_root(dataset), dataset.storage_path, image_path)


def get_file_path(dataset: StandardDataset, file_path: str) -> Path:
    return resolve_dataset_file(dataset_root(dataset), file_path)


def list_files(
    dataset: StandardDataset,
    *,
    page: int = 1,
    page_size: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    return build_file_listing(dataset_root(dataset), dataset.storage_path, page=page, page_size=page_size)


__all__ = [
    "build_annotations_payload",
    "build_file_listing",
    "build_statistics",
    "build_view_payload_from_index",
    "build_yolo_view_index",
    "dataset_thumbnail_url",
    "dataset_with_statistics",
    "first_image_preview_url",
    "get_file_path",
    "get_image_annotations",
    "get_statistics",
    "get_view",
    "list_files",
    "load_cached_statistics",
    "load_cached_view_index",
    "refresh_statistics",
    "refresh_view_index",
    "static_dataset_url",
    "utcnow",
    "write_cached_statistics",
    "write_cached_view_index",
]
