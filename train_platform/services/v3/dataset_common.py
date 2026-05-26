from __future__ import annotations

import json
import math
import os
import shutil
import stat
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import quote, urlencode

import yaml
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.enums import DatasetSplit
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS
from train_platform.utils.zip_encoding import safe_zip_member_relpath

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


_DATASET_INTERNAL_FILE_NAMES = {".dataset_stats.json", ".dataset_view_index.json"}
_JSON_CACHE_LOCK = threading.Lock()
_JSON_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_safe_relative_path(value: str | Path | None) -> Path:
    rel = Path(str(value or "").strip().replace("\\", "/"))
    if not str(rel) or rel.is_absolute() or ".." in rel.parts:
        raise ValidationError("Invalid relative path")
    return rel


def to_storage_token(path: Path) -> str:
    return path.resolve(strict=False).relative_to(settings.datasets_dir.resolve()).as_posix()


def resolve_storage_token(token: str | Path) -> Path:
    rel = ensure_safe_relative_path(token)
    path = (settings.datasets_dir / rel).resolve(strict=False)
    try:
        path.relative_to(settings.datasets_dir.resolve())
    except Exception as exc:  # pragma: no cover
        raise ValidationError("Unsafe dataset path") from exc
    return path


def clear_directory(path: Path) -> None:
    if path.exists():
        def _onerror(func, raw_path, _exc_info):
            try:
                os.chmod(raw_path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            except Exception:
                pass
            func(raw_path)

        shutil.rmtree(path, onerror=_onerror)
    path.mkdir(parents=True, exist_ok=True)


def overlay_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            overlay_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def copy_tree(src: Path, dst: Path) -> None:
    clear_directory(dst)
    overlay_tree(src, dst)


def save_upload_to_temp(upload, suffix: str | None = None) -> Path:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix or "", dir=settings.temp_dir)
    os.close(fd)
    out = Path(name)
    with out.open("wb") as f:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, f)
    upload.file.seek(0)
    return out


def unpack_uploaded_archive(upload, destination: Path) -> Path:
    filename = str(getattr(upload, "filename", "") or "archive.zip")
    suffix = Path(filename).suffix or ".zip"
    temp_archive = save_upload_to_temp(upload, suffix=suffix)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if suffix.lower() == ".zip":
            return safe_extract_zip(temp_archive, destination)
        shutil.unpack_archive(str(temp_archive), str(destination))
    except Exception as exc:
        raise ValidationError(f"Unsupported or invalid archive: {exc}") from exc
    finally:
        temp_archive.unlink(missing_ok=True)

    children = [p for p in destination.iterdir() if p.name not in ("__MACOSX",)]
    files = [p for p in children if p.is_file()]
    dirs = [p for p in children if p.is_dir()]
    if not files and len(dirs) == 1:
        return dirs[0]
    return destination


def safe_extract_zip(archive_path: Path, destination: Path) -> Path:
    archive = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve(strict=False)
    try:
        with zipfile.ZipFile(archive) as zf:
            members: list[tuple[zipfile.ZipInfo, Path]] = []
            seen: set[str] = set()
            for info in zf.infolist():
                rel = safe_zip_member_relpath(info)
                key = rel.as_posix().lower() if os.name == "nt" else rel.as_posix()
                if key in seen and not info.is_dir():
                    raise ValidationError("ZIP contains duplicate file paths")
                seen.add(key)
                target = (dest_root / rel).resolve(strict=False)
                try:
                    target.relative_to(dest_root)
                except Exception as exc:
                    raise ValidationError("ZIP contains a path outside destination") from exc
                members.append((info, rel))

            for info, rel in members:
                target = (dest_root / rel).resolve(strict=False)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Unsupported or invalid ZIP archive: {exc}") from exc

    children = [p for p in destination.iterdir() if p.name not in ("__MACOSX",)]
    files = [p for p in children if p.is_file()]
    dirs = [p for p in children if p.is_dir()]
    if not files and len(dirs) == 1:
        return dirs[0]
    return destination


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.name not in _DATASET_INTERNAL_FILE_NAMES)


def iter_image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def count_tree(root: Path) -> tuple[int, int]:
    total_files = 0
    total_size = 0
    for path in iter_files(root):
        total_files += 1
        try:
            total_size += int(path.stat().st_size)
        except Exception:
            pass
    return total_files, total_size


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[str(path.resolve(strict=False))] = (mtime, data)
    return path


def load_cached_json_file(path: Path) -> dict[str, Any] | None:
    return _load_cached_json(Path(path))


def write_cached_json_file(path: Path, data: dict[str, Any]) -> Path:
    return _write_cached_json(Path(path), data)


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


def maybe_find_data_yaml(root: Path) -> Path | None:
    candidates = [root / "data.yaml", root / "dataset.yaml"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in root.rglob("*.yaml"):
        if candidate.name.lower() in {"data.yaml", "dataset.yaml"}:
            return candidate
    return None


def read_data_yaml(root: Path) -> dict[str, Any]:
    path = maybe_find_data_yaml(root)
    if not path:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_class_names(root: Path) -> list[str]:
    data = read_data_yaml(root)
    names = data.get("names")
    if isinstance(names, dict):
        out: list[str] = []
        for _, value in sorted(names.items(), key=lambda item: int(item[0])):
            out.append(str(value))
        return out
    if isinstance(names, list):
        return [str(x) for x in names]
    classes_txt = root / "classes.txt"
    if classes_txt.exists():
        try:
            return [line.strip() for line in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        except Exception:
            return []
    return []


def detect_split_from_relpath(rel_path: str | Path) -> DatasetSplit | None:
    parts = Path(rel_path).parts
    lowered = [part.lower() for part in parts]
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        if split.value in lowered:
            return split
    return None


def image_size(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def guess_label_path(root: Path, image_rel_path: str | Path) -> Path:
    rel = Path(image_rel_path)
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return root / Path(*parts).with_suffix(".txt")
    return root / rel.with_suffix(".txt")


def _parse_yolo_boxes(
    lines: list[str],
    class_names: list[str],
    *,
    width: int | None,
    height: int | None,
    include_boxes: bool,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    boxes: list[dict[str, Any]] = []
    class_ids: set[int] = set()
    object_count = 0
    for line in lines:
        parts = [p for p in line.strip().split() if p]
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            continue
        object_count += 1
        class_ids.add(class_id)
        if not include_boxes:
            continue
        if width and height:
            x1 = max(0.0, (xc - w / 2.0) * width)
            y1 = max(0.0, (yc - h / 2.0) * height)
            x2 = min(float(width), (xc + w / 2.0) * width)
            y2 = min(float(height), (yc + h / 2.0) * height)
        else:
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "x1": float(round(x1, 4)),
                "y1": float(round(y1, 4)),
                "x2": float(round(x2, 4)),
                "y2": float(round(y2, 4)),
            }
        )
    return boxes, object_count, sorted(class_ids)


def read_yolo_boxes(root: Path, image_rel_path: str | Path, class_names: list[str]) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    image_path = root / Path(image_rel_path)
    width, height = image_size(image_path)
    label_path = guess_label_path(root, image_rel_path)
    if not label_path.exists():
        return width, height, []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return width, height, []
    boxes, _object_count, _class_ids = _parse_yolo_boxes(
        lines,
        class_names,
        width=width,
        height=height,
        include_boxes=True,
    )
    return width, height, boxes


def read_yolo_box_summary(root: Path, image_rel_path: str | Path, class_names: list[str]) -> tuple[int | None, int | None, int, list[int]]:
    image_path = root / Path(image_rel_path)
    width, height = image_size(image_path)
    label_path = guess_label_path(root, image_rel_path)
    if not label_path.exists():
        return width, height, 0, []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return width, height, 0, []
    _boxes, object_count, class_ids = _parse_yolo_boxes(
        lines,
        class_names,
        width=width,
        height=height,
        include_boxes=False,
    )
    return width, height, object_count, class_ids


def static_dataset_url(storage_token: str, rel_path: str | Path) -> str:
    rel = ensure_safe_relative_path(rel_path).as_posix()
    base = ensure_safe_relative_path(storage_token).as_posix()
    return f"/static/datasets/{base}/{rel}"


def dataset_thumbnail_url(
    dataset_kind: str,
    dataset_id: int,
    rel_path: str | Path,
    *,
    version_id: int | None = None,
    size: int | None = None,
) -> str:
    rel = ensure_safe_relative_path(rel_path).as_posix()
    encoded_rel = quote(rel, safe="/")
    base = f"/api/v3/thumbnails/{quote(str(dataset_kind).strip(), safe='')}/{int(dataset_id)}/{encoded_rel}"
    params: dict[str, str] = {}
    if version_id is not None:
        params["version_id"] = str(int(version_id))
    if size is not None:
        params["size"] = str(int(size))
    qs = urlencode(params)
    return f"{base}?{qs}" if qs else base


def build_view_payload(
    root: Path,
    storage_token: str,
    image_rows: list[Any],
    *,
    page: int,
    page_size: int,
    thumbnail_url_builder: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    class_names = read_class_names(root)
    category_image_ids: dict[int, set[int]] = {}
    items: list[dict[str, Any]] = []
    total_items = len(image_rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_rows = image_rows[start:end]
    for idx, row in enumerate(image_rows, start=1):
        rel_path = str(getattr(row, "path", "") or "")
        _w, _h, boxes = read_yolo_boxes(root, rel_path, class_names)
        image_id = int(getattr(row, "image_id", idx) or idx)
        for box in boxes:
            category_image_ids.setdefault(int(box["class_id"]), set()).add(image_id)
    for idx, row in enumerate(page_rows, start=1):
        rel_path = str(getattr(row, "path", "") or "")
        width, height, boxes = read_yolo_boxes(root, rel_path, class_names)
        classes = sorted({int(box["class_id"]) for box in boxes})
        items.append(
            {
                "id": int(getattr(row, "image_id", idx) or idx),
                "name": Path(rel_path).name,
                "path": rel_path,
                "url": static_dataset_url(storage_token, rel_path),
                "thumbnail_url": thumbnail_url_builder(rel_path) if thumbnail_url_builder else static_dataset_url(storage_token, rel_path),
                "width": width,
                "height": height,
                "object_count": len(boxes),
                "classes": classes,
            }
        )

    categories = [
        {
            "class_id": class_id,
            "name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
            "count": len(image_ids),
        }
        for class_id, image_ids in sorted(category_image_ids.items())
    ]
    total_pages = math.ceil(total_items / page_size) if page_size else 1
    return {
        "categories": categories,
        "items": items,
        "meta": {
            "page": int(page),
            "page_size": int(page_size),
            "total_items": int(total_items),
            "total_pages": int(total_pages or 1),
        },
    }


def build_yolo_view_index(root: Path, image_rows: list[Any], *, max_workers: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve(strict=False)
    class_names = read_class_names(root)
    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(image_rows, start=1):
        if isinstance(row, dict):
            rel_path = str(row.get("path") or "")
            item_id = int(row.get("id") or row.get("image_id") or idx)
        else:
            rel_path = str(getattr(row, "path", "") or "")
            item_id = int(getattr(row, "image_id", idx) or idx)
        if not rel_path:
            continue
        entries.append(
            {
                "id": item_id,
                "path": rel_path,
                "name": Path(rel_path).name,
            }
        )

    def _process(entry: dict[str, Any]) -> dict[str, Any]:
        width, height, object_count, classes = read_yolo_box_summary(root, entry["path"], class_names)
        return {
            **entry,
            "width": width,
            "height": height,
            "object_count": int(object_count),
            "classes": [int(x) for x in classes],
        }

    workers = max(1, int(max_workers or settings.view_index_max_workers or 1))
    if len(entries) <= 1 or workers <= 1:
        items = [_process(entry) for entry in entries]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(entries)))) as executor:
            items = list(executor.map(_process, entries))

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
    if class_id_value is None:
        filtered_items = raw_items
    else:
        filtered_items = [
            item
            for item in raw_items
            if isinstance(item, dict) and class_id_value in (item.get("classes") or [])
        ]

    start = max(0, (int(page) - 1) * int(page_size))
    end = start + int(page_size)
    page_items = filtered_items[start:end]

    items: list[dict[str, Any]] = []
    for idx, item in enumerate(page_items, start=start + 1):
        rel_path = str(item.get("path") or "")
        classes = [int(x) for x in (item.get("classes") or [])]
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
    if total_pages <= 0:
        total_pages = 1
    return {
        "categories": categories,
        "items": items,
        "meta": {
            "page": int(page),
            "page_size": int(page_size),
            "total_items": int(total_items),
            "total_pages": int(total_pages),
        },
    }


def build_annotations_payload(root: Path, storage_token: str, image_rel_path: str) -> dict[str, Any]:
    class_names = read_class_names(root)
    width, height, boxes = read_yolo_boxes(root, image_rel_path, class_names)
    return {
        "image_path": str(image_rel_path),
        "image_url": static_dataset_url(storage_token, image_rel_path),
        "width": width,
        "height": height,
        "object_count": len(boxes),
        "boxes": boxes,
    }


def build_file_listing(root: Path, storage_token: str, *, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
    files = sorted(iter_files(root), key=lambda p: p.relative_to(root).as_posix())
    total = len(files)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    items: list[dict[str, Any]] = []
    for path in files[start:end]:
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
    """Return (object/target count, used class ids) from YOLO label files."""
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
            parts = [p for p in line.strip().split() if p]
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
        if total_files is None:
            total_files = counted_files
        if total_size_bytes is None:
            total_size_bytes = counted_size

    annotation_count, used_class_ids = _scan_yolo_annotation_summary(root)
    try:
        declared_class_count = len(read_class_names(root))
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
        "total_images": int(total_images),
        "num_images": int(total_images),
        "image_count": int(total_images),
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


def upsert_yaml_path(root: Path) -> str | None:
    data_yaml = maybe_find_data_yaml(root)
    if not data_yaml:
        return None
    return data_yaml.relative_to(settings.datasets_dir.resolve()).as_posix() if data_yaml.is_absolute() else data_yaml.as_posix()


def commit_refresh(db: Session, row: Any) -> Any:
    db.commit()
    db.refresh(row)
    return row
