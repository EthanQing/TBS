from __future__ import annotations

import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.enums import DatasetSplit
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


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
        shutil.rmtree(path, ignore_errors=True)
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


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


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


def read_yolo_boxes(root: Path, image_rel_path: str | Path, class_names: list[str]) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    image_path = root / Path(image_rel_path)
    width, height = image_size(image_path)
    label_path = guess_label_path(root, image_rel_path)
    if not label_path.exists():
        return width, height, []
    boxes: list[dict[str, Any]] = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return width, height, []
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
    return width, height, boxes


def static_dataset_url(storage_token: str, rel_path: str | Path) -> str:
    rel = ensure_safe_relative_path(rel_path).as_posix()
    base = ensure_safe_relative_path(storage_token).as_posix()
    return f"/static/datasets/{base}/{rel}"


def build_view_payload(root: Path, storage_token: str, image_rows: list[Any], *, page: int, page_size: int) -> dict[str, Any]:
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
                "url": static_dataset_url(storage_token, rel_path),
                "thumbnail_url": static_dataset_url(storage_token, rel_path),
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


def build_statistics(root: Path, *, image_count: int | None = None) -> dict[str, Any]:
    total_files, total_size = count_tree(root)
    annotation_count = 0
    for label in root.rglob("*.txt"):
        try:
            annotation_count += len([line for line in label.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()])
        except Exception:
            continue
    total_images = int(image_count) if image_count is not None else len(iter_image_files(root))
    return {
        "total_files": int(total_files),
        "total_size_bytes": int(total_size),
        "total_size_mb": round(float(total_size) / (1024 * 1024), 2),
        "total_images": int(total_images),
        "annotations_count": int(annotation_count),
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
