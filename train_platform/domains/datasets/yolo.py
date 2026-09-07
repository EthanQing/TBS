from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from train_platform.models.v3.enums import DatasetSplit, DatasetType
from train_platform.platform.filesystem.atomic import atomic_write_text
from train_platform.utils.exceptions import ValidationError
from train_platform.domains.datasets.images import IMAGE_EXTS

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


CLASS_FILE_NAMES = ("class_names.txt", "classes.txt", "obj.names", "names.txt", "classnames.txt")

_DEFAULT_DATASET_YAML_FILENAMES: tuple[str, ...] = (
    "data.yaml",
    "data.yml",
    "dataset.yaml",
    "dataset.yml",
)


def _safe_stem(name: str | None) -> str | None:
    """
    Best-effort conversion of a dataset name into a safe filename stem.
    """
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    # Avoid path traversal / separators; keep only the last path segment.
    s = Path(s.replace("\\", "/")).name
    if not s or s in (".", ".."):
        return None
    return s


def _load_yaml_dict(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def _looks_like_yolo_dataset_yaml(cfg: Any) -> bool:
    """
    Heuristic to avoid picking random .yml files (e.g. docker-compose.yml) as dataset configs.
    """
    if not isinstance(cfg, dict):
        return False
    # Typical YOLO dataset keys
    if any(k in cfg for k in ("train", "val", "test")):
        return True
    if "names" in cfg or "nc" in cfg:
        return True
    if "path" in cfg and ("train" in cfg or "val" in cfg):
        return True
    return False


def find_yolo_dataset_yaml(dataset_dir: Path, *, dataset_name: str | None = None) -> Path | None:
    """
    Locate a YOLO-style dataset YAML inside `dataset_dir`.

    We prefer stable/expected names first ("data.yaml"), but many public datasets ship
    configs named after the dataset itself (e.g. "HomeObjects-3K.yaml").
    """
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        return None

    candidates: list[str] = list(_DEFAULT_DATASET_YAML_FILENAMES)
    stem = _safe_stem(dataset_name)
    if stem:
        candidates.extend([f"{stem}.yaml", f"{stem}.yml"])

    for fname in candidates:
        p = (root / fname).resolve(strict=False)
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue

    yaml_files = []
    try:
        yaml_files.extend(root.glob("*.yaml"))
        yaml_files.extend(root.glob("*.yml"))
    except Exception:
        yaml_files = []

    yaml_files = [p for p in yaml_files if p.exists() and p.is_file()]
    if not yaml_files:
        return None

    if len(yaml_files) == 1:
        return yaml_files[0].resolve(strict=False)

    # Content-based selection.
    best: Path | None = None
    for p in sorted(yaml_files, key=lambda x: x.name.lower()):
        cfg = _load_yaml_dict(p)
        # Strong signal: train+val present.
        if cfg.get("train") is not None and cfg.get("val") is not None:
            return p.resolve(strict=False)
        if best is None and _looks_like_yolo_dataset_yaml(cfg):
            best = p

    return best.resolve(strict=False) if best else None


def maybe_find_data_yaml(root: Path) -> Path | None:
    candidates = [Path(root) / "data.yaml", Path(root) / "dataset.yaml"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in Path(root).rglob("*.yaml"):
        if candidate.name.lower() in {"data.yaml", "dataset.yaml"}:
            return candidate
    return None


def read_data_yaml(root: Path) -> dict[str, Any]:
    path = maybe_find_data_yaml(Path(root))
    if not path:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_class_names(root: Path) -> list[str]:
    data = read_data_yaml(Path(root))
    names = data.get("names")
    if isinstance(names, dict):
        try:
            return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError):
            return [str(value) for value in names.values()]
    if isinstance(names, list):
        return [str(value) for value in names]
    classes_txt = Path(root) / "classes.txt"
    if classes_txt.exists() and classes_txt.is_file():
        return _read_lines(classes_txt)
    return []


def read_export_class_names(dataset_dir: Path) -> list[str]:
    root = Path(dataset_dir)
    for name in CLASS_FILE_NAMES:
        path = root / name
        if path.exists() and path.is_file():
            return _read_lines(path)

    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth > 3:
            directories[:] = []
            continue
        directories[:] = [item for item in directories if item.lower() not in {"images", "labels", ".versions"}]
        for name in CLASS_FILE_NAMES:
            if name in filenames:
                path = current_path / name
                if path.exists() and path.is_file():
                    return _read_lines(path)
    return []


def detect_split_from_relpath(rel_path: str | Path) -> DatasetSplit | None:
    lowered = [part.lower() for part in Path(rel_path).parts]
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        if split.value in lowered:
            return split
    return None


def image_size(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def guess_label_path(root: Path, image_rel_path: str | Path) -> Path:
    rel = Path(image_rel_path)
    parts = list(rel.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        parts[lowered.index("images")] = "labels"
        return Path(root) / Path(*parts).with_suffix(".txt")
    return Path(root) / rel.with_suffix(".txt")


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
        parts = [part for part in line.strip().split() if part]
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, box_width, box_height = (float(part) for part in parts[1:5])
        except (TypeError, ValueError):
            continue
        object_count += 1
        class_ids.add(class_id)
        if not include_boxes:
            continue
        if width and height:
            x1 = max(0.0, (xc - box_width / 2.0) * width)
            y1 = max(0.0, (yc - box_height / 2.0) * height)
            x2 = min(float(width), (xc + box_width / 2.0) * width)
            y2 = min(float(height), (yc + box_height / 2.0) * height)
        else:
            x1 = xc - box_width / 2.0
            y1 = yc - box_height / 2.0
            x2 = xc + box_width / 2.0
            y2 = yc + box_height / 2.0
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
    image_path = Path(root) / Path(image_rel_path)
    width, height = image_size(image_path)
    label_path = guess_label_path(Path(root), image_rel_path)
    if not label_path.exists():
        return width, height, []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return width, height, []
    boxes, _count, _classes = _parse_yolo_boxes(lines, class_names, width=width, height=height, include_boxes=True)
    return width, height, boxes


def read_yolo_box_summary(root: Path, image_rel_path: str | Path, class_names: list[str]) -> tuple[int | None, int | None, int, list[int]]:
    image_path = Path(root) / Path(image_rel_path)
    width, height = image_size(image_path)
    label_path = guess_label_path(Path(root), image_rel_path)
    if not label_path.exists():
        return width, height, 0, []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return width, height, 0, []
    _boxes, count, classes = _parse_yolo_boxes(lines, class_names, width=width, height=height, include_boxes=False)
    return width, height, count, classes


def find_yolo_export_root(root: Path) -> Optional[Path]:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return None
    best: tuple[int, int, Path] | None = None
    for current, directories, _filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth > 6:
            directories[:] = []
            continue
        by_lower = {directory.lower(): directory for directory in directories}
        if "images" not in by_lower or "labels" not in by_lower:
            continue
        images_dir = current_path / by_lower["images"]
        labels_dir = current_path / by_lower["labels"]
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        if not any(path.suffix.lower() == ".txt" for path in labels_dir.rglob("*")):
            continue
        image_count = 0
        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                image_count += 1
                if image_count >= 10:
                    break
        if image_count <= 0:
            continue
        candidate = (image_count, depth, current_path)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    return best[2] if best else None


def classify_json(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return "unknown"
        if int(path.stat().st_size or 0) > 20 * 1024 * 1024:
            head = path.read_text(encoding="utf-8", errors="ignore")[: 256 * 1024].lower()
            return "labelme" if '"shapes"' in head and any(key in head for key in ('"imagewidth"', '"imageheight"', '"imagepath"')) else "unknown"
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, dict) and "shapes" in data and any(key in data for key in ("imageWidth", "imageHeight", "imagePath", "imagewidth", "imageheight")):
            return "labelme"
    except Exception:
        return "unknown"
    return "unknown"


def detect_dataset_format(root: Path, dataset_type: DatasetType) -> dict[str, Any]:
    result: dict[str, Any] = {"format": "no_images", "yolo_root": None, "labelme_json": None}
    if dataset_type != DatasetType.DETECTION:
        result["format"] = "yolo"
        return result
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return result
    yolo_root = find_yolo_export_root(root)
    if yolo_root is not None:
        result.update(format="yolo", yolo_root=yolo_root)
        return result
    has_images = False
    json_paths: list[Path] = []
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        rel = current_path.relative_to(root)
        if len(rel.parts) > 4:
            directories[:] = []
            continue
        directories[:] = [directory for directory in directories if directory.lower() not in {".versions", ".thumbnails", "__macosx"}]
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() in IMAGE_EXTS:
                has_images = True
            elif path.suffix.lower() == ".json":
                json_paths.append(path)
    if not has_images:
        return result
    if json_paths:
        for path in json_paths:
            if classify_json(path) == "labelme":
                result.update(format="labelme", labelme_json=path)
                return result
        if any(classify_json(path) == "unknown" for path in json_paths):
            result["format"] = "unknown_json"
            return result
    result["format"] = "images_only"
    return result


def validate_dataset_structure(dataset_dir: Path, dataset_type: DatasetType) -> None:
    if dataset_type != DatasetType.DETECTION:
        return
    has_images = False
    has_labels = False
    has_json = False
    for current, _directories, filenames in os.walk(Path(dataset_dir)):
        current_path = Path(current)
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix in IMAGE_EXTS:
                has_images = True
            elif suffix == ".txt" and "labels" in {part.lower() for part in current_path.parts}:
                has_labels = True
            elif suffix == ".json":
                has_json = True
        if has_images and (has_labels or has_json):
            break
    if not has_images:
        raise ValidationError("No image files found in dataset directory")
    if not has_labels and not has_json:
        raise ValidationError("No label files found")
    if not any(path.is_file() for path in list(Path(dataset_dir).glob("*.yaml")) + list(Path(dataset_dir).glob("*.yml"))):
        create_yolo_data_yaml(Path(dataset_dir), Path(dataset_dir) / "data.yaml")


def create_yolo_data_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    root = Path(dataset_dir)
    train_path: str | None = None
    val_path: str | None = None
    structures = [
        {"train": root / "images" / "train", "val": root / "images" / "val"},
        {"train": root / "train" / "images", "val": root / "val" / "images"},
        {"train": root / "images", "val": root / "images"},
    ]
    for structure in structures:
        if structure["train"].exists():
            train_path = str(structure["train"].relative_to(root))
            break
    for structure in structures:
        if structure["val"].exists():
            val_path = str(structure["val"].relative_to(root))
            break
    if not val_path and train_path:
        val_path = train_path
    class_names = read_export_class_names(root)
    labels_dir = root / "labels"
    if not class_names and labels_dir.exists():
        for label_file in labels_dir.rglob("*.txt"):
            try:
                for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip():
                        class_id = int(line.split()[0])
                        while len(class_names) <= class_id:
                            class_names.append(f"class_{len(class_names)}")
            except (OSError, ValueError, IndexError):
                continue
    if not class_names:
        class_names = ["class_0"]
    payload = {"train": train_path or "images", "val": val_path or "images", "nc": len(class_names), "names": class_names}
    atomic_write_text(Path(yaml_path), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def merge_classes(dataset_dir: Path, source_root: Path) -> dict[str, Any]:
    existing: list[str] = []
    yaml_path = find_yolo_dataset_yaml(Path(dataset_dir))
    if yaml_path is not None:
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
            names = config.get("names", [])
            if isinstance(names, list):
                existing = [str(name) for name in names]
            elif isinstance(names, dict):
                existing = [str(names.get(index, f"class_{index}")) for index in range(max((int(key) for key in names), default=-1) + 1)]
        except Exception:
            existing = []
    incoming = read_export_class_names(Path(source_root))
    if not incoming:
        return {"added_classes": [], "total_classes": len(existing)}
    if not existing:
        update_data_yaml_classes(Path(dataset_dir), incoming)
        return {"added_classes": incoming, "total_classes": len(incoming)}
    if len(incoming) < len(existing):
        raise ValidationError(f"类别不兼容：上传的压缩包包含 {len(incoming)} 个类别，但数据集已有 {len(existing)} 个类别。无法减少类别数量。")
    for index, current in enumerate(existing):
        if incoming[index] != current:
            raise ValidationError(f"类别不兼容：第 {index + 1} 个类别不匹配。现有: '{current}', 上传: '{incoming[index]}'。请确保上传的压缩包中 classnames.txt 的前 {len(existing)} 个类别与数据集现有类别完全一致。")
    added = incoming[len(existing) :]
    if added:
        update_data_yaml_classes(Path(dataset_dir), incoming)
    return {"added_classes": added, "total_classes": len(incoming)}


def update_data_yaml_classes(dataset_dir: Path, class_names: list[str]) -> None:
    path = Path(dataset_dir) / "data.yaml"
    config: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
            if isinstance(loaded, dict):
                config = loaded
        except Exception:
            pass
    config.update(names=class_names, nc=len(class_names))
    config.setdefault("train", "images")
    config.setdefault("val", "images")
    atomic_write_text(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def _read_lines(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        try:
            content = path.read_text(encoding="gbk", errors="ignore")
        except OSError:
            return []
    return [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]


__all__ = [
    "CLASS_FILE_NAMES",
    "classify_json",
    "create_yolo_data_yaml",
    "detect_dataset_format",
    "detect_split_from_relpath",
    "find_yolo_dataset_yaml",
    "find_yolo_export_root",
    "guess_label_path",
    "image_size",
    "maybe_find_data_yaml",
    "merge_classes",
    "read_class_names",
    "read_data_yaml",
    "read_export_class_names",
    "read_yolo_box_summary",
    "read_yolo_boxes",
    "update_data_yaml_classes",
    "validate_dataset_structure",
]
