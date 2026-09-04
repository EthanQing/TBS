"""LabelMe annotation primitives shared by dataset domains and importers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from train_platform.utils.exceptions import ValidationError
from train_platform.domains.datasets.images import IMAGE_EXTS


_SKIP_DIRS = {".git", "__macosx", ".thumbnails", ".versions"}


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def iter_regular_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for cur, dirnames, filenames in os.walk(root):
        cur_path = Path(cur)
        rel = cur_path.relative_to(root)
        if rel.parts and rel.parts[0].lower() in _SKIP_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for name in filenames:
            out.append(cur_path / name)
    return out


def _pair_key(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[0].lower() in {"image", "images", "annotation", "annotations", "json", "labels"}:
        parts = parts[1:]
    return "/".join(parts).lower()


def collect_image_json_pair_details(
    source_root: Path,
    *,
    skip_dirs: set[str] | None = None,
    skip_files: set[str] | None = None,
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    root = Path(source_root).resolve(strict=False)
    excluded_dirs = _SKIP_DIRS | {str(name).lower() for name in (skip_dirs or set())}
    excluded_files = {str(name).lower() for name in (skip_files or set())}
    image_by_key: dict[str, Path] = {}
    json_by_key: dict[str, Path] = {}
    for cur, dirnames, filenames in os.walk(root):
        cur_path = Path(cur)
        rel = cur_path.relative_to(root)
        if rel.parts and rel.parts[0].lower() in excluded_dirs:
            dirnames[:] = []
            continue
        dirnames[:] = [directory for directory in dirnames if directory.lower() not in excluded_dirs]
        for filename in filenames:
            if filename.lower() in excluded_files:
                continue
            path = cur_path / filename
            ext = path.suffix.lower()
            if ext in IMAGE_EXTS:
                image_by_key.setdefault(_pair_key(root, path), path)
            elif ext == ".json":
                json_by_key.setdefault(_pair_key(root, path), path)
    keys = sorted(set(image_by_key) & set(json_by_key))
    warnings: list[str] = []
    unmatched_details: list[str] = []
    extra_images = sorted(set(image_by_key) - set(json_by_key))
    extra_json = sorted(set(json_by_key) - set(image_by_key))
    if extra_images:
        warnings.append(f"Unmatched images: {len(extra_images)}")
        unmatched_details.extend(f"{image_by_key[key].name}: missing json annotation" for key in extra_images[:50])
    if extra_json:
        warnings.append(f"Unmatched json: {len(extra_json)}")
        unmatched_details.extend(f"{json_by_key[key].name}: missing image file" for key in extra_json[:50])
    if unmatched_details:
        warnings.extend(f"Skipped {item}" for item in unmatched_details[:50])
    return [(image_by_key[key], json_by_key[key]) for key in keys], warnings, unmatched_details


def collect_image_json_pairs(source_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    pairs, warnings, _details = collect_image_json_pair_details(source_root)
    return pairs, [warning for warning in warnings if not warning.startswith("Skipped ")]


def choose_image_link(source_root: Path) -> tuple[Path, str]:
    source_root = Path(source_root).resolve(strict=False)
    images_dir = source_root / "images"
    if images_dir.exists() and images_dir.is_dir():
        return images_dir.resolve(strict=False), "images"
    return source_root, "images/source"


def image_rel_for_source(source_root: Path, source_image_root: Path, image_path: Path, image_rel_prefix: str) -> str:
    rel = Path(image_path).resolve(strict=False).relative_to(Path(source_image_root).resolve(strict=False)).as_posix()
    return f"{image_rel_prefix.strip('/')}/{rel}".strip("/")


def label_rel_for_image(image_rel: str | Path) -> str:
    rel = Path(str(image_rel).replace("\\", "/"))
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt").as_posix()
    return rel.with_suffix(".txt").as_posix()


def points_to_bbox(points: list, label: str) -> BBox:
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return BBox(float(x_min), float(y_min), float(x_max), float(y_max), label)


def extract_label(raw_label: str, strategy, separator: str = "%") -> str:
    parts = [p.strip() for p in raw_label.split(separator) if p.strip()]
    if not parts:
        return ""
    if strategy == "full":
        return separator.join(parts)
    if strategy == "leaf":
        return parts[-1]
    if strategy == "root":
        return parts[0]
    if isinstance(strategy, int):
        return separator.join(parts[:strategy])
    return raw_label


def normalize_label_key(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\uFF05", "%").replace("\u3000", " ")
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(ch, "")
    return s.strip()


def _uses_bottom_left_origin(version: Any) -> bool:
    if isinstance(version, bool):
        return False
    if isinstance(version, Real):
        return float(version) == 1.0
    if isinstance(version, str):
        return version.strip() == "1"
    return False


def _coerce_dimension(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _annotation_image_height(cfg: dict, data: Any, json_path: str) -> float:
    candidates = [cfg.get("image_height")]
    if isinstance(data, dict):
        candidates.append(data.get("imageHeight"))
    for candidate in candidates:
        height = _coerce_dimension(candidate)
        if height is not None:
            return height
    raise ValidationError(f"Missing image height for bottom-left origin annotation: {json_path}")


def _normalize_annotation_points(points: list, *, bottom_left_origin: bool, image_height: Optional[float]) -> list:
    if not bottom_left_origin:
        return points
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            normalized.append(point)
            continue
        x, y = point[0], point[1]
        normalized.append([x, float(image_height) - float(y), *list(point[2:])])
    return normalized


def _load_annotation_data(json_path: str) -> Any:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(json_path, "r", encoding="gbk", errors="ignore") as f:
            return json.load(f)


def resolve_annotation_shapes(
    cfg: dict,
    *,
    data: Any = None,
) -> list[tuple[dict[str, Any], str]]:
    """Return annotation shapes after applying shared LabelMe filtering and mapping rules."""
    json_path = cfg["annotation_path"]
    label_strategy = cfg["label_strategy"]
    label_sep = cfg["label_separator"]
    min_prob = cfg["min_probability"]
    skip_hidden = cfg["skip_hidden"]
    skip_outside = cfg["skip_outside"]
    label_strategy_norm = str(label_strategy or "").strip().lower()

    raw_mapping = cfg.get("label_mapping")
    label_mapping: Optional[Dict[str, str]] = None
    label_mapping_norm: Optional[Dict[str, str]] = None
    if isinstance(raw_mapping, dict) and raw_mapping:
        label_mapping = {str(k): v for k, v in raw_mapping.items()}
        label_mapping_norm = {}
        for k, v in label_mapping.items():
            nk = normalize_label_key(k)
            if nk in label_mapping_norm and label_mapping_norm[nk] != v:
                raise ValidationError(f"Conflicting label mappings for normalized key: {nk}")
            label_mapping_norm[nk] = v

    def mapping_value_is_discard(value: Any) -> bool:
        return str(value if value is not None else "").strip() in {"", "__DISCARD__"}

    def lookup_mapping_value(label: str) -> tuple[Any, bool]:
        if label_mapping is None:
            return None, False
        if label in label_mapping:
            return label_mapping.get(label), True
        if label_mapping_norm is not None:
            norm_key = normalize_label_key(label)
            if norm_key in label_mapping_norm:
                return label_mapping_norm.get(norm_key), True
        return None, False

    def iter_parent_labels(label: str):
        if not label_sep:
            return
        seen: set[str] = set()
        for candidate in (str(label or "").strip(), normalize_label_key(label)):
            parts = [part.strip() for part in str(candidate).split(label_sep) if part.strip()]
            for index in range(1, len(parts)):
                parent = label_sep.join(parts[:index])
                if parent and parent not in seen:
                    seen.add(parent)
                    yield parent

    def has_discarded_parent(label: str) -> bool:
        if label_mapping is None:
            return False
        for parent in iter_parent_labels(label):
            value, found = lookup_mapping_value(parent)
            if found and mapping_value_is_discard(value):
                return True
        return False

    if data is None:
        data = _load_annotation_data(json_path)
    if isinstance(data, dict) and "shapes" in data:
        shapes = data["shapes"]
    elif isinstance(data, list):
        shapes = data
    else:
        raise ValidationError(f"Unrecognized json structure: {json_path}")

    missing_labels: set[str] = set()
    valid: list[tuple[dict[str, Any], str]] = []
    for shape in shapes:
        if skip_hidden and shape.get("hidden", False):
            continue
        if skip_outside and shape.get("outside", False):
            continue
        if shape.get("probability", 1.0) < min_prob:
            continue
        points = shape.get("points", [])
        if not points or len(points) < 2:
            continue

        raw_label_stripped = str(shape.get("label", "unknown")).strip()
        if not raw_label_stripped:
            continue
        if label_mapping is not None:
            mapped_label, matched_mapping = lookup_mapping_value(raw_label_stripped)
            if matched_mapping and mapping_value_is_discard(mapped_label):
                continue
            if has_discarded_parent(raw_label_stripped):
                continue
            if matched_mapping:
                raw_label_stripped = str(mapped_label).strip()
            elif label_strategy_norm == "mapping":
                missing_labels.add(raw_label_stripped)
                continue

        label_name = extract_label(raw_label_stripped, label_strategy, label_sep).strip()
        if label_name:
            valid.append((shape, label_name))

    if missing_labels:
        sample = ", ".join(list(missing_labels)[:10])
        raise ValidationError(
            f"Missing label mappings for {len(missing_labels)} labels in {json_path}: {sample}"
        )
    return valid


def parse_annotations(cfg: dict) -> Tuple[List[BBox], Dict[str, int]]:
    json_path = cfg["annotation_path"]
    label_map = cfg["label_map"]
    data = _load_annotation_data(json_path)
    valid_shapes = resolve_annotation_shapes(cfg, data=data)

    auto_map: Dict[str, int] = label_map.copy() if label_map else {}

    def get_cid(name: str) -> int:
        if name not in auto_map:
            auto_map[name] = len(auto_map)
        return auto_map[name]

    bottom_left_origin = isinstance(data, dict) and _uses_bottom_left_origin(data.get("version"))
    image_height = _annotation_image_height(cfg, data, json_path) if bottom_left_origin else None

    bboxes: List[BBox] = []
    for shape, label_name in valid_shapes:
        pts = _normalize_annotation_points(
            shape.get("points", []),
            bottom_left_origin=bottom_left_origin,
            image_height=image_height,
        )
        stype = str(shape.get("shape_type", "polygon")).lower()

        if stype == "rectangle" and len(pts) == 2:
            x_min = min(pts[0][0], pts[1][0])
            y_min = min(pts[0][1], pts[1][1])
            x_max = max(pts[0][0], pts[1][0])
            y_max = max(pts[0][1], pts[1][1])
            bbox = BBox(x_min, y_min, x_max, y_max, label_name)
        elif stype == "circle" and len(pts) >= 2:
            cx, cy = pts[0]
            ex, ey = pts[1]
            r = math.hypot(cx - ex, cy - ey)
            bbox = BBox(cx - r, cy - r, cx + r, cy + r, label_name)
        else:
            bbox = points_to_bbox(pts, label_name)

        if bbox.width <= 0 or bbox.height <= 0:
            continue
        bbox.class_id = get_cid(label_name)
        bboxes.append(bbox)
    return bboxes, auto_map


def bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = np.clip((bbox.x_min + bbox.x_max) / 2.0 / img_w, 0, 1)
    cy = np.clip((bbox.y_min + bbox.y_max) / 2.0 / img_h, 0, 1)
    w = np.clip(bbox.width / img_w, 0, 1)
    h = np.clip(bbox.height / img_h, 0, 1)
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
