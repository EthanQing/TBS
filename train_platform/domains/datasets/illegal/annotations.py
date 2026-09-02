from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from train_platform.utils.exceptions import ValidationError


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


def _normalize_label_key(value: Any) -> str:
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


def parse_annotations(cfg: dict) -> Tuple[List[BBox], Dict[str, int]]:
    json_path = cfg["annotation_path"]
    label_map = cfg["label_map"]
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
            nk = _normalize_label_key(k)
            if nk in label_mapping_norm and label_mapping_norm[nk] != v:
                raise ValidationError(f"Conflicting label mappings for normalized key: {nk}")
            label_mapping_norm[nk] = v
    missing_labels: set[str] = set()

    def mapping_value_is_discard(value: Any) -> bool:
        return str(value if value is not None else "").strip() in {"", "__DISCARD__"}

    def lookup_mapping_value(label: str) -> tuple[Any, bool]:
        if label_mapping is None:
            return None, False
        if label in label_mapping:
            return label_mapping.get(label), True
        if label_mapping_norm is not None:
            norm_key = _normalize_label_key(label)
            if norm_key in label_mapping_norm:
                return label_mapping_norm.get(norm_key), True
        return None, False

    def iter_parent_labels(label: str):
        if not label_sep:
            return
        seen: set[str] = set()
        for candidate in (str(label or "").strip(), _normalize_label_key(label)):
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

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(json_path, "r", encoding="gbk", errors="ignore") as f:
            data = json.load(f)

    bboxes: List[BBox] = []
    auto_map: Dict[str, int] = label_map.copy() if label_map else {}

    def get_cid(name: str) -> int:
        if name not in auto_map:
            auto_map[name] = len(auto_map)
        return auto_map[name]

    if isinstance(data, dict) and "shapes" in data:
        shapes = data["shapes"]
    elif isinstance(data, list):
        shapes = data
    else:
        raise ValidationError(f"Unrecognized json structure: {json_path}")

    bottom_left_origin = isinstance(data, dict) and _uses_bottom_left_origin(data.get("version"))
    image_height = _annotation_image_height(cfg, data, json_path) if bottom_left_origin else None

    for shape in shapes:
        if skip_hidden and shape.get("hidden", False):
            continue
        if skip_outside and shape.get("outside", False):
            continue
        if shape.get("probability", 1.0) < min_prob:
            continue

        pts = shape.get("points", [])
        if not pts or len(pts) < 2:
            continue
        pts = _normalize_annotation_points(
            pts,
            bottom_left_origin=bottom_left_origin,
            image_height=image_height,
        )

        raw_label = str(shape.get("label", "unknown"))
        raw_label_stripped = raw_label.strip()
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
        if not label_name:
            continue
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

    if missing_labels:
        sample = ", ".join(list(missing_labels)[:10])
        raise ValidationError(
            f"Missing label mappings for {len(missing_labels)} labels in {json_path}: {sample}"
        )

    return bboxes, auto_map


def bbox_to_yolo(bbox: BBox, img_w: int, img_h: int) -> str:
    cx = np.clip((bbox.x_min + bbox.x_max) / 2.0 / img_w, 0, 1)
    cy = np.clip((bbox.y_min + bbox.y_max) / 2.0 / img_h, 0, 1)
    w = np.clip(bbox.width / img_w, 0, 1)
    h = np.clip(bbox.height / img_h, 0, 1)
    return f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
