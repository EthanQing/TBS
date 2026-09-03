from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from train_platform.domains.datasets.labelme import BBox


@dataclass
class SliceInfo:
    idx: int
    x: int
    y: int
    w: int
    h: int
    is_negative: bool = False
    bboxes: List[BBox] = field(default_factory=list)


def plan_slices(
    img_w: int,
    img_h: int,
    bboxes: List[BBox],
    slice_size: int,
    overlap: float,
    padding: int,
    negative_ratio: float,
) -> List[SliceInfo]:
    if not bboxes:
        return []

    stride = max(1, int(slice_size * (1 - overlap)))
    grid_cols = max(1, math.ceil((img_w - slice_size) / stride) + 1)
    grid_rows = max(1, math.ceil((img_h - slice_size) / stride) + 1)

    active_cells = set()
    for bbox in bboxes:
        bx0 = max(0, bbox.x_min - padding)
        by0 = max(0, bbox.y_min - padding)
        bx1 = min(img_w, bbox.x_max + padding)
        by1 = min(img_h, bbox.y_max + padding)

        c0 = max(0, int(bx0 // stride))
        c1 = min(grid_cols - 1, int(bx1 // stride))
        r0 = max(0, int(by0 // stride))
        r1 = min(grid_rows - 1, int(by1 // stride))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                active_cells.add((r, c))

    def make_slice(r, c, idx, is_neg=False):
        x = min(c * stride, max(0, img_w - slice_size))
        y = min(r * stride, max(0, img_h - slice_size))
        w = min(slice_size, img_w - x)
        h = min(slice_size, img_h - y)
        return SliceInfo(idx=idx, x=x, y=y, w=w, h=h, is_negative=is_neg)

    seen = set()
    slices: List[SliceInfo] = []
    for (r, c) in sorted(active_cells):
        current = make_slice(r, c, len(slices), is_neg=False)
        key = (current.x, current.y)
        if key not in seen:
            seen.add(key)
            slices.append(current)
    n_positive = len(slices)

    if negative_ratio > 0:
        inactive_cells = [
            (r, c)
            for r in range(grid_rows)
            for c in range(grid_cols)
            if (r, c) not in active_cells
        ]
        n_neg = min(max(1, int(n_positive * negative_ratio)), len(inactive_cells))
        if n_neg > 0 and inactive_cells:
            rng = np.random.default_rng()
            chosen = (
                rng.choice(len(inactive_cells), size=n_neg, replace=False)
                if len(inactive_cells) > n_neg
                else range(len(inactive_cells))
            )
            for cell_index in chosen:
                r, c = inactive_cells[int(cell_index)]
                current = make_slice(r, c, len(slices), is_neg=True)
                key = (current.x, current.y)
                if key not in seen:
                    seen.add(key)
                    slices.append(current)
    return slices


def assign_labels(
    slices: List[SliceInfo],
    bboxes: List[BBox],
    min_area_ratio: float,
    min_visibility: float,
    min_pixel_size: int,
) -> List[SliceInfo]:
    bucket = 1024
    bbox_buckets: Dict[Tuple[int, int], List[int]] = {}
    bbox_x_min = np.asarray([bbox.x_min for bbox in bboxes], dtype=np.float32)
    bbox_y_min = np.asarray([bbox.y_min for bbox in bboxes], dtype=np.float32)
    bbox_x_max = np.asarray([bbox.x_max for bbox in bboxes], dtype=np.float32)
    bbox_y_max = np.asarray([bbox.y_max for bbox in bboxes], dtype=np.float32)
    bbox_width = bbox_x_max - bbox_x_min
    bbox_height = bbox_y_max - bbox_y_min
    bbox_area = np.maximum(0.0, bbox_width) * np.maximum(0.0, bbox_height)
    bbox_labels = [bbox.label for bbox in bboxes]
    bbox_class_ids = np.asarray([bbox.class_id for bbox in bboxes], dtype=np.int32)

    for bbox_idx, bbox in enumerate(bboxes):
        r0, r1 = int(bbox.y_min) // bucket, int(bbox.y_max) // bucket
        c0, c1 = int(bbox.x_min) // bucket, int(bbox.x_max) // bucket
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_buckets.setdefault((r, c), []).append(bbox_idx)

    bbox_bucket_arrays = {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in bbox_buckets.items()
    }

    for current in slices:
        sx0, sy0 = current.x, current.y
        sx1, sy1 = current.x + current.w, current.y + current.h
        r0, r1 = sy0 // bucket, sy1 // bucket
        c0, c1 = sx0 // bucket, sx1 // bucket

        candidate_arrays: List[np.ndarray] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                candidate = bbox_bucket_arrays.get((r, c))
                if candidate is not None and candidate.size > 0:
                    candidate_arrays.append(candidate)

        if not candidate_arrays:
            continue

        candidate_ids = (
            candidate_arrays[0]
            if len(candidate_arrays) == 1
            else np.unique(np.concatenate(candidate_arrays))
        )

        ix0 = np.maximum(float(sx0), bbox_x_min[candidate_ids])
        iy0 = np.maximum(float(sy0), bbox_y_min[candidate_ids])
        ix1 = np.minimum(float(sx1), bbox_x_max[candidate_ids])
        iy1 = np.minimum(float(sy1), bbox_y_max[candidate_ids])

        inter_w = ix1 - ix0
        inter_h = iy1 - iy0
        valid = (inter_w > 0.0) & (inter_h > 0.0)
        if not valid.any():
            continue

        inter_area = inter_w * inter_h
        valid &= bbox_area[candidate_ids] > 0.0
        valid &= (inter_area / np.maximum(bbox_area[candidate_ids], 1e-6)) >= float(min_area_ratio)
        valid &= (inter_w / np.maximum(bbox_width[candidate_ids], 1e-6)) >= float(min_visibility)
        valid &= (inter_h / np.maximum(bbox_height[candidate_ids], 1e-6)) >= float(min_visibility)
        valid &= inter_w >= float(min_pixel_size)
        valid &= inter_h >= float(min_pixel_size)
        if not valid.any():
            continue

        kept_ids = candidate_ids[valid]
        kept_ix0 = ix0[valid] - float(sx0)
        kept_iy0 = iy0[valid] - float(sy0)
        kept_ix1 = ix1[valid] - float(sx0)
        kept_iy1 = iy1[valid] - float(sy0)
        current.bboxes.extend(
            BBox(
                x_min=float(kept_ix0[pos]),
                y_min=float(kept_iy0[pos]),
                x_max=float(kept_ix1[pos]),
                y_max=float(kept_iy1[pos]),
                label=bbox_labels[int(bbox_idx)],
                class_id=int(bbox_class_ids[int(bbox_idx)]),
            )
            for pos, bbox_idx in enumerate(kept_ids)
        )

    return slices


def post_filter_slices(slices: List[SliceInfo], action: str = "discard") -> List[SliceInfo]:
    kept: List[SliceInfo] = []
    for current in slices:
        has_labels = len(current.bboxes) > 0
        was_positive = not current.is_negative
        if was_positive and not has_labels:
            if action == "negative":
                current.is_negative = True
                kept.append(current)
        else:
            kept.append(current)

    for idx, current in enumerate(kept):
        current.idx = idx
    return kept
