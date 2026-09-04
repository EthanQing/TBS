from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image

from train_platform.domains.datasets.illegal.publishing.images import (
    SKIPPABLE_IMAGE_ERRORS,
    BaseImageReader,
)
from train_platform.domains.datasets.illegal.publishing.slicing import SliceInfo
from train_platform.domains.datasets.labelme import bbox_to_yolo
from train_platform.utils.exceptions import ValidationError
from train_platform.domains.datasets.images import IMAGE_EXTS


@dataclass(frozen=True)
class SliceOutputConfig:
    output_dir: Path
    output_format: str
    prefix: str
    jpg_quality: int
    png_compress_level: int
    negative_ratio: float


def save_slices(
    config: SliceOutputConfig,
    slices: List[SliceInfo],
    reader: BaseImageReader,
) -> Dict[str, int]:
    output_dir = Path(config.output_dir)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ext = str(config.output_format).lower().strip(".")
    prefix = str(config.prefix)
    quality = int(config.jpg_quality)
    png_compress_level = int(config.png_compress_level)
    save_negative = float(config.negative_ratio) > 0

    stats = {"total": 0, "with_labels": 0, "empty": 0, "total_labels": 0}
    slices_to_save = [sl for sl in slices if len(sl.bboxes) > 0 or save_negative]
    if not slices_to_save:
        return stats

    for current in sorted(slices_to_save, key=lambda item: (item.y, item.x, item.idx)):
        has_labels = len(current.bboxes) > 0
        try:
            rgb = reader.read_window_rgb(current.x, current.y, current.w, current.h)
        except SKIPPABLE_IMAGE_ERRORS as exc:
            raise ValidationError(
                f"Unreadable image window {Path(str(getattr(reader, 'image_path', ''))).name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        name = f"{prefix}_{current.idx:06d}"
        img_path = img_dir / f"{name}.{ext}"
        try:
            pil_img = Image.fromarray(rgb)
            if ext in ("jpg", "jpeg"):
                pil_img.save(img_path, quality=quality, optimize=False)
            elif ext == "png":
                pil_img.save(img_path, compress_level=max(0, min(9, png_compress_level)))
            else:
                pil_img.save(img_path)
        except SKIPPABLE_IMAGE_ERRORS as exc:
            raise ValidationError(
                f"Failed to write converted image {img_path.name}: {type(exc).__name__}: {exc}"
            ) from exc

        lbl_path = lbl_dir / f"{name}.txt"
        lines = [bbox_to_yolo(bbox, current.w, current.h) for bbox in current.bboxes]
        with open(lbl_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))
                f.write("\n")

        stats["total"] += 1
        if has_labels:
            stats["with_labels"] += 1
            stats["total_labels"] += len(current.bboxes)
        else:
            stats["empty"] += 1

    return stats


def remap_label_files(output_root: Path, old_to_new_class_ids: dict[int, int]) -> None:
    if not old_to_new_class_ids:
        return
    label_roots = [output_root / "labels"]
    for split_name in ("train", "val", "test"):
        split_dir = output_root / "labels" / split_name
        if split_dir.exists():
            label_roots.append(split_dir)

    for label_root in label_roots:
        if not label_root.exists() or not label_root.is_dir():
            continue
        for label_path in sorted(label_root.glob("*.txt")):
            lines = label_path.read_text(encoding="utf-8").splitlines()
            remapped: list[str] = []
            changed = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    old_class_id = int(float(parts[0]))
                except (TypeError, ValueError):
                    remapped.append(stripped)
                    continue
                new_class_id = old_to_new_class_ids.get(old_class_id)
                if new_class_id is None:
                    continue
                if new_class_id != old_class_id:
                    changed = True
                remapped.append(" ".join([str(new_class_id), *parts[1:]]))
            if changed or len(remapped) != len([line for line in lines if line.strip()]):
                label_path.write_text(
                    ("\n".join(remapped) + "\n") if remapped else "",
                    encoding="utf-8",
                )


def write_class_files(
    output_root: Path,
    label_map: Dict[str, int],
    successful_labels: set[str],
    split_summary: Optional[dict],
) -> list[str]:
    class_names = [
        name
        for name, _cid in sorted(label_map.items(), key=lambda item: item[1])
        if name and name.strip() and name in successful_labels
    ]
    if not class_names:
        raise ValidationError("No valid labels remained after publish conversion")

    classes_path = output_root / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    yaml_payload: dict[str, Any]
    if split_summary and split_summary.get("total_images", 0) > 0:
        train_dir = "images/train" if (output_root / "images" / "train").exists() else "images"
        val_dir = "images/val" if (output_root / "images" / "val").exists() else train_dir
        yaml_payload = {
            "train": train_dir,
            "val": val_dir,
            "nc": len(class_names),
            "names": class_names,
        }
        if (output_root / "images" / "test").exists():
            yaml_payload["test"] = "images/test"
    else:
        yaml_payload = {
            "train": "images",
            "val": "images",
            "nc": len(class_names),
            "names": class_names,
        }

    with open(output_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, allow_unicode=True, sort_keys=False)
    return class_names


def normalize_split_config(split_config: Optional[dict]) -> dict[str, Any]:
    raw = split_config if isinstance(split_config, dict) else {}
    train = float(raw.get("train") or 0)
    val = float(raw.get("val") or 0)
    test = float(raw.get("test") or 0)
    if min(train, val, test) < 0 or max(train, val, test) > 1:
        raise ValidationError("train / val / test must be between 0 and 1")
    total = train + val + test
    if total <= 0:
        return {
            "enabled": False,
            "train": 1.0,
            "val": 0.0,
            "test": 0.0,
            "shuffle": False,
            "seed": None,
        }
    if abs(total - 1.0) > 0.001:
        raise ValidationError("train + val + test must equal 1")
    if train <= 0:
        raise ValidationError("train split must be greater than 0")
    return {
        "enabled": True,
        "train": train,
        "val": val,
        "test": test,
        "shuffle": bool(raw.get("shuffle", True)),
        "seed": int(raw["seed"])
        if raw.get("seed") is not None and str(raw.get("seed")).strip() != ""
        else None,
    }


def iter_generated_pairs(output_root: Path) -> list[tuple[Path, Path]]:
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def apply_split(output_root: Path, *, split_config: Optional[dict]) -> Optional[dict[str, Any]]:
    cfg = normalize_split_config(split_config)
    if not cfg.get("enabled"):
        return None

    pairs = iter_generated_pairs(output_root)
    total = len(pairs)
    if total <= 0:
        raise ValidationError("No converted images available for split")

    items = list(pairs)
    if cfg["shuffle"]:
        rng = random.Random(cfg["seed"]) if cfg["seed"] is not None else random.Random()
        rng.shuffle(items)

    desired = {
        "train": total * float(cfg["train"]),
        "val": total * float(cfg["val"]),
        "test": total * float(cfg["test"]),
    }
    counts = {name: int(math.floor(value)) for name, value in desired.items()}
    remaining = total - sum(counts.values())
    priority = {"train": 0, "val": 1, "test": 2}
    if remaining > 0:
        remainders = sorted(
            ((desired[name] - counts[name], priority[name], name) for name in ("train", "val", "test")),
            key=lambda item: (-item[0], item[1]),
        )
        idx = 0
        while remaining > 0 and remainders:
            _, _, name = remainders[idx % len(remainders)]
            counts[name] += 1
            remaining -= 1
            idx += 1

    if counts["train"] <= 0:
        donor = next((name for name in ("val", "test") if counts[name] > 1), None)
        if donor is None:
            donor = next((name for name in ("val", "test") if counts[name] > 0), None)
        if donor is not None:
            counts[donor] -= 1
        counts["train"] += 1

    train_count = counts["train"]
    val_count = counts["val"]
    test_count = counts["test"]
    assignments = (
        [("train", item) for item in items[:train_count]]
        + [("val", item) for item in items[train_count : train_count + val_count]]
        + [("test", item) for item in items[train_count + val_count :]]
    )

    for split_name, (image_path, label_path) in assignments:
        image_target = output_root / "images" / split_name / image_path.name
        label_target = output_root / "labels" / split_name / label_path.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(image_target))
        shutil.move(str(label_path), str(label_target))

    return {
        "total_images": total,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_ratio": round((train_count / total), 6) if total else 0.0,
        "val_ratio": round((val_count / total), 6) if total else 0.0,
        "test_ratio": round((test_count / total), 6) if total else 0.0,
        "seed": cfg["seed"],
        "shuffle": bool(cfg["shuffle"]),
    }
