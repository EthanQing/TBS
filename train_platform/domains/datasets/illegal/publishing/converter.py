from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from train_platform.core.config import settings
from train_platform.domains.datasets.illegal.publishing.images import (
    SKIPPABLE_IMAGE_ERRORS,
    open_image_reader,
)
from train_platform.domains.datasets.illegal.publishing.output import (
    SliceOutputConfig,
    apply_split,
    remap_label_files,
    save_slices,
    write_class_files,
)
from train_platform.domains.datasets.illegal.publishing.slicing import (
    SliceInfo,
    assign_labels,
    plan_slices,
    post_filter_slices,
)
from train_platform.domains.datasets.labelme import (
    BBox,
    collect_image_json_pair_details,
    normalize_label_key,
    parse_annotations,
    resolve_annotation_shapes,
)
from train_platform.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

SKIPPABLE_CONVERSION_ERRORS = (ValidationError,) + SKIPPABLE_IMAGE_ERRORS
FATAL_CONVERSION_ERRORS = (KeyboardInterrupt, SystemExit, MemoryError)

DEFAULT_CONFIG: dict[str, Any] = {
    "slice_enabled": True,
    "slice_size": 1280,
    "overlap": 0.2,
    "padding": 64,
    "min_area_ratio": 0.3,
    "min_visibility": 0.15,
    "min_pixel_size": 5,
    "min_probability": 0.0,
    "skip_hidden": True,
    "skip_outside": True,
    "label_strategy": "mapping",
    "label_separator": "%",
    "negative_ratio": 0.1,
    "empty_positive_action": "discard",
    "output_format": "jpg",
    "jpg_quality": 95,
    "png_compress_level": 1,
}


@dataclass(frozen=True)
class ConversionConfig:
    slice_enabled: bool
    slice_size: int
    overlap: float
    padding: int
    min_area_ratio: float
    min_visibility: float
    min_pixel_size: int
    min_probability: float
    skip_hidden: bool
    skip_outside: bool
    label_strategy: Any
    label_separator: str
    negative_ratio: float
    empty_positive_action: str
    output_format: str
    jpg_quality: int
    png_compress_level: int

    def annotation_options(
        self,
        *,
        annotation_path: Path,
        label_map: Dict[str, int],
        label_mapping: Optional[dict[str, str]],
        image_height: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "annotation_path": str(annotation_path),
            "label_map": label_map,
            "label_mapping": label_mapping,
            "label_strategy": self.label_strategy,
            "label_separator": self.label_separator,
            "min_probability": self.min_probability,
            "skip_hidden": self.skip_hidden,
            "skip_outside": self.skip_outside,
            "image_height": image_height,
        }


@dataclass(frozen=True)
class PairConversionInput:
    source_root: Path
    output_root: Path
    image_path: Path
    annotation_path: Path
    label_mapping: Optional[dict[str, str]]
    label_map: Dict[str, int]
    config: ConversionConfig


def normalize_conversion_config(publish_config: Optional[dict]) -> ConversionConfig:
    values = dict(DEFAULT_CONFIG)
    raw = publish_config if isinstance(publish_config, dict) else {}
    conversion = raw.get("conversion") if isinstance(raw.get("conversion"), dict) else {}
    slice_cfg = conversion.get("slice") if isinstance(conversion.get("slice"), dict) else {}
    flat_cfg = raw.get("slice") if isinstance(raw.get("slice"), dict) else {}
    merged = {**flat_cfg, **slice_cfg}

    if merged.get("enabled") is not None:
        values["slice_enabled"] = bool(merged.get("enabled"))
    for key in (
        "slice_size",
        "overlap",
        "padding",
        "min_area_ratio",
        "min_visibility",
        "min_pixel_size",
        "negative_ratio",
        "empty_positive_action",
        "output_format",
        "jpg_quality",
        "png_compress_level",
        "label_separator",
        "label_strategy",
    ):
        if merged.get(key) is not None:
            values[key] = merged.get(key)

    values["slice_size"] = max(64, int(values["slice_size"]))
    values["padding"] = max(0, int(values["padding"]))
    values["overlap"] = max(0.0, min(0.95, float(values["overlap"])))
    values["min_area_ratio"] = max(0.0, min(1.0, float(values["min_area_ratio"])))
    values["min_visibility"] = max(0.0, min(1.0, float(values["min_visibility"])))
    values["min_pixel_size"] = max(1, int(values["min_pixel_size"]))
    values["negative_ratio"] = max(0.0, min(1.0, float(values["negative_ratio"])))
    values["empty_positive_action"] = str(values["empty_positive_action"] or "discard")
    values["label_separator"] = str(values["label_separator"] or "%")
    values["label_strategy"] = values["label_strategy"] or "mapping"
    values["output_format"] = str(values["output_format"] or "jpg").lower().strip(".") or "jpg"
    if values["output_format"] not in {"jpg", "jpeg", "png", "bmp", "webp"}:
        values["output_format"] = "jpg"
    return ConversionConfig(
        slice_enabled=bool(values["slice_enabled"]),
        slice_size=int(values["slice_size"]),
        overlap=float(values["overlap"]),
        padding=int(values["padding"]),
        min_area_ratio=float(values["min_area_ratio"]),
        min_visibility=float(values["min_visibility"]),
        min_pixel_size=int(values["min_pixel_size"]),
        min_probability=float(values["min_probability"]),
        skip_hidden=bool(values["skip_hidden"]),
        skip_outside=bool(values["skip_outside"]),
        label_strategy=values["label_strategy"],
        label_separator=str(values["label_separator"]),
        negative_ratio=float(values["negative_ratio"]),
        empty_positive_action=str(values["empty_positive_action"]),
        output_format=str(values["output_format"]),
        jpg_quality=int(values["jpg_quality"]),
        png_compress_level=int(values["png_compress_level"]),
    )


def _build_effective_mapping(
    label_mapping: Optional[dict[str, str]],
    label_filters: Optional[list[str]],
    *,
    label_separator: str = "%",
) -> dict[str, str]:
    mapping = {
        str(key).strip(): str(value).strip()
        for key, value in (label_mapping or {}).items()
        if str(key).strip()
    }
    filters = {str(item).strip() for item in (label_filters or []) if str(item).strip()}
    effective: dict[str, str] = {}
    for raw_label, mapped_label in mapping.items():
        if mapped_label in {"", "__DISCARD__"}:
            effective[raw_label] = "__DISCARD__"
        elif filters:
            effective[raw_label] = mapped_label if mapped_label in filters else "__DISCARD__"
        else:
            effective[raw_label] = mapped_label

    separator = str(label_separator or "%")
    deleted_roots = [
        raw_label
        for raw_label, mapped_label in effective.items()
        if mapped_label in {"", "__DISCARD__"}
    ]
    if not deleted_roots:
        return effective

    def is_descendant(label: str, parent: str) -> bool:
        if label == parent:
            return False
        if separator:
            label_norm = normalize_label_key(label)
            parent_norm = normalize_label_key(parent)
            return label.startswith(f"{parent}{separator}") or label_norm.startswith(f"{parent_norm}{separator}")
        return False

    for raw_label in list(effective.keys()):
        if effective.get(raw_label) in {"", "__DISCARD__"}:
            continue
        if any(is_descendant(raw_label, parent) for parent in deleted_roots):
            effective[raw_label] = "__DISCARD__"
    return effective


def _pair_input(
    source_root: Path,
    output_root: Path,
    image_path: Path,
    annotation_path: Path,
    *,
    label_mapping: Optional[dict[str, str]],
    label_map: Dict[str, int],
    config: ConversionConfig,
) -> PairConversionInput:
    return PairConversionInput(
        source_root=source_root,
        output_root=output_root,
        image_path=image_path,
        annotation_path=annotation_path,
        label_mapping=label_mapping,
        label_map=dict(label_map),
        config=config,
    )


def _scan_label_map(
    pairs: list[tuple[Path, Path]],
    *,
    label_mapping: Optional[dict[str, str]],
    config: ConversionConfig,
) -> Dict[str, int]:
    label_map: Dict[str, int] = {}
    for image_path, annotation_path in pairs:
        try:
            options = config.annotation_options(
                annotation_path=annotation_path,
                label_map=label_map,
                label_mapping=label_mapping,
            )
            for _shape, label_name in resolve_annotation_shapes(options):
                if label_name not in label_map:
                    label_map[label_name] = len(label_map)
        except SKIPPABLE_CONVERSION_ERRORS as exc:
            logger.warning(
                "Skipped illegal dataset publish label pre-scan sample %s / %s: %s",
                image_path.name,
                annotation_path.name,
                exc,
            )
            continue
        except FATAL_CONVERSION_ERRORS:
            raise
        except Exception as exc:
            logger.warning(
                "Skipped illegal dataset publish label pre-scan sample %s / %s: %s",
                image_path.name,
                annotation_path.name,
                exc,
            )
            continue
    return label_map


def _convert_pair(pair: PairConversionInput) -> tuple[dict[str, int], set[str]]:
    config = pair.config
    with open_image_reader(str(pair.image_path), slice_size=config.slice_size) as reader:
        img_w, img_h = int(reader.width), int(reader.height)
        options = config.annotation_options(
            annotation_path=pair.annotation_path,
            label_map=pair.label_map,
            label_mapping=pair.label_mapping,
            image_height=img_h,
        )
        bboxes, _label_map = parse_annotations(options)
        if not bboxes:
            raise ValidationError(f"No valid annotations in {pair.annotation_path}")

        for bbox in bboxes:
            bbox.x_min = max(0.0, bbox.x_min)
            bbox.y_min = max(0.0, bbox.y_min)
            bbox.x_max = min(float(img_w), bbox.x_max)
            bbox.y_max = min(float(img_h), bbox.y_max)
        bboxes = [bbox for bbox in bboxes if bbox.width > 0 and bbox.height > 0]
        if not bboxes:
            raise ValidationError(f"No valid annotations after clipping for {pair.annotation_path}")

        if config.slice_enabled:
            slices = plan_slices(
                img_w,
                img_h,
                bboxes,
                slice_size=config.slice_size,
                overlap=config.overlap,
                padding=config.padding,
                negative_ratio=config.negative_ratio,
            )
            if not slices:
                raise ValidationError(f"No slices planned for {pair.annotation_path}")
            slices = assign_labels(
                slices,
                bboxes,
                min_area_ratio=config.min_area_ratio,
                min_visibility=config.min_visibility,
                min_pixel_size=config.min_pixel_size,
            )
            slices = post_filter_slices(slices, action=config.empty_positive_action)
        else:
            slices = [
                SliceInfo(
                    idx=0,
                    x=0,
                    y=0,
                    w=img_w,
                    h=img_h,
                    is_negative=False,
                    bboxes=[
                        BBox(
                            x_min=float(bbox.x_min),
                            y_min=float(bbox.y_min),
                            x_max=float(bbox.x_max),
                            y_max=float(bbox.y_max),
                            label=str(bbox.label),
                            class_id=int(bbox.class_id),
                        )
                        for bbox in bboxes
                    ],
                )
            ]

        rel_stem = str(pair.image_path.relative_to(pair.source_root).with_suffix(""))
        rel_stem = rel_stem.replace(os.sep, "_").replace("/", "_")
        stats = save_slices(
            SliceOutputConfig(
                output_dir=pair.output_root,
                output_format=config.output_format,
                prefix=f"{rel_stem}_slice",
                jpg_quality=config.jpg_quality,
                png_compress_level=config.png_compress_level,
                negative_ratio=config.negative_ratio,
            ),
            slices,
            reader,
        )
        return stats, {str(bbox.label) for bbox in bboxes if str(bbox.label).strip()}


def convert_dataset(
    source_root: Path,
    output_root: Path,
    *,
    label_mapping: Optional[dict[str, str]] = None,
    label_filters: Optional[list[str]] = None,
    publish_config: Optional[dict] = None,
    split_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    source_root = Path(source_root).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    if not source_root.exists() or not source_root.is_dir():
        raise ValidationError("Dataset root not found for conversion")

    config = normalize_conversion_config(publish_config)
    output_root.mkdir(parents=True, exist_ok=True)
    effective_mapping = _build_effective_mapping(
        label_mapping,
        label_filters,
        label_separator=config.label_separator,
    )
    pairs, warnings, unmatched_files = collect_image_json_pair_details(
        source_root,
        skip_dirs={"labels", ".versions", ".thumbnails", "__macosx"},
        skip_files={".dataset_stats.json", ".dataset_view_index.json", ".mounted_manifest.json"},
    )
    if not pairs:
        detail = "; ".join(unmatched_files[:5])
        suffix = f". Skipped unmatched files: {detail}" if detail else ""
        raise ValidationError(f"No image/json pairs found for illegal dataset publish{suffix}")

    if callable(progress_callback):
        progress_callback(
            "converting",
            {
                "message": f"已匹配 {len(pairs)} 组图片和标注，开始转换",
                "processed": 0,
                "completed": 0,
                "total": len(pairs),
                "skipped": len(unmatched_files),
            },
        )

    global_label_map = _scan_label_map(
        pairs,
        label_mapping=effective_mapping if effective_mapping else None,
        config=config,
    )
    processed = 0
    completed = 0
    skipped_files: list[str] = list(unmatched_files)
    successful_labels: set[str] = set()
    aggregate_stats = {"images": 0, "slices": 0, "labels": 0, "empty_slices": 0}

    def skip_pair(image_path: Path, annotation_path: Path, exc: BaseException) -> None:
        nonlocal completed
        skip_name = f"{image_path.name} / {annotation_path.name}"
        skipped_files.append(f"{skip_name}: {exc}")
        warnings.append(f"Skipped {skip_name}: {exc}")
        logger.warning("Skipped illegal dataset publish sample %s: %s", skip_name, exc)
        completed += 1
        if callable(progress_callback):
            progress_callback(
                "converting",
                {
                    "message": f"跳过 {skip_name}: {exc}",
                    "processed": processed,
                    "completed": completed,
                    "total": len(pairs),
                    "skipped": len(skipped_files),
                    "current_file": skip_name,
                },
            )

    def apply_success(annotation_path: Path, stats: dict[str, int], labels: set[str]) -> None:
        nonlocal processed, completed
        processed += 1
        completed += 1
        aggregate_stats["images"] += 1
        aggregate_stats["slices"] += int(stats.get("total", 0))
        aggregate_stats["labels"] += int(stats.get("total_labels", 0))
        aggregate_stats["empty_slices"] += int(stats.get("empty", 0))
        successful_labels.update({str(label) for label in labels if str(label).strip()})
        if callable(progress_callback):
            progress_callback(
                "converting",
                {
                    "message": f"已转换 {processed}/{len(pairs)} 组数据",
                    "processed": processed,
                    "completed": completed,
                    "total": len(pairs),
                    "skipped": len(skipped_files),
                    "current_file": annotation_path.name,
                },
            )

    def convert_pair(image_path: Path, annotation_path: Path) -> tuple[dict[str, int], set[str]]:
        pair = _pair_input(
            source_root,
            output_root,
            image_path,
            annotation_path,
            label_mapping=effective_mapping if effective_mapping else None,
            label_map=global_label_map,
            config=config,
        )
        return _convert_pair(pair)

    max_workers = min(max(1, int(settings.illegal_dataset_publish_max_workers or 1)), len(pairs))
    if max_workers <= 1:
        for image_path, annotation_path in pairs:
            try:
                stats, labels = convert_pair(image_path, annotation_path)
            except SKIPPABLE_CONVERSION_ERRORS as exc:
                skip_pair(image_path, annotation_path, exc)
                continue
            except FATAL_CONVERSION_ERRORS:
                raise
            except Exception as exc:
                skip_pair(image_path, annotation_path, exc)
                continue
            apply_success(annotation_path, stats, labels)
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="illegal-publish") as executor:
            futures = {
                executor.submit(convert_pair, image_path, annotation_path): (image_path, annotation_path)
                for image_path, annotation_path in pairs
            }
            for future in as_completed(futures):
                image_path, annotation_path = futures[future]
                try:
                    stats, labels = future.result()
                except SKIPPABLE_CONVERSION_ERRORS as exc:
                    skip_pair(image_path, annotation_path, exc)
                    continue
                except FATAL_CONVERSION_ERRORS:
                    raise
                except Exception as exc:
                    skip_pair(image_path, annotation_path, exc)
                    continue
                apply_success(annotation_path, stats, labels)

    final_label_map = {
        label: idx
        for idx, label in enumerate(
            label
            for label, _cid in sorted(global_label_map.items(), key=lambda item: item[1])
            if label and label.strip() and label in successful_labels
        )
    }
    if final_label_map:
        old_to_new_class_ids = {
            int(old_cid): int(final_label_map[label])
            for label, old_cid in global_label_map.items()
            if label in final_label_map
        }
        remap_label_files(output_root, old_to_new_class_ids)

    if processed == 0:
        skipped_summary = "; ".join(skipped_files[:5])
        raise ValidationError(
            f"All {len(pairs)} image/json pairs failed conversion. Details: {skipped_summary}"
        )
    if aggregate_stats["slices"] <= 0:
        raise ValidationError("No valid YOLO samples were generated during publish conversion")

    if callable(progress_callback):
        progress_callback(
            "converting",
            {
                "message": "转换完成，正在整理切分与类别信息",
                "processed": processed,
                "completed": completed,
                "total": len(pairs),
                "skipped": len(skipped_files),
            },
        )
    split_summary = apply_split(output_root, split_config=split_config)
    class_names = write_class_files(output_root, final_label_map, successful_labels, split_summary)

    return {
        "pairs_total": len(pairs),
        "pairs_processed": processed,
        "pairs_skipped": len(skipped_files),
        "skipped_details": skipped_files,
        "warnings": warnings,
        "class_names": class_names,
        "stats": aggregate_stats,
        "split_summary": split_summary,
        "normalized_slice_config": {
            "enabled": config.slice_enabled,
            "slice_size": config.slice_size,
            "overlap": config.overlap,
            "padding": config.padding,
            "min_area_ratio": config.min_area_ratio,
            "min_visibility": config.min_visibility,
            "min_pixel_size": config.min_pixel_size,
            "negative_ratio": config.negative_ratio,
            "empty_positive_action": config.empty_positive_action,
            "output_format": config.output_format,
        },
    }
