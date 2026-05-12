from __future__ import annotations

"""
Seed data definitions for empty databases.

Keep this file declarative and easy to extend: when you add a new model family,
add a new helper and append its output to DEFAULT_ARCHITECTURES.
"""

from train_platform.models.v3.enums import TaskType


def ultralytics_detection_variants(
    *,
    family: str,
    variants: list[str],
    engine: str = "ultralytics-yolo",
) -> list[dict]:
    """
    Build Ultralytics detection architecture rows for one model family.
    """
    return [
        {
            "family": family,
            "variant": v,
            "task_type": TaskType.DETECTION,
            "engine": engine,
        }
        for v in variants
    ]


def yolov8_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    Default YOLOv8 detection architectures.

    Five standard variants: n/s/m/l/x.
    """
    return ultralytics_detection_variants(
        family="YOLOv8",
        variants=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        engine=engine,
    )


def yolov9_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    YOLOv9 detection variants.
    """
    return ultralytics_detection_variants(
        family="YOLOv9",
        variants=["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
        engine=engine,
    )


def yolov10_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    YOLOv10 detection variants.
    """
    return ultralytics_detection_variants(
        family="YOLOv10",
        variants=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"],
        engine=engine,
    )


def yolo11_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    YOLO11 detection variants.
    """
    return ultralytics_detection_variants(
        family="YOLO11",
        variants=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        engine=engine,
    )


def yolo12_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    YOLO12 detection variants.
    """
    return ultralytics_detection_variants(
        family="YOLO12",
        variants=["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"],
        engine=engine,
    )


def yolo26_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    YOLO26 detection variants.
    """
    return ultralytics_detection_variants(
        family="YOLO26",
        variants=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        engine=engine,
    )


def rtdetr_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    RT-DETR detection variants (Ultralytics integration).
    """
    return ultralytics_detection_variants(
        family="RT-DETR",
        variants=["rtdetr-l", "rtdetr-x"],
        engine=engine,
    )

def yolov8_classification_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    Default YOLOv8 classification architectures.

    Five standard variants: n/s/m/l/x.
    """
    variants = ["yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls"]
    return [
        {
            "family": "YOLOv8-cls",
            "variant": v,
            "task_type": TaskType.CLASSIFICATION,
            "engine": engine,
        }
        for v in variants
    ]

def yolov8_segmentation_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    Default YOLOv8 segmentation architectures.

    Five standard variants: n/s/m/l/x.
    """
    variants = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
    return [
        {
            "family": "YOLOv8-seg",
            "variant": v,
            "task_type": TaskType.SEGMENTATION,
            "engine": engine,
        }
        for v in variants
    ]


def paddle_det_detection_variants(*, engine: str = "paddle-det") -> list[dict]:
    """
    PaddleDetection detection architectures.

    PP-YOLOE+ (s/m/l/x) and PicoDet (s/l), split into separate families for
    cleaner UI grouping.
    """
    ppyoloe = [
        ("ppyoloe_s", "PP-YOLOE+ Small", "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml"),
        ("ppyoloe_m", "PP-YOLOE+ Medium", "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml"),
        ("ppyoloe_l", "PP-YOLOE+ Large", "configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml"),
        ("ppyoloe_x", "PP-YOLOE+ XLarge", "configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml"),
    ]
    picodet = [
        ("picodet_s", "PicoDet Small (轻量级)", "configs/picodet/picodet_s_320_coco_lcnet.yml"),
        ("picodet_l", "PicoDet Large", "configs/picodet/picodet_l_640_coco_lcnet.yml"),
    ]
    results = []
    for v in ppyoloe:
        results.append({
            "family": "PP-YOLOE",
            "variant": v[0],
            "task_type": TaskType.DETECTION,
            "engine": engine,
            "description": v[1],
            "default_params": {"config_path": v[2]},
        })
    for v in picodet:
        results.append({
            "family": "PicoDet",
            "variant": v[0],
            "task_type": TaskType.DETECTION,
            "engine": engine,
            "description": v[1],
            "default_params": {"config_path": v[2]},
        })
    return results


DEFAULT_ARCHITECTURES: list[dict] = [
    *yolov8_detection_variants(),
    *yolov9_detection_variants(),
    *yolov10_detection_variants(),
    *yolo11_detection_variants(),
    *yolo12_detection_variants(),
    *yolo26_detection_variants(),
    *rtdetr_detection_variants(),
    # *paddle_det_detection_variants(),
    # *yolov8_classification_variants(),
    # *yolov8_segmentation_variants(),
]
