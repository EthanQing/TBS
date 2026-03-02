from __future__ import annotations

"""
Seed data definitions for empty databases.

Keep this file declarative and easy to extend: when you add a new model family,
add a new helper and append its output to DEFAULT_ARCHITECTURES.
"""

from train_platform.models.enums import TaskType


def yolov8_detection_variants(*, engine: str = "ultralytics-yolo") -> list[dict]:
    """
    Default YOLOv8 detection architectures.

    Five standard variants: n/s/m/l/x.
    """
    variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    return [
        {
            "family": "YOLOv8",
            "variant": v,
            "task_type": TaskType.DETECTION,
            "engine": engine,
        }
        for v in variants
    ]

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
    *paddle_det_detection_variants(),
    # *yolov8_classification_variants(),
    # *yolov8_segmentation_variants(),
]

