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


DEFAULT_ARCHITECTURES: list[dict] = [
    *yolov8_detection_variants(),
    *yolov8_classification_variants(),
]

