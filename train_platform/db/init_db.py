from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.db.session import SessionLocal


def init_db() -> None:
    """
    Seed minimal reference data.

    NOTE: Schema migrations should be managed via Alembic (see alembic.ini).
    This function intentionally does NOT call Base.metadata.create_all().
    """
    with SessionLocal() as db:
        _seed_architectures(db)


def _seed_architectures(db: Session) -> None:
    from train_platform.models.enums import TaskType
    from train_platform.models.architecture import ModelArchitecture

    defaults = [
        # Ultralytics YOLO (detection)
        dict(family="YOLOv8", variant="yolov8n", task_type=TaskType.DETECTION, engine="ultralytics-yolo"),
        dict(family="YOLOv8", variant="yolov8s", task_type=TaskType.DETECTION, engine="ultralytics-yolo"),
        dict(family="YOLOv11", variant="yolo11n", task_type=TaskType.DETECTION, engine="ultralytics-yolo"),
        dict(family="YOLOv11", variant="yolo11s", task_type=TaskType.DETECTION, engine="ultralytics-yolo"),
        # Placeholders for future engines (no trainer yet)
        dict(family="MMDet", variant="rtmdet_tiny", task_type=TaskType.DETECTION, engine="mmdet"),
        dict(family="DETR", variant="detr_resnet50", task_type=TaskType.DETECTION, engine="detr"),
    ]

    added = 0
    for d in defaults:
        exists = (
            db.query(ModelArchitecture)
            .filter(
                ModelArchitecture.family == d["family"],
                ModelArchitecture.variant == d["variant"],
                ModelArchitecture.task_type == d["task_type"],
            )
            .first()
        )
        if exists:
            continue
        db.add(ModelArchitecture(**d))
        added += 1

    if added:
        db.commit()
