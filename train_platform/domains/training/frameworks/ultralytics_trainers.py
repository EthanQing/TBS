from __future__ import annotations

from typing import Any, Type

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer


class _ExecutionPathsMixin:
    """Restore platform paths after checkpoint args load and before BaseTrainer creates output dirs."""

    def check_resume(self, overrides: dict[str, Any]) -> None:
        current_paths = {
            key: overrides[key]
            for key in ("data", "project", "name", "exist_ok", "save_dir")
            if key in overrides
        }
        super().check_resume(overrides)
        for key, value in current_paths.items():
            setattr(self.args, key, value)


class PlatformDetectionTrainer(_ExecutionPathsMixin, DetectionTrainer):
    # Generated DDP scripts must be able to import each concrete trainer by module and name.
    pass


class PlatformSegmentationTrainer(_ExecutionPathsMixin, SegmentationTrainer):
    pass


class PlatformPoseTrainer(_ExecutionPathsMixin, PoseTrainer):
    pass


class PlatformOBBTrainer(_ExecutionPathsMixin, OBBTrainer):
    pass


class PlatformClassificationTrainer(_ExecutionPathsMixin, ClassificationTrainer):
    pass


class PlatformRTDETRTrainer(_ExecutionPathsMixin, RTDETRTrainer):
    pass


_TRAINERS_BY_BASE: dict[type, Type] = {
    DetectionTrainer: PlatformDetectionTrainer,
    SegmentationTrainer: PlatformSegmentationTrainer,
    PoseTrainer: PlatformPoseTrainer,
    OBBTrainer: PlatformOBBTrainer,
    ClassificationTrainer: PlatformClassificationTrainer,
    RTDETRTrainer: PlatformRTDETRTrainer,
}


def platform_trainer_for(base_trainer: type) -> Type:
    try:
        return _TRAINERS_BY_BASE[base_trainer]
    except KeyError as exc:
        raise ValueError(f"unsupported Ultralytics trainer: {base_trainer!r}") from exc


__all__ = ["platform_trainer_for"]
