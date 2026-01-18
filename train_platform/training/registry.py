from __future__ import annotations

from typing import List

from train_platform.training.plugins.base import TrainerPlugin
from train_platform.training.plugins.stubs import NotImplementedTrainer
from train_platform.training.plugins.ultralytics_yolo import UltralyticsYOLOTrainer


def get_trainer(*, model_family: str) -> TrainerPlugin:
    mf = (model_family or "").strip()
    plugins: List[TrainerPlugin] = [
        UltralyticsYOLOTrainer(),
        NotImplementedTrainer(name="mmdet", model_family="mmdet"),
        NotImplementedTrainer(name="mmdet", model_family="mmdetection"),
        NotImplementedTrainer(name="detr", model_family="detr"),
        NotImplementedTrainer(name="detr", model_family="dert"),
    ]

    for p in plugins:
        try:
            if p.can_handle(mf):
                return p
        except Exception:
            continue
    raise ValueError(f"No trainer registered for model_family='{mf}'")

