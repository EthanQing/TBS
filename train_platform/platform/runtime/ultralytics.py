from __future__ import annotations


def apply_torch_safe_load_patches() -> None:
    """Apply compatibility patches required by supported Ultralytics releases."""

    try:
        import torch
        import torch.nn as nn
        import torch.serialization
        from ultralytics.nn.modules import Bottleneck, BottleneckCSP, C2f, Conv, SPPF
        from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel

        safe_classes = [
            DetectionModel,
            SegmentationModel,
            ClassificationModel,
            nn.modules.container.Sequential,
            Conv,
            Bottleneck,
            BottleneckCSP,
            C2f,
            SPPF,
        ]
        torch.serialization.add_safe_globals(safe_classes)

        import ultralytics.nn.tasks

        def patched_torch_safe_load(weight):
            try:
                return torch.load(weight, map_location="cpu"), weight
            except Exception:
                return torch.load(weight, map_location="cpu", weights_only=False), weight

        ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
    except Exception:
        # The helper is intentionally best-effort: environments without the
        # optional Ultralytics dependency still support non-YOLO capabilities.
        pass


__all__ = ["apply_torch_safe_load_patches"]
