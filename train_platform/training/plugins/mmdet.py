from __future__ import annotations

from typing import Any, Dict

from train_platform.training.plugins.base import TrainContext


class MMDetTrainer:
    plugin_id = "mmdet"
    name = "mmdet"
    display_name = "MMDetection"
    implemented = False

    def can_handle(self, model_family: str) -> bool:
        mf = (model_family or "").strip().lower()
        return any(x in mf for x in ("mmdet", "detr", "rtmdet", "dert"))

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "config_path": {"type": "string"},
                "work_dir": {"type": "string"},
                "resume_from": {"type": "string"},
                "load_from": {"type": "string"},
            },
            "additionalProperties": True,
        }

    def normalize_config(self, raw: Dict[str, Any] | None) -> Dict[str, Any]:
        return dict(raw or {})

    def run(self, ctx: TrainContext, *, config: Dict[str, Any] | None = None) -> None:
        raise NotImplementedError("MMDetection trainer plugin is not implemented yet.")
