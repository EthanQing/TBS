from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from train_platform.training.plugins.base import TrainContext


@dataclass(frozen=True)
class NotImplementedTrainer:
    plugin_id: str
    name: str
    model_family: str
    display_name: str | None = None
    implemented: bool = False

    def can_handle(self, model_family: str) -> bool:
        return model_family.strip().lower() == self.model_family.strip().lower()

    def get_config_schema(self) -> Dict[str, Any]:
        return {}

    def normalize_config(self, raw: Dict[str, Any] | None) -> Dict[str, Any]:
        return dict(raw or {})

    def run(self, ctx: TrainContext, *, config: Dict[str, Any] | None = None) -> None:
        raise NotImplementedError(
            f"Trainer '{self.name}' for model_family='{self.model_family}' is not implemented yet."
        )
