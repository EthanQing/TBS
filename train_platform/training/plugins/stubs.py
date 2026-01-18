from __future__ import annotations

from dataclasses import dataclass

from train_platform.training.plugins.base import TrainContext


@dataclass(frozen=True)
class NotImplementedTrainer:
    name: str
    model_family: str

    def can_handle(self, model_family: str) -> bool:
        return model_family.strip().lower() == self.model_family.strip().lower()

    def run(self, ctx: TrainContext) -> None:
        raise NotImplementedError(
            f"Trainer '{self.name}' for model_family='{self.model_family}' is not implemented yet."
        )

