from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Protocol


CancelRequestedFn = Callable[[], bool]
UpsertEpochMetricsFn = Callable[[int, Dict[str, float]], None]


@dataclass(frozen=True)
class TrainContext:
    job_id: str
    job: Any
    dataset_path: Path
    run_dir: Path
    cancel_requested: CancelRequestedFn
    upsert_epoch_metrics: UpsertEpochMetricsFn


class TrainerPlugin(Protocol):
    plugin_id: str
    name: str
    display_name: str
    implemented: bool

    def can_handle(self, model_family: str) -> bool: ...

    def get_config_schema(self) -> Dict[str, Any]: ...

    def normalize_config(self, raw: Dict[str, Any] | None) -> Dict[str, Any]: ...

    def run(self, ctx: TrainContext, *, config: Dict[str, Any] | None = None) -> None: ...
