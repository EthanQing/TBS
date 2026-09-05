from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Mapping


class TrainingContext:
    """Small, process-local TBS Training SDK v1 context."""

    __slots__ = (
        "run_id",
        "dataset_path",
        "output_dir",
        "epochs",
        "batch_size",
        "image_size",
        "learning_rate",
        "optimizer",
        "workers",
        "device",
        "custom_args",
        "_cancel_marker_path",
        "_event_path",
        "_event_stream",
        "_output_lock",
    )

    def __init__(
        self,
        *,
        run_id: str,
        dataset_path: Path,
        output_dir: Path,
        epochs: int,
        batch_size: int,
        image_size: int,
        learning_rate: float,
        optimizer: str,
        workers: int,
        device: str,
        custom_args: Mapping[str, Any] | None = None,
        _cancel_marker_path: Path | None = None,
        _event_path: Path | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.learning_rate = float(learning_rate)
        self.optimizer = str(optimizer)
        self.workers = int(workers)
        self.device = str(device)
        self.custom_args = dict(custom_args or {})
        self._cancel_marker_path = Path(_cancel_marker_path) if _cancel_marker_path else None
        self._event_path = Path(_event_path) if _event_path else None
        self._event_stream = None
        if self._event_path is not None:
            self._event_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_stream = self._event_path.open("a", encoding="utf-8", buffering=1)
        self._output_lock = threading.Lock()

    def report_metrics(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")

        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric '{key}' must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"metric '{key}' must be finite")
            normalized[key] = number

        self._emit({"type": "metrics", "epoch": epoch, "metrics": normalized})

    def should_cancel(self) -> bool:
        return bool(self._cancel_marker_path and self._cancel_marker_path.is_file())

    def log(self, message: str) -> None:
        self._emit({"type": "log", "message": str(message)})

    def _emit(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if self._event_stream is None:
            raise RuntimeError("TrainingContext event channel is not initialized")
        with self._output_lock:
            self._event_stream.write(line + "\n")
            self._event_stream.flush()


__all__ = ["TrainingContext"]
