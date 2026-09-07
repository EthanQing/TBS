from __future__ import annotations

import json
import math
import threading
import re
from pathlib import Path, PureWindowsPath
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

    def report_artifact(
        self,
        role: str,
        path: str | Path,
        *,
        format: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(role, str):
            raise TypeError("artifact role must be a string")
        role_value = role
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", role_value):
            raise ValueError("artifact role must match [a-z][a-z0-9_.-]* and be at most 64 characters")

        path_value = str(path)
        if not path_value or "\x00" in path_value:
            raise ValueError("artifact path must be a non-empty relative path")
        windows_path = PureWindowsPath(path_value)
        if (
            Path(path_value).is_absolute()
            or windows_path.is_absolute()
            or windows_path.root
            or windows_path.drive
            or path_value.startswith(("\\\\", "/"))
        ):
            raise ValueError("artifact path must be relative to the training output directory")
        path_parts = path_value.replace("\\", "/").split("/")
        if any(part == ".." for part in path_parts):
            raise ValueError("artifact path must not contain parent traversal")
        normalized_path = "/".join(part for part in path_parts if part not in ("", "."))
        if not normalized_path:
            raise ValueError("artifact path must identify a file below the training output directory")
        if format is not None and not isinstance(format, str):
            raise TypeError("artifact format must be a string")
        if meta is not None and not isinstance(meta, Mapping):
            raise TypeError("artifact meta must be a mapping")

        event: dict[str, Any] = {
            "type": "artifact",
            "role": role_value,
            "path": normalized_path,
        }
        if format is not None:
            event["format"] = format
        if meta is not None:
            event["meta"] = dict(meta)
        try:
            json.dumps(event, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact format and meta must be JSON serializable") from exc
        self._emit(event)

    def _emit(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if self._event_stream is None:
            raise RuntimeError("TrainingContext event channel is not initialized")
        with self._output_lock:
            self._event_stream.write(line + "\n")
            self._event_stream.flush()


__all__ = ["TrainingContext"]
