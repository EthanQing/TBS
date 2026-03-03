from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable


MetricUpsertFn = Callable[[int, dict[str, float]], None]


class VisualDLScalarBridge:
    """
    Optional VisualDL scalar bridge (Paddle local dev helper).

    This reader is best-effort and intentionally lightweight:
    - disabled by default (`metrics_source=callback`)
    - when enabled (`metrics_source=hybrid`), it tails VisualDL scalar logs
      and forwards new points into the same epoch-metrics sink.
    """

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        upsert_epoch_metrics: MetricUpsertFn,
        poll_interval_sec: float = 5.0,
    ) -> None:
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.upsert_epoch_metrics = upsert_epoch_metrics
        self.poll_interval_sec = max(1.0, float(poll_interval_sec))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor_by_key: dict[tuple[str, str, str], int] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name=f"vdl-bridge-{self.run_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_sec):
            try:
                self.poll_once()
            except Exception:
                # Best-effort only; training must not fail because bridge fails.
                continue

    def _discover_log_roots(self) -> list[Path]:
        roots: list[Path] = []
        preferred = [
            self.run_dir / "vdl_log_dir",
            self.run_dir / "vdl",
            self.run_dir,
        ]
        seen: set[str] = set()
        for root in preferred:
            key = str(root.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            if root.exists() and root.is_dir():
                roots.append(root)
        return roots

    @staticmethod
    def _map_scalar_tag_to_metrics(tag: str, value: float) -> dict[str, float]:
        out: dict[str, float] = {}
        tag_norm = str(tag or "").strip()
        if not tag_norm:
            return out

        # Preserve original scalar name.
        out[tag_norm] = float(value)

        lower = tag_norm.lower()

        # Common aliases for unified frontend charts.
        if "bbox_ap50" in lower or ("map50" in lower and "95" not in lower):
            out["AP50"] = float(value)
            out["mAP50"] = float(value)
            out["metrics/mAP50(B)"] = float(value)
        if "bbox_map" in lower or "map50-95" in lower:
            out["mAP"] = float(value)
            out["metrics/mAP50-95(B)"] = float(value)
        if "precision" in lower:
            out["precision"] = float(value)
            out["metrics/precision(B)"] = float(value)
        if "recall" in lower:
            out["recall"] = float(value)
            out["metrics/recall(B)"] = float(value)

        if lower in ("train/learning_rate", "learning_rate", "lr", "train/lr"):
            out["lr"] = float(value)

        return out

    def poll_once(self) -> None:
        try:
            from visualdl import LogReader
        except Exception:
            return

        updates_by_step: dict[int, dict[str, float]] = {}

        for root in self._discover_log_roots():
            try:
                reader = LogReader(str(root))
                runs = list(reader.runs(update=True) or [])
                tags = reader.tags() or {}
            except Exception:
                continue

            for run in runs:
                run_prefix = f"{run}\\"
                for full_tag, component in tags.items():
                    if component != "scalar" or not str(full_tag).startswith(run_prefix):
                        continue
                    encoded_tag = str(full_tag)[len(run_prefix):]
                    cursor_key = (str(root), str(run), str(encoded_tag))
                    last_seen = int(self._cursor_by_key.get(cursor_key, -1))
                    max_seen = last_seen
                    try:
                        records = reader.get_log_data("scalar", run, encoded_tag) or []
                    except Exception:
                        records = []
                    for rec in records:
                        if not isinstance(rec, (list, tuple)) or len(rec) < 4:
                            continue
                        step = int(rec[0]) if str(rec[0]).strip() else 0
                        if step <= last_seen:
                            continue
                        max_seen = max(max_seen, step)
                        raw_tag = str(rec[1] or "")
                        try:
                            value = float(rec[3])
                        except Exception:
                            continue
                        mapped = self._map_scalar_tag_to_metrics(raw_tag, value)
                        if not mapped:
                            continue
                        updates_by_step.setdefault(step, {}).update(mapped)
                    self._cursor_by_key[cursor_key] = max_seen

        for step, metrics in sorted(updates_by_step.items()):
            if metrics:
                self.upsert_epoch_metrics(int(step), metrics)

