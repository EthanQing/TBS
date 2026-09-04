from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunResult
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.platform.filesystem.locations import resolve_training_path
from train_platform.platform.runtime.paddledetection import resolve_paddledet_config_path

from .artifacts import index_completion_artifacts
from .service import TrainingRunService


class TrainingRunBenchmarkService:
    """Inference measurements and result enrichment for completed training runs."""

    def __init__(self) -> None:
        self._worker = ModelWorkerClient()

    def ensure_benchmark_image(self) -> Path:
        try:
            from PIL import Image
        except Exception as exc:
            raise ValidationError("Pillow is required for benchmark image generation") from exc

        out_dir = (settings.temp_dir / "benchmark_inputs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthetic_640.jpg"
        if not out_path.exists():
            Image.new("RGB", (640, 640), color=(0, 0, 0)).save(out_path, format="JPEG", quality=95)
        return out_path

    @staticmethod
    def _run_image_size(run: TrainingRun) -> int:
        params = getattr(run, "parameters", None)
        raw = getattr(params, "image_size", None) if params is not None else None
        try:
            value = int(float(raw))
        except Exception:
            return 640
        return value if value > 0 else 640

    def measure_yolo_stats(self, run: TrainingRun, result: TrainingRunResult) -> Dict[str, Any]:
        arch = run.architecture
        engine = str(getattr(arch, "engine", "") or "ultralytics-yolo").strip().lower()
        if engine != "ultralytics-yolo":
            return {}

        weights_rel = str(result.best_weights_path or result.last_weights_path or "").strip()
        if not weights_rel:
            return {}
        weights_path = resolve_training_path(weights_rel)
        if not weights_path.exists() or not weights_path.is_file():
            return {}

        try:
            return self._worker.get_yolo_model_stats(
                weights_path=weights_path,
                image_path=self.ensure_benchmark_image(),
                imgsz=self._run_image_size(run),
                conf=0.25,
                iou=0.45,
                warmup=1,
                iters=5,
            )
        except Exception:
            return {}

    def measure_inference_latency(
        self,
        db: Session,
        *,
        run: TrainingRun,
        benchmark_image: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        warmup: int = 1,
        iters: int = 5,
    ) -> float:
        if not run.result:
            raise ConflictError("Run has no result artifacts")

        weights_rel = str(run.result.best_weights_path or run.result.last_weights_path or "").strip()
        if not weights_rel:
            raise ConflictError("Run has no weights path")

        weights_path = resolve_training_path(weights_rel)
        if not weights_path.exists() or not weights_path.is_file():
            raise NotFoundError(f"Weights not found: {weights_path}")

        arch = run.architecture
        engine = str(getattr(arch, "engine", "") or "ultralytics-yolo").strip().lower()
        config_path = None
        if engine == "paddle-det":
            params = arch.default_params if isinstance(getattr(arch, "default_params", None), dict) else {}
            config_path = resolve_paddledet_config_path(params.get("config_path"))
            if not config_path:
                raise ValidationError("Paddle model missing valid config_path")

        warmup = max(0, int(warmup))
        iters = max(1, int(iters))
        for _ in range(warmup):
            self._worker.execute_model(
                engine=engine,
                weights_path=weights_path,
                image_path=benchmark_image,
                conf=float(conf),
                iou=float(iou),
                config_path=config_path,
            )

        timings: List[float] = []
        for _ in range(iters):
            started = time.perf_counter()
            self._worker.execute_model(
                engine=engine,
                weights_path=weights_path,
                image_path=benchmark_image,
                conf=float(conf),
                iou=float(iou),
                config_path=config_path,
            )
            timings.append((time.perf_counter() - started) * 1000.0)
        return round(float(statistics.median(timings)), 4)

    @staticmethod
    def _as_number(value: Any) -> float | None:
        try:
            number = float(value)
        except Exception:
            return None
        return number if number == number else None

    def benchmark_inference_times(
        self,
        db: Session,
        *,
        run_ids: List[str],
        force: bool = False,
    ) -> Dict[str, Any]:
        ids: List[str] = []
        seen = set()
        for value in run_ids or []:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ids.append(normalized)
        if not ids:
            raise ValidationError("run_ids is required")
        if len(ids) > 20:
            raise ValidationError("run_ids cannot exceed 20")

        benchmark_image = self.ensure_benchmark_image()
        items: List[Dict[str, Any]] = []
        for run_id in ids:
            run: TrainingRun | None = None
            engine: str | None = None
            try:
                run = TrainingRunService().get_run(db, run_id)
                engine = str(getattr(run.architecture, "engine", "") or "").strip().lower() or None
                if run.status != TrainingRunStatus.COMPLETED:
                    items.append(
                        {
                            "run_id": run_id,
                            "status": "skipped",
                            "inference_time_ms": None,
                            "engine": engine,
                            "message": "run is not completed",
                        }
                    )
                    continue

                cached = self._as_number(getattr(run.result, "inference_time_ms", None) if run.result else None)
                if cached is not None and cached > 0 and not force:
                    items.append(
                        {
                            "run_id": str(run.run_id),
                            "status": "cached",
                            "inference_time_ms": cached,
                            "engine": engine,
                            "message": "",
                        }
                    )
                    continue

                if run.result is None:
                    run.result = TrainingRunResult(run_id=str(run.run_id))
                    db.add(run.result)
                    db.flush()

                engine_norm = str(engine or "ultralytics-yolo").strip().lower()
                if engine_norm == "ultralytics-yolo":
                    if not run.result.best_weights_path and not run.result.last_weights_path:
                        index_completion_artifacts(db, str(run.run_id))
                        db.flush()
                        db.refresh(run)
                    stats = self.measure_yolo_stats(run, run.result)
                    measured = self._as_number(stats.get("inference_time_ms"))
                    if measured is None:
                        raise RuntimeError("YOLO worker did not return inference_time_ms")
                    flops = self._as_number(stats.get("flops"))
                    if flops and flops > 0:
                        run.result.flops = int(flops)
                else:
                    measured = self.measure_inference_latency(
                        db,
                        run=run,
                        benchmark_image=benchmark_image,
                        conf=0.25,
                        iou=0.45,
                        warmup=1,
                        iters=5,
                    )

                run.result.inference_time_ms = measured
                db.add(run)
                db.commit()
                items.append(
                    {
                        "run_id": str(run.run_id),
                        "status": "measured",
                        "inference_time_ms": measured,
                        "engine": engine,
                        "message": "",
                    }
                )
            except Exception as exc:
                db.rollback()
                items.append(
                    {
                        "run_id": str(getattr(run, "run_id", run_id)),
                        "status": "failed",
                        "inference_time_ms": None,
                        "engine": engine,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {"items": items}


__all__ = ["TrainingRunBenchmarkService"]
