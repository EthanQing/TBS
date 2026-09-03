from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.model_assets.runtime import resolve_architecture_config_path
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunArtifact,
    TrainingRunEpochMetric,
    TrainingRunEvent,
    TrainingRunResult,
)
from train_platform.models.v3.training_run_meta import TrainingRunMeta
from train_platform.domains.training.runs import TrainingRunService as TrainingRunDomainService
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.repositories.v3.training_run_meta_repo import TrainingRunMetaRepository
from train_platform.domains.training.runs import compute_epoch_metric_snapshots, index_completion_artifacts
from train_platform.utils.path_utils import resolve_training_path
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is expected in runtime image.
    Image = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


ENGINE_FRAMEWORK_MAP: dict[str, tuple[str, str]] = {
    "ultralytics-yolo": ("pytorch", "PyTorch"),
    "paddle-det": ("paddle", "Paddle"),
}

COMPARE_METRIC_KEYS: tuple[str, ...] = (
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
)

REPORT_CORE_METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "mAP50-95": (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
        "mAP50-95",
        "mAP",
        "map",
        "bbox_map",
        "bbox_mAP",
        "eval/bbox_mAP",
        "eval/bbox_map",
    ),
    "mAP50": (
        "metrics/mAP50(B)",
        "metrics/mAP50(M)",
        "mAP50",
        "AP50",
        "ap50",
        "bbox_ap50",
        "bbox_AP50",
        "eval/bbox_AP50",
        "eval/bbox_ap50",
    ),
    "mAP75": (
        "metrics/mAP75(B)",
        "metrics/mAP75(M)",
        "mAP75",
        "AP75",
        "ap75",
        "bbox_ap75",
        "bbox_AP75",
        "eval/bbox_AP75",
        "eval/bbox_ap75",
    ),
    "Precision": (
        "metrics/precision(B)",
        "metrics/precision(M)",
        "precision",
        "Precision",
        "bbox_precision",
        "eval/bbox_precision",
    ),
    "Recall": (
        "metrics/recall(B)",
        "metrics/recall(M)",
        "recall",
        "Recall",
        "bbox_recall",
        "eval/bbox_recall",
    ),
}


class FrameworkCompareConflict(ConflictError):
    def __init__(self, message: str, framework_groups: Dict[str, List[str]]) -> None:
        super().__init__(message)
        self.framework_groups = framework_groups


def _ensure_aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _tail_text_file(path, *, lines: int) -> str:
    """
    Read last N lines from a text file without loading the whole file.

    Returns empty string if file does not exist.
    """
    try:
        if not path or not path.exists() or not path.is_file():
            return ""
    except Exception:
        return ""

    # Read from end in binary chunks (works for large files and Windows CRLF).
    chunk_size = 4096
    data = b""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0 and data.count(b"\n") <= int(lines):
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos, os.SEEK_SET)
                data = f.read(read_size) + data
                if pos == 0:
                    break
    except Exception:
        return ""

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = str(data)

    parts = text.splitlines()
    tail = parts[-int(lines) :] if parts else []
    return "\n".join(tail)


class TrainingRunService:
    def __init__(self) -> None:
        self.meta_repo = TrainingRunMetaRepository()
        self._worker = ModelWorkerClient()
        self.run_service = TrainingRunDomainService()

    def _get_run(self, db: Session, run_id: str) -> TrainingRun:
        return self.run_service.get_run(db, run_id)

    # --------------------
    # metrics/events/artifacts
    # --------------------
    def list_events(self, db: Session, run_id: str, *, limit: int = 200) -> list[TrainingRunEvent]:
        self._get_run(db, run_id)
        return (
            db.query(TrainingRunEvent)
            .filter(TrainingRunEvent.run_id == str(run_id))
            .order_by(TrainingRunEvent.created_at.desc())
            .limit(int(limit))
            .all()
        )

    def list_epoch_metrics(self, db: Session, run_id: str, *, limit: int = 5000) -> list[TrainingRunEpochMetric]:
        self._get_run(db, run_id)
        return (
            db.query(TrainingRunEpochMetric)
            .filter(TrainingRunEpochMetric.run_id == str(run_id))
            .order_by(TrainingRunEpochMetric.epoch.asc())
            .limit(int(limit))
            .all()
        )

    def list_artifacts(self, db: Session, run_id: str) -> list[TrainingRunArtifact]:
        self._get_run(db, run_id)
        return (
            db.query(TrainingRunArtifact)
            .filter(TrainingRunArtifact.run_id == str(run_id))
            .order_by(TrainingRunArtifact.created_at.desc())
            .all()
        )

    # --------------------
    # meta
    # --------------------
    def get_meta(self, db: Session, run_id: str) -> TrainingRunMeta:
        self._get_run(db, run_id)
        meta = self.meta_repo.get_by_run_id(db, run_id)
        if meta:
            return meta

        meta = TrainingRunMeta(run_id=str(run_id))
        db.add(meta)
        db.commit()
        db.refresh(meta)
        return meta

    def update_meta(self, db: Session, run_id: str, *, patch: dict) -> TrainingRunMeta:
        self._get_run(db, run_id)
        meta = self.meta_repo.get_by_run_id(db, run_id)
        if not meta:
            meta = TrainingRunMeta(run_id=str(run_id))
            db.add(meta)
            db.flush()

        if "creator" in patch:
            meta.creator = patch["creator"]
        if "group" in patch:
            meta.group_name = patch["group"]
        if "tags" in patch:
            meta.tags = patch["tags"]
        if "notes" in patch:
            meta.notes = patch["notes"]
        if "extra" in patch:
            meta.extra = patch["extra"]

        db.commit()
        db.refresh(meta)
        return meta

    def mark_project_card_reviewed(self, db: Session, run_id: str, *, source: str | None = None) -> dict:
        run = self._get_run(db, run_id)
        if bool(getattr(run, "hidden", False)) or run.status != TrainingRunStatus.COMPLETED:
            raise ValidationError("Only visible completed training runs can be marked as reviewed")

        meta = self.meta_repo.get_by_run_id(db, run_id)
        if not meta:
            meta = TrainingRunMeta(run_id=str(run_id))
            db.add(meta)
            db.flush()

        extra = dict(meta.extra) if isinstance(meta.extra, dict) else {}
        reviewed_at = str(extra.get("project_card_reviewed_at") or "").strip()
        if not reviewed_at:
            reviewed_at = _utcnow().isoformat()
            extra["project_card_reviewed_at"] = reviewed_at
        source_norm = str(source or "").strip()
        if source_norm:
            extra["project_card_review_source"] = source_norm[:64]
        meta.extra = extra

        db.commit()
        db.refresh(meta)

        try:
            reviewed_dt = datetime.fromisoformat(reviewed_at)
        except Exception:
            reviewed_dt = _utcnow()
        return {
            "run_id": str(run.run_id),
            "reviewed": True,
            "reviewed_at": reviewed_dt,
            "source": extra.get("project_card_review_source"),
        }

    # --------------------
    # logs
    # --------------------
    def tail_logs(self, db: Session, run_id: str, *, which: str = "stdout", lines: int = 200) -> str:
        """
        Best-effort tail of worker-produced logs.

        `which`: stdout | stderr
        """
        self._get_run(db, run_id)

        which = (which or "").strip().lower()
        if which not in ("stdout", "stderr"):
            raise ValidationError("which must be 'stdout' or 'stderr'")

        lines = int(lines)
        if lines < 1 or lines > 20000:
            raise ValidationError("lines must be between 1 and 20000")

        log_name = "train.stdout.log" if which == "stdout" else "train.stderr.log"
        path = settings.training_dir / str(run_id) / "logs" / log_name
        return _tail_text_file(path, lines=lines)

    @staticmethod
    def _resolve_framework(engine: str | None) -> tuple[str, str]:
        raw = str(engine or "").strip().lower()
        if not raw:
            return "engine:unknown", "Engine: unknown"
        mapped = ENGINE_FRAMEWORK_MAP.get(raw)
        if mapped:
            return mapped
        return f"engine:{raw}", f"Engine: {raw}"

    @staticmethod
    def _as_number(value: Any) -> float | None:
        try:
            n = float(value)
        except Exception:
            return None
        return n if n == n else None

    def _build_metric_summary(self, db: Session, run: TrainingRun) -> Dict[str, Any]:
        best: Dict[str, float] = {}
        final: Dict[str, float] = {}
        used_result = False
        used_epoch = False

        result_best = run.result.best_metrics if run.result and isinstance(run.result.best_metrics, dict) else {}
        result_final = run.result.final_metrics if run.result and isinstance(run.result.final_metrics, dict) else {}

        for key in COMPARE_METRIC_KEYS:
            n_best = self._as_number(result_best.get(key))
            if n_best is not None:
                best[key] = n_best
                used_result = True
            n_final = self._as_number(result_final.get(key))
            if n_final is not None:
                final[key] = n_final
                used_result = True

        needs_epoch = any((key not in best) or (key not in final) for key in COMPARE_METRIC_KEYS)
        if needs_epoch:
            rows = (
                db.query(TrainingRunEpochMetric)
                .filter(TrainingRunEpochMetric.run_id == str(run.run_id))
                .order_by(TrainingRunEpochMetric.epoch.asc())
                .all()
            )
            if rows:
                epoch_best: Dict[str, float] = {}
                epoch_final: Dict[str, float] = {}
                for row in rows:
                    metrics = row.metrics if isinstance(row.metrics, dict) else {}
                    for key in COMPARE_METRIC_KEYS:
                        n = self._as_number(metrics.get(key))
                        if n is None:
                            continue
                        epoch_final[key] = n
                        prev = epoch_best.get(key)
                        if prev is None or n > prev:
                            epoch_best[key] = n
                if epoch_best or epoch_final:
                    used_epoch = True
                for key in COMPARE_METRIC_KEYS:
                    if key not in best and key in epoch_best:
                        best[key] = epoch_best[key]
                    if key not in final and key in epoch_final:
                        final[key] = epoch_final[key]

        source = None
        if used_result and used_epoch:
            source = "mixed"
        elif used_epoch:
            source = "epoch_fallback"
        elif used_result:
            source = "result"

        return {"best": best, "final": final, "source": source}

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")

    @staticmethod
    def _duration_seconds(started_at: Optional[datetime], finished_at: Optional[datetime]) -> float | None:
        start = _ensure_aware_utc(started_at)
        end = _ensure_aware_utc(finished_at)
        if start is None or end is None:
            return None
        try:
            return max(0.0, round((end - start).total_seconds(), 3))
        except Exception:
            return None

    @classmethod
    def _pick_metric_value(cls, metrics: Dict[str, Any], candidates: tuple[str, ...]) -> float | None:
        if not isinstance(metrics, dict):
            return None
        lowered = {str(k).lower(): k for k in metrics.keys()}
        for key in candidates:
            if key in metrics:
                n = cls._as_number(metrics.get(key))
                if n is not None:
                    return n
            actual = lowered.get(str(key).lower())
            if actual is not None:
                n = cls._as_number(metrics.get(actual))
                if n is not None:
                    return n
        return None

    @classmethod
    def _extract_core_metrics(
        cls,
        best_metrics: Optional[Dict[str, Any]],
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        best = best_metrics if isinstance(best_metrics, dict) else {}
        final = final_metrics if isinstance(final_metrics, dict) else {}
        out: Dict[str, float] = {}
        for label, candidates in REPORT_CORE_METRIC_CANDIDATES.items():
            value = cls._pick_metric_value(best, candidates)
            if value is None:
                value = cls._pick_metric_value(final, candidates)
            if value is not None:
                out[label] = value
        return out

    @staticmethod
    def _save_period_from_params(additional_params: Optional[Dict[str, Any]]) -> int | None:
        add = additional_params if isinstance(additional_params, dict) else {}
        for key in ("save_period", "snapshot_epoch"):
            raw = add.get(key)
            if raw is None:
                continue
            try:
                value = int(raw)
            except Exception:
                continue
            return value
        return None

    @staticmethod
    def _run_image_size(run: TrainingRun) -> int:
        params = getattr(run, "parameters", None)
        raw = getattr(params, "image_size", None) if params is not None else None
        try:
            value = int(float(raw))
        except Exception:
            return 640
        return value if value > 0 else 640

    def _ensure_report_metric_snapshots(self, db: Session, run: TrainingRun) -> TrainingRunResult | None:
        result = run.result
        best_present = bool(result and isinstance(result.best_metrics, dict) and result.best_metrics)
        final_present = bool(result and isinstance(result.final_metrics, dict) and result.final_metrics)
        if best_present and final_present:
            return result

        best_metrics, final_metrics = compute_epoch_metric_snapshots(db, str(run.run_id))
        if best_metrics is None or final_metrics is None:
            return result

        if result is None:
            result = TrainingRunResult(run_id=str(run.run_id))
            db.add(result)
            db.flush()
            run.result = result

        result.best_metrics = best_metrics
        result.final_metrics = final_metrics
        db.add(result)
        db.commit()
        db.refresh(result)
        return result

    def _measure_run_yolo_stats(self, run: TrainingRun, result: TrainingRunResult) -> Dict[str, Any]:
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
                image_path=self._ensure_benchmark_image(),
                imgsz=self._run_image_size(run),
                conf=0.25,
                iou=0.45,
                warmup=1,
                iters=5,
            )
        except Exception:
            # Report artifact statistics are best effort; worker failures should not
            # make an otherwise completed training run unavailable.
            return {}

    def _ensure_report_artifacts(self, db: Session, run: TrainingRun) -> TrainingRunResult | None:
        result = run.result
        needs_index = result is None
        if result is not None:
            needs_index = any(
                value is None
                for value in (
                    result.best_weights_path,
                    result.last_weights_path,
                    result.model_size_mb,
                )
            )
        if needs_index:
            try:
                index_completion_artifacts(db, str(run.run_id))
                db.commit()
                db.refresh(run)
                result = run.result
            except Exception:
                db.rollback()
                result = run.result

        needs_flops = result is not None and (self._as_number(result.flops) in (None, 0))
        needs_latency = result is not None and (self._as_number(result.inference_time_ms) in (None, 0))
        if result is not None and (needs_flops or needs_latency):
            try:
                stats = self._measure_run_yolo_stats(run, result)
                changed = False
                flops = self._as_number(stats.get("flops"))
                if needs_flops and flops and flops > 0:
                    result.flops = int(flops)
                    changed = True
                latency = self._as_number(stats.get("inference_time_ms"))
                if needs_latency and latency and latency > 0:
                    result.inference_time_ms = latency
                    changed = True
                if changed:
                    db.add(result)
                    db.commit()
                    db.refresh(result)
            except Exception:
                db.rollback()

        engine = str(getattr(run.architecture, "engine", "") or "ultralytics-yolo").strip().lower()
        if result is not None and engine != "ultralytics-yolo" and (self._as_number(result.inference_time_ms) in (None, 0)):
            try:
                measured = self._measure_run_inference_latency(
                    db,
                    run=run,
                    benchmark_image=self._ensure_benchmark_image(),
                    conf=0.25,
                    iou=0.45,
                    warmup=1,
                    iters=5,
                )
                result.inference_time_ms = measured
                db.add(result)
                db.commit()
                db.refresh(result)
            except Exception:
                db.rollback()

        return result

    def build_report(self, db: Session, run_id: str) -> Dict[str, Any]:
        run = self._get_run(db, run_id)
        if run.status != TrainingRunStatus.COMPLETED:
            raise ValidationError("训练尚未完成，报告不可用")

        result = self._ensure_report_metric_snapshots(db, run)
        if result is None:
            result = run.result
        result = self._ensure_report_artifacts(db, run) or result

        arch = run.architecture
        if arch is None:
            arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()
        if arch is None:
            raise NotFoundError("Architecture not found")

        dataset = run.standard_dataset
        if dataset is None:
            dataset = (
                db.query(StandardDataset)
                .filter(StandardDataset.standard_dataset_id == int(run.standard_dataset_id))
                .first()
            )

        params = run.parameters
        if params is None:
            raise NotFoundError("Training run parameters not found")

        engine = str(getattr(arch, "engine", "") or "").strip().lower()
        framework_key, framework_label = self._resolve_framework(engine)

        best_metrics = result.best_metrics if result and isinstance(result.best_metrics, dict) else None
        final_metrics = result.final_metrics if result and isinstance(result.final_metrics, dict) else None
        core_metrics = self._extract_core_metrics(best_metrics, final_metrics)

        if not core_metrics:
            # Legacy fallback compatible with the comparison page's historical behavior.
            metric_summary = self._build_metric_summary(db, run)
            core_metrics = self._extract_core_metrics(
                metric_summary.get("best") if isinstance(metric_summary, dict) else None,
                metric_summary.get("final") if isinstance(metric_summary, dict) else None,
            )

        model_size_mb = self._as_number(getattr(result, "model_size_mb", None) if result else None)
        inference_time_ms = self._as_number(getattr(result, "inference_time_ms", None) if result else None)
        if inference_time_ms is not None and inference_time_ms <= 0:
            inference_time_ms = None
        flops = self._as_number(getattr(result, "flops", None) if result else None)
        flops_out = int(flops) if flops is not None and flops > 0 else None

        return {
            "basic": {
                "run_id": str(run.run_id),
                "name": run.name,
                "framework_label": framework_label,
                "framework_key": framework_key,
                "engine": engine,
                "status": self._enum_value(run.status),
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "duration_seconds": self._duration_seconds(run.started_at, run.finished_at),
            },
            "dataset": {
                "dataset_id": int(dataset.standard_dataset_id) if dataset is not None else None,
                "dataset_name": str(dataset.name) if dataset is not None else None,
                # StandardDataset currently has no explicit version column.
                "dataset_version": None,
            },
            "architecture": {
                "architecture_id": int(arch.architecture_id),
                "family": str(getattr(arch, "family", "") or ""),
                "variant": str(getattr(arch, "variant", "") or ""),
                "task_type": self._enum_value(getattr(arch, "task_type", "")),
                "description": getattr(arch, "description", None),
                "pretrained_path": getattr(arch, "pretrained_path", None),
            },
            "parameters": {
                "epochs": int(params.epochs),
                "batch_size": int(params.batch_size),
                "image_size": int(params.image_size),
                "learning_rate": self._as_number(params.learning_rate),
                "lr_scheduler": str(getattr(params, "lr_scheduler", None) or "linear"),
                "patience": int(params.patience),
                "device": str(params.device),
                "workers": int(params.workers),
                "optimizer": str(params.optimizer),
                "use_pretrained": bool(params.use_pretrained),
                "save_period": self._save_period_from_params(params.additional_params),
                "augmentation": params.augmentation if isinstance(params.augmentation, dict) else None,
                "loss_weights": params.loss_weights if isinstance(params.loss_weights, dict) else None,
                "additional_params": params.additional_params if isinstance(params.additional_params, dict) else None,
            },
            "metrics": {
                "best_metrics": best_metrics,
                "final_metrics": final_metrics,
                "core_metrics": core_metrics or None,
            },
            "artifacts": {
                "best_weights_path": result.best_weights_path if result else None,
                "last_weights_path": result.last_weights_path if result else None,
                "model_size_mb": model_size_mb,
                "inference_time_ms": inference_time_ms,
                "flops": flops_out,
            },
        }

    def _ensure_benchmark_image(self) -> Path:
        if Image is None:
            raise ValidationError("Pillow is required for benchmark image generation")
        out_dir = (settings.temp_dir / "benchmark_inputs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthetic_640.jpg"
        if not out_path.exists():
            Image.new("RGB", (640, 640), color=(0, 0, 0)).save(out_path, format="JPEG", quality=95)
        return out_path

    def _measure_run_inference_latency(
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
            config_path = resolve_architecture_config_path(arch)
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
            t0 = time.perf_counter()
            self._worker.execute_model(
                engine=engine,
                weights_path=weights_path,
                image_path=benchmark_image,
                conf=float(conf),
                iou=float(iou),
                config_path=config_path,
            )
            timings.append((time.perf_counter() - t0) * 1000.0)

        return round(float(statistics.median(timings)), 4)

    def benchmark_inference_times(
        self,
        db: Session,
        *,
        run_ids: List[str],
        force: bool = False,
    ) -> Dict[str, Any]:
        ids: List[str] = []
        seen = set()
        for x in run_ids or []:
            s = str(x or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            ids.append(s)
        if not ids:
            raise ValidationError("run_ids is required")
        if len(ids) > 20:
            raise ValidationError("run_ids cannot exceed 20")

        benchmark_image = self._ensure_benchmark_image()
        items: List[Dict[str, Any]] = []

        for rid in ids:
            run: TrainingRun | None = None
            engine: str | None = None
            try:
                run = self._get_run(db, rid)
                engine = str(getattr(run.architecture, "engine", "") or "").strip().lower() or None

                if run.status != TrainingRunStatus.COMPLETED:
                    items.append(
                        {
                            "run_id": rid,
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
                    stats = self._measure_run_yolo_stats(run, run.result)
                    measured = self._as_number(stats.get("inference_time_ms"))
                    if measured is None:
                        raise RuntimeError("YOLO worker did not return inference_time_ms")
                    flops = self._as_number(stats.get("flops"))
                    if flops and flops > 0:
                        run.result.flops = int(flops)
                else:
                    measured = self._measure_run_inference_latency(
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
            except Exception as e:
                db.rollback()
                items.append(
                    {
                        "run_id": str(getattr(run, "run_id", rid)),
                        "status": "failed",
                        "inference_time_ms": None,
                        "engine": engine,
                        "message": f"{type(e).__name__}: {e}",
                    }
                )

        return {"items": items}

    # --------------------
    # compare
    # --------------------
    def compare_runs(self, db: Session, run_ids: List[str]) -> Dict[str, Any]:
        ids: List[str] = []
        seen = set()
        for x in run_ids or []:
            s = str(x or "").strip()
            if not s or s in seen:
                continue
            ids.append(s)
            seen.add(s)

        if len(ids) < 2:
            raise ValidationError("At least 2 distinct run_ids are required for comparison")

        runs_out: List[Dict[str, Any]] = []
        params_by_run: Dict[str, Dict[str, Any]] = {}
        framework_groups: Dict[str, List[str]] = {}

        for rid in ids:
            run = self._get_run(db, rid)
            arch = run.architecture
            engine = str(getattr(arch, "engine", "") or "").strip().lower()
            framework_key, framework_label = self._resolve_framework(engine)
            framework_groups.setdefault(framework_key, []).append(str(run.run_id))

            p: Dict[str, Any] = {}
            if run.parameters is not None:
                p = {
                    "epochs": int(run.parameters.epochs),
                    "batch_size": int(run.parameters.batch_size),
                    "image_size": int(run.parameters.image_size),
                    "learning_rate": float(run.parameters.learning_rate),
                    "lr_scheduler": str(getattr(run.parameters, "lr_scheduler", None) or "linear"),
                    "patience": int(run.parameters.patience),
                    "device": str(run.parameters.device),
                    "workers": int(run.parameters.workers),
                    "use_pretrained": bool(run.parameters.use_pretrained),
                    "optimizer": str(run.parameters.optimizer),
                    "augmentation": run.parameters.augmentation,
                    "loss_weights": run.parameters.loss_weights,
                }
                add = run.parameters.additional_params or {}
                if isinstance(add, dict):
                    for k, v in add.items():
                        if k not in p:
                            p[k] = v

            best_metrics = None
            final_metrics = None
            model_size_mb = None
            inference_time_ms = None
            if run.result is not None:
                best_metrics = run.result.best_metrics
                final_metrics = run.result.final_metrics
                try:
                    model_size_mb = float(run.result.model_size_mb) if run.result.model_size_mb is not None else None
                except Exception:
                    model_size_mb = None
                try:
                    inference_time_ms = float(run.result.inference_time_ms) if run.result.inference_time_ms is not None else None
                    if inference_time_ms is not None and inference_time_ms <= 0:
                        inference_time_ms = None
                except Exception:
                    inference_time_ms = None
            metric_summary = self._build_metric_summary(db, run)

            runs_out.append(
                {
                    "run_id": run.run_id,
                    "name": run.name,
                    "status": run.status,
                    "project_id": int(run.project_id),
                    "standard_dataset_id": int(run.standard_dataset_id),
                    "architecture_id": int(run.architecture_id),
                    "created_at": run.created_at,
                    "engine": engine or None,
                    "framework_key": framework_key,
                    "framework_label": framework_label,
                    "family": str(getattr(arch, "family", "") or "") or None,
                    "variant": str(getattr(arch, "variant", "") or "") or None,
                    "parameters": p,
                    "best_metrics": best_metrics,
                    "final_metrics": final_metrics,
                    "metric_summary": metric_summary,
                    "model_size_mb": model_size_mb,
                    "inference_time_ms": inference_time_ms,
                }
            )
            params_by_run[run.run_id] = p

        if len(framework_groups) > 1:
            grouped = {k: sorted(v) for k, v in framework_groups.items()}
            raise FrameworkCompareConflict("Only runs from the same framework can be compared", grouped)

        def _norm(v: Any) -> str:
            try:
                return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                return str(v)

        all_keys = sorted({k for d in params_by_run.values() for k in d.keys()})
        diff: Dict[str, Dict[str, Any]] = {}
        for k in all_keys:
            vals = {rid: params_by_run[rid].get(k) for rid in params_by_run.keys()}
            if len({_norm(v) for v in vals.values()}) > 1:
                diff[k] = vals

        return {"runs": runs_out, "parameter_diff": diff}
