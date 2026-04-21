from __future__ import annotations

import json
import os
import shutil
import statistics
import time
import uuid
from pathlib import Path
import yaml
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.enums import LogLevel, TrainingRunStatus
from train_platform.models.v3.inference import InferenceRun
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.project import Project
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunArtifact,
    TrainingRunEpochMetric,
    TrainingRunEvent,
    TrainingRunParameters,
    TrainingRunResult,
)
from train_platform.models.v3.training_run_meta import TrainingRunMeta
from train_platform.repositories.v3.training_run_meta_repo import TrainingRunMetaRepository
from train_platform.repositories.v3.training_run_repo import TrainingRunRepository
from train_platform.training.registry import get_plugin
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.training_artifacts import index_completion_artifacts
from train_platform.utils.path_utils import resolve_dataset_path, resolve_training_path
from train_platform.utils.training_params import validate_training_params_for_engine
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.services.v3.inference_service import InferenceService
from train_platform.services.v3.alarm_service import AlarmService

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


def _safe_remove_dir(path: Path) -> None:
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


class TrainingRunService:
    def __init__(self) -> None:
        self.runs = TrainingRunRepository()
        self.meta_repo = TrainingRunMetaRepository()

    def _try_eval_alarms(self, db: Session, run_ids: List[str]) -> None:
        AlarmService.try_evaluate_training_rules(db, run_ids=run_ids)

    def _stdout_has_completion_marker(self, run_id: str) -> bool:
        marker = f"[train_entry] completed run_id={str(run_id)}"
        path = settings.training_dir / str(run_id) / "logs" / "train.stdout.log"
        tail = _tail_text_file(path, lines=120)
        return marker in tail

    def _maybe_repair_run_status(self, db: Session, run: TrainingRun) -> bool:
        """
        Recover false FAILED/RUNNING status when the worker heartbeat was lost but the training
        subprocess actually completed and wrote its completion marker to stdout.

        Returns True if the run was modified and needs commit.
        """
        if not run:
            return False

        now = _utcnow()
        stale_after = int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120"))
        threshold = now - timedelta(seconds=stale_after)

        msg = str(getattr(run, "error_message", "") or "").strip()
        heartbeat_lost_failed = (
            run.status == TrainingRunStatus.FAILED and msg == "Worker heartbeat lost; marking as failed"
        )
        heartbeat_at = _ensure_aware_utc(getattr(run, "heartbeat_at", None))
        stale_running = run.status == TrainingRunStatus.RUNNING and heartbeat_at is not None and heartbeat_at < threshold

        if not (heartbeat_lost_failed or stale_running):
            return False

        if not self._stdout_has_completion_marker(run.run_id):
            return False

        run.status = TrainingRunStatus.COMPLETED
        run.finished_at = run.finished_at or now
        run.error_message = None
        run.worker_id = None
        run.claimed_at = None
        run.pid = None
        run.heartbeat_at = None
        try:
            run.progress = max(int(getattr(run, "progress", 0) or 0), 100)
        except Exception:
            run.progress = 100

        db.add(
            TrainingRunEvent(
                run_id=str(run.run_id),
                level=LogLevel.INFO,
                event_type="recovered",
                message="Recovered run status to COMPLETED (completion marker found in stdout)",
            )
        )
        try:
            index_completion_artifacts(db, str(run.run_id))
        except Exception:
            pass

        return True

    # --------------------
    # CRUD
    # --------------------
    def get_run(self, db: Session, run_id: str) -> TrainingRun:
        run = self.runs.get(db, str(run_id))
        if not run:
            raise NotFoundError("Training run not found")
        if self._maybe_repair_run_status(db, run):
            db.commit()
            db.refresh(run)
            self._try_eval_alarms(db, [str(run.run_id)])
        return run

    def list_runs(
        self,
        db: Session,
        *,
        project_id: Optional[int] = None,
        status: Optional[TrainingRunStatus] = None,
        standard_dataset_id: Optional[int] = None,
        architecture_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
        include_hidden: bool = False,
    ) -> list[TrainingRun]:
        runs = self.runs.list(
            db,
            project_id=project_id,
            status=status,
            standard_dataset_id=standard_dataset_id,
            architecture_id=architecture_id,
            skip=skip,
            limit=limit,
            include_hidden=include_hidden,
        )

        dirty = False
        for r in runs:
            if self._maybe_repair_run_status(db, r):
                dirty = True
        if dirty:
            db.commit()
            self._try_eval_alarms(db, [str(r.run_id) for r in runs])

        return runs

    def create_run(self, db: Session, *, obj: dict) -> TrainingRun:
        project_id = int(obj["project_id"])
        architecture_id = int(obj["architecture_id"])
        params = obj["parameters"]

        project = db.query(Project).filter(Project.project_id == project_id).first()
        if not project:
            raise NotFoundError("Project not found")

        dataset = (
            db.query(StandardDataset)
            .filter(StandardDataset.standard_dataset_id == int(project.standard_dataset_id))
            .first()
        )
        if not dataset:
            raise NotFoundError("Standard dataset not found")

        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == architecture_id).first()
        if not arch:
            raise NotFoundError("Architecture not found")
        if arch.task_type != project.task_type:
            raise ValidationError("Architecture task_type does not match project task_type")
        arch_engine = str(getattr(arch, "engine", "") or "").strip().lower()
        try:
            plugin = get_plugin(arch_engine)
        except Exception as e:
            raise ValidationError(f"Architecture engine is not registered: {arch_engine}") from e
        if not bool(getattr(plugin, "implemented", True)):
            raise ValidationError(
                f"Architecture engine '{arch_engine}' is not implemented yet; select another framework plugin"
            )

        try:
            params = validate_training_params_for_engine(arch_engine, params)
        except ValueError as e:
            raise ValidationError(str(e)) from e

        additional_params = params.get("additional_params")
        if additional_params is not None and not isinstance(additional_params, dict):
            raise ValidationError("parameters.additional_params must be an object")
        if isinstance(additional_params, dict) and "framework_config" in additional_params:
            framework_config = additional_params.get("framework_config")
            if framework_config is not None and not isinstance(framework_config, dict):
                raise ValidationError("parameters.additional_params.framework_config must be an object")
            try:
                normalized_framework_config = plugin.normalize_config(framework_config or {})
            except Exception as e:
                raise ValidationError(f"Invalid framework_config for engine '{arch_engine}': {e}") from e
            if not isinstance(normalized_framework_config, dict):
                raise ValidationError("framework_config normalize result must be an object")
            params = dict(params)
            params["additional_params"] = dict(additional_params)
            params["additional_params"]["framework_config"] = normalized_framework_config

        # Check dataset train/val split before training (no auto-split).
        has_split = False
        try:
            dataset_root = resolve_dataset_path(dataset.storage_path)
            if not dataset_root.exists() or not dataset_root.is_dir():
                raise ConflictError("Standard dataset path does not exist; upload dataset files first")

            data_yaml = find_yolo_dataset_yaml(dataset_root, dataset_name=str(getattr(dataset, "name", "") or "") or None)
            if data_yaml and data_yaml.exists():
                try:
                    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
                except Exception:
                    cfg = {}
                if isinstance(cfg, dict):
                    train_p = cfg.get("train")
                    val_p = cfg.get("val")

                    def _path_ok(p):
                        if not p:
                            return False
                        if isinstance(p, (list, tuple)):
                            return all(_path_ok(x) for x in p) if p else False
                        s = str(p).strip()
                        if not s:
                            return False
                        pp = Path(s)
                        if not pp.is_absolute():
                            pp = (dataset_root / pp).resolve(strict=False)
                        return pp.exists()

                    if _path_ok(train_p) and _path_ok(val_p):
                        has_split = True
        except ValidationError:
            raise
        except ConflictError:
            raise
        except Exception:
            raise ValidationError("Failed to validate dataset split for training")

        run_id = str(uuid.uuid4())
        name = str(obj.get("name") or "").strip() or f"{arch.variant}-{run_id[:8]}"

        run = TrainingRun(
            run_id=run_id,
            project_id=project.project_id,
            standard_dataset_id=int(dataset.standard_dataset_id),
            architecture_id=arch.architecture_id,
            name=name,
            status=TrainingRunStatus.CREATED,
            progress=0,
            current_epoch=0,
            total_epochs=int(params.get("epochs") or 0) if params else None,
            hidden=False,
            run_dir=run_id,
            config=None,
        )
        db.add(run)
        db.flush()

        db.add(
            TrainingRunParameters(
                run_id=run_id,
                epochs=int(params.get("epochs", 100)),
                batch_size=int(params.get("batch_size", 16)),
                image_size=int(params.get("image_size", 640)),
                learning_rate=float(params.get("learning_rate", 0.01)),
                patience=int(params.get("patience", 50)),
                device=str(params.get("device") or "auto"),
                workers=int(params.get("workers", 8)),
                use_pretrained=bool(params.get("use_pretrained", True)),
                optimizer=str(params.get("optimizer") or "AdamW"),
                augmentation=params.get("augmentation"),
                additional_params=params.get("additional_params"),
            )
        )

        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="created", message="Run created"))
        if not has_split:
            db.add(
                TrainingRunEvent(
                    run_id=run_id,
                    level=LogLevel.WARNING,
                    event_type="dataset_split_missing",
                    message="Standard dataset has no valid train/val split; proceeding without enforced split",
                )
            )
        db.commit()

        return self.get_run(db, run_id)

    def update_run(self, db: Session, run_id: str, *, patch: dict) -> TrainingRun:
        run = self.get_run(db, run_id)
        if "name" in patch and patch["name"] is not None:
            run.name = str(patch["name"]).strip()
        db.commit()
        db.refresh(run)
        return run

    # --------------------
    # queue control
    # --------------------
    def queue_run(self, db: Session, run_id: str) -> TrainingRun:
        run = self.get_run(db, run_id)

        if run.status in (TrainingRunStatus.RUNNING, TrainingRunStatus.COMPLETED):
            raise ConflictError(f"Run status is {run.status}; cannot queue")
        if run.status == TrainingRunStatus.DELETED:
            raise ConflictError("Run is deleted")

        if run.queued_at is None:
            run.queued_at = _utcnow()
        run.hidden = False
        run.status = TrainingRunStatus.QUEUED
        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="queued", message="Run queued"))
        db.commit()
        db.refresh(run)
        self._try_eval_alarms(db, [str(run.run_id)])
        return run

    def resume_run(self, db: Session, run_id: str) -> TrainingRun:
        run = self.get_run(db, run_id)

        if run.status == TrainingRunStatus.COMPLETED:
            raise ConflictError("Run is COMPLETED and cannot be resumed; create a new training run instead")

        # 1. Validation: only allow resuming cancelled or failed runs.
        if run.status not in (TrainingRunStatus.CANCELLED, TrainingRunStatus.FAILED):
            raise ConflictError(f"Run status is {run.status}; must be CANCELLED or FAILED to resume")

        # 2. Check for weights
        weights_path = settings.training_dir / str(run_id) / "weights" / "last.pt"
        if not weights_path.exists():
            raise ConflictError("Cannot resume: 'last.pt' weights not found in run directory")

        # 3. Update parameters to enable resume
        if run.parameters:
            add = run.parameters.additional_params or {}
            if isinstance(add, dict):
                add["resume_training"] = True
                add["resume_job_id"] = None # Clear this to force self-resume
                run.parameters.additional_params = add
                db.add(run.parameters)
        
        # 4. Reset status flags
        run.cancel_requested_at = None
        run.cancel_reason = None
        run.error_message = None
        run.finished_at = None
        
        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="resumed", message="Run resume requested"))
        db.commit()

        # 5. Queue it
        return self.queue_run(db, run_id)


    def request_cancel(self, db: Session, run_id: str, *, reason: Optional[str] = None) -> TrainingRun:
        run = self.get_run(db, run_id)
        if run.cancel_requested_at is None:
            run.cancel_requested_at = _utcnow()
        if reason:
            run.cancel_reason = str(reason)

        # If not started yet, cancel immediately.
        if run.status in (TrainingRunStatus.CREATED, TrainingRunStatus.QUEUED):
            run.status = TrainingRunStatus.CANCELLED
            run.finished_at = _utcnow()

        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="cancel_requested", message=reason or "Cancel requested"))
        db.commit()
        db.refresh(run)
        self._try_eval_alarms(db, [str(run.run_id)])
        return run

    def request_delete(self, db: Session, run_id: str) -> TrainingRun:
        run = self.get_run(db, run_id)

        run.hidden = True
        if run.delete_requested_at is None:
            run.delete_requested_at = _utcnow()
        if run.cancel_requested_at is None:
            run.cancel_requested_at = _utcnow()

        # If already terminal, mark deleted immediately.
        if run.status in (TrainingRunStatus.COMPLETED, TrainingRunStatus.FAILED, TrainingRunStatus.CANCELLED):
            run.status = TrainingRunStatus.DELETED
            run.finished_at = run.finished_at or _utcnow()

        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="delete_requested", message="Delete requested"))
        db.commit()
        db.refresh(run)
        self._try_eval_alarms(db, [str(run.run_id)])
        return run

    def delete_run(self, db: Session, run_id: str, *, force: bool = False) -> TrainingRun:
        run = self.get_run(db, run_id)

        model_versions = db.query(ModelVersion).filter(ModelVersion.run_id == str(run.run_id)).all()
        if model_versions and not force:
            detail = f"{len(model_versions)} model version(s)"
            raise ConflictError(f"Cannot delete training run; {detail} still reference it")

        if model_versions and force:
            mv_ids = [int(m.model_version_id) for m in model_versions]
            dep_ids: list[int] = []
            if mv_ids:
                deployments = db.query(Deployment).filter(Deployment.model_version_id.in_(mv_ids)).all()
                dep_ids = [int(d.deployment_id) for d in deployments]
                inf_filters = [InferenceRun.model_version_id.in_(mv_ids)]
                if dep_ids:
                    inf_filters.append(InferenceRun.deployment_id.in_(dep_ids))
                for inf in db.query(InferenceRun).filter(or_(*inf_filters)).all():
                    db.delete(inf)
                for dep in deployments:
                    db.delete(dep)
            for mv in model_versions:
                db.delete(mv)

        run = self.request_delete(db, str(run.run_id))
        if run.status != TrainingRunStatus.RUNNING:
            _safe_remove_dir(settings.training_dir / str(run.run_id))
        return run

    # --------------------
    # metrics/events/artifacts
    # --------------------
    def list_events(self, db: Session, run_id: str, *, limit: int = 200) -> list[TrainingRunEvent]:
        self.get_run(db, run_id)
        return (
            db.query(TrainingRunEvent)
            .filter(TrainingRunEvent.run_id == str(run_id))
            .order_by(TrainingRunEvent.created_at.desc())
            .limit(int(limit))
            .all()
        )

    def list_epoch_metrics(self, db: Session, run_id: str, *, limit: int = 5000) -> list[TrainingRunEpochMetric]:
        self.get_run(db, run_id)
        return (
            db.query(TrainingRunEpochMetric)
            .filter(TrainingRunEpochMetric.run_id == str(run_id))
            .order_by(TrainingRunEpochMetric.epoch.asc())
            .limit(int(limit))
            .all()
        )

    def list_artifacts(self, db: Session, run_id: str) -> list[TrainingRunArtifact]:
        self.get_run(db, run_id)
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
        self.get_run(db, run_id)
        meta = self.meta_repo.get_by_run_id(db, run_id)
        if meta:
            return meta

        meta = TrainingRunMeta(run_id=str(run_id))
        db.add(meta)
        db.commit()
        db.refresh(meta)
        return meta

    def update_meta(self, db: Session, run_id: str, *, patch: dict) -> TrainingRunMeta:
        self.get_run(db, run_id)
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

    # --------------------
    # logs
    # --------------------
    def tail_logs(self, db: Session, run_id: str, *, which: str = "stdout", lines: int = 200) -> str:
        """
        Best-effort tail of worker-produced logs.

        `which`: stdout | stderr
        """
        self.get_run(db, run_id)

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

        infer = InferenceService()
        config_path = None
        if engine == "paddle-det":
            config_path = infer._resolve_paddle_config_path(arch)
            if not config_path:
                raise ValidationError("Paddle model missing valid config_path")

        warmup = max(0, int(warmup))
        iters = max(1, int(iters))

        for _ in range(warmup):
            infer._run_by_engine(
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
            infer._run_by_engine(
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
                run = self.get_run(db, rid)
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
                if cached is not None and not force:
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

                measured = self._measure_run_inference_latency(
                    db,
                    run=run,
                    benchmark_image=benchmark_image,
                    conf=0.25,
                    iou=0.45,
                    warmup=1,
                    iters=5,
                )

                if run.result is None:
                    run.result = TrainingRunResult(run_id=str(run.run_id))
                    db.add(run.result)
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
            run = self.get_run(db, rid)
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
                    "patience": int(run.parameters.patience),
                    "device": str(run.parameters.device),
                    "workers": int(run.parameters.workers),
                    "use_pretrained": bool(run.parameters.use_pretrained),
                    "optimizer": str(run.parameters.optimizer),
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
