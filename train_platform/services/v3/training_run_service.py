from __future__ import annotations

import json
import os
import requests
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
from train_platform.utils.training_artifacts import compute_epoch_metric_snapshots, index_completion_artifacts
from train_platform.utils.path_utils import resolve_dataset_path, resolve_training_path
from train_platform.utils.training_augmentations import normalize_training_augmentation
from train_platform.utils.training_loss_weights import normalize_training_loss_weights
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

        try:
            normalized_augmentation = normalize_training_augmentation(
                params.get("augmentation"),
                engine=arch_engine,
                task_type=project.task_type,
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e
        params = dict(params)
        params["augmentation"] = normalized_augmentation

        try:
            normalized_loss_weights = normalize_training_loss_weights(
                params.get("loss_weights"),
                engine=arch_engine,
                task_type=project.task_type,
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e
        params["loss_weights"] = normalized_loss_weights

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
                loss_weights=params.get("loss_weights"),
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

        # 2. Check whether a resumable Ultralytics checkpoint exists.
        # If it does not, we fall back to a fresh restart with the saved parameters.
        weights_path = settings.training_dir / str(run_id) / "weights" / "last.pt"
        has_resume_checkpoint = weights_path.exists()

        # 3. Update parameters to enable resume
        if run.parameters:
            add = run.parameters.additional_params or {}
            if not isinstance(add, dict):
                add = {}
            add["resume_training"] = bool(has_resume_checkpoint)
            add["resume_job_id"] = None  # Clear this to force self-resume semantics for Ultralytics
            run.parameters.additional_params = add
            db.add(run.parameters)

        # 4. Reset status flags
        run.cancel_requested_at = None
        run.cancel_reason = None
        run.error_message = None
        run.finished_at = None
        run.worker_id = None
        run.claimed_at = None
        run.pid = None
        run.heartbeat_at = None
        if not has_resume_checkpoint:
            run.current_epoch = 0
            run.progress = 0
        
        resume_message = (
            "Run resume requested using weights/last.pt"
            if has_resume_checkpoint
            else "No weights/last.pt found; queued run to restart with saved parameters"
        )
        db.add(TrainingRunEvent(run_id=run_id, level=LogLevel.INFO, event_type="resumed", message=resume_message))
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

        worker_url = os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")
        timeout = float(os.getenv("INFERENCE_WORKER_TIMEOUT", "120"))
        payload = {
            "weights_path": str(weights_path),
            "image_path": str(self._ensure_benchmark_image()),
            "imgsz": self._run_image_size(run),
            "conf": 0.25,
            "iou": 0.45,
            "warmup": 1,
            "iters": 5,
        }
        resp = requests.post(
            f"{worker_url}/internal/model-stats/yolo",
            json=payload,
            timeout=timeout,
            headers=InferenceService()._internal_request_headers(),
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        if data.get("error"):
            return {}
        return data if isinstance(data, dict) else {}

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
        run = self.get_run(db, run_id)
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
