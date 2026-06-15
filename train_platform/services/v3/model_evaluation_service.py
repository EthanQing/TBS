from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.v3.enums import DatasetSplit, DatasetType, ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.models.v3.training_run import TrainingRun
from train_platform.schemas.v3.inference_jobs import InferenceModelCandidate
from train_platform.schemas.v3.model_evaluations import ModelEvaluationCreate, ModelEvaluationOut
from train_platform.services.v3.dataset_common import guess_label_path, read_class_names, read_yolo_boxes, resolve_storage_token
from train_platform.services.v3.inference_job_service import InferenceJobService
from train_platform.services.v3.inference_service import InferenceService
from train_platform.services.v3.model_evaluation_metrics import compute_detection_metrics
from train_platform.services.v3.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


_LOCKS_GUARD = threading.Lock()
_JOB_LOCKS: Dict[str, threading.Lock] = {}
_ACTIVE_CREATE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utcnow()).isoformat()


def _parse_time(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValidationError("Evaluation job not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValidationError(f"Failed to read evaluation status: {type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("Invalid evaluation status payload")
    return data


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = dict(data or {})
    payload["updated_at"] = _to_iso()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _job_lock(job_id: str) -> threading.Lock:
    key = str(job_id)
    with _LOCKS_GUARD:
        lock = _JOB_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _JOB_LOCKS[key] = lock
        return lock


class ModelEvaluationService:
    ACTIVE_STATUSES = {"queued", "running"}
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    ACTIVE_STALE_AFTER = timedelta(hours=4)

    def __init__(self) -> None:
        self._infer = InferenceService()
        self._mv_svc = ModelVersionService()
        self._model_listing = InferenceJobService()

    def jobs_root(self) -> Path:
        root = settings.temp_dir / "model_evaluations"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def job_dir(self, job_id: str) -> Path:
        d = self.jobs_root() / str(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def results_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "results.jsonl"

    def list_evaluable_models(self, db: Session, *, project_id: int | None = None) -> List[InferenceModelCandidate]:
        return self._model_listing.list_inferable_models(db, project_id=project_id)

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _has_active_job(self) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        for status_file in self.jobs_root().glob("*/status.json"):
            try:
                data = _read_json(status_file)
            except Exception:
                continue
            status = str(data.get("status") or "").strip().lower()
            if status not in self.ACTIVE_STATUSES:
                continue
            updated = _parse_time(data.get("updated_at"))
            if updated and (now - updated) > self.ACTIVE_STALE_AFTER:
                continue
            return data
        return None

    def get_active_job(self, *, include_items: bool = False) -> Optional[ModelEvaluationOut]:
        active = self._has_active_job()
        if not active:
            return None
        job_id = str(active.get("job_id") or "").strip()
        if not job_id:
            return None
        return self.get_job(job_id, include_items=include_items)

    def _ensure_model_version_for_payload(
        self,
        db: Session,
        *,
        model_version_id: int | None,
        run_id: str | None,
    ) -> int:
        if model_version_id is not None:
            row = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
            if not row:
                raise NotFoundError("Model version not found")
            return int(row.model_version_id)

        rid = str(run_id or "").strip()
        if not rid:
            raise ValidationError("Missing model_version_id/run_id")

        existing = (
            db.query(ModelVersion)
            .filter(ModelVersion.run_id == rid)
            .order_by(ModelVersion.created_at.desc(), ModelVersion.model_version_id.desc())
            .first()
        )
        if existing:
            return int(existing.model_version_id)

        run = db.query(TrainingRun).filter(TrainingRun.run_id == rid).first()
        if not run:
            raise NotFoundError("Training run not found")
        if run.status != TrainingRunStatus.COMPLETED:
            raise ConflictError("Only completed runs can be evaluated")

        base = f"run-{rid[:8]}"
        for i in range(1, 200):
            version = base if i == 1 else f"{base}-{i}"
            try:
                mv = self._mv_svc.register_from_run(
                    db,
                    run_id=rid,
                    version=version,
                    stage=ModelStage.DEVELOPMENT,
                    description="Auto-created for model evaluation jobs",
                )
                return int(mv.model_version_id)
            except ConflictError:
                continue
        raise ConflictError("Failed to auto-register model version for evaluation")

    def create_job(self, db: Session, payload: ModelEvaluationCreate) -> ModelEvaluationOut:
        with _ACTIVE_CREATE_LOCK:
            active = self._has_active_job()
            if active:
                jid = active.get("job_id") or "unknown"
                st = active.get("status") or "running"
                raise ConflictError(f"Another evaluation job is active (job_id={jid}, status={st})")

            dataset = self._resolve_dataset(db, payload.standard_dataset_id)
            mv_id = self._ensure_model_version_for_payload(
                db,
                model_version_id=payload.model_version_id,
                run_id=payload.run_id,
            )
            ctx = self._infer.resolve_model_context(db, model_version_id=mv_id)
            rows = self._select_image_rows(db, dataset, scope=payload.scope)
            if not rows:
                raise ValidationError("No images found for selected evaluation scope")

            job_id = self._new_job_id()
            self.job_dir(job_id)
            status: Dict[str, Any] = {
                "job_id": job_id,
                "status": "queued",
                "phase": "preparing",
                "progress": 0,
                "processed": 0,
                "total": int(len(rows)),
                "seq": 1,
                "last_result_id": 0,
                "model_version_id": int(mv_id),
                "run_id": str(ctx.get("run_id") or payload.run_id or ""),
                "standard_dataset_id": int(dataset.standard_dataset_id),
                "dataset_name": str(dataset.name),
                "scope": str(payload.scope),
                "conf": float(payload.conf),
                "iou": float(payload.iou),
                "engine": str(ctx.get("engine") or ""),
                "family": ctx.get("family"),
                "variant": ctx.get("variant"),
                "cancel_requested": False,
                "result": {"metrics": None},
                "error_message": None,
                "created_at": _to_iso(),
                "updated_at": _to_iso(),
            }
            _write_json_atomic(self.status_path(job_id), status)

        thread = threading.Thread(target=self._run_job_thread, args=(job_id,), daemon=True)
        thread.start()
        return self.get_job(job_id, include_items=False)

    def _resolve_dataset(self, db: Session, standard_dataset_id: int) -> StandardDataset:
        dataset = db.query(StandardDataset).filter(StandardDataset.standard_dataset_id == int(standard_dataset_id)).first()
        if not dataset:
            raise NotFoundError("Standard dataset not found")
        if dataset.dataset_type != DatasetType.DETECTION:
            raise ValidationError("Model evaluation currently supports detection standard datasets only")
        if str(dataset.format or "").strip().lower() != "yolo":
            raise ValidationError("Model evaluation currently supports YOLO standard datasets only")
        root = resolve_storage_token(dataset.storage_path)
        if not root.exists() or not root.is_dir():
            raise NotFoundError(f"Dataset files not found: {root}")
        return dataset

    def _select_image_rows(self, db: Session, dataset: StandardDataset, *, scope: str) -> list[StandardDatasetImage]:
        q = db.query(StandardDatasetImage).filter(
            StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
        )
        scope_norm = str(scope or "all").strip().lower()
        if scope_norm == "train":
            q = q.filter(StandardDatasetImage.split == DatasetSplit.TRAIN)
        elif scope_norm == "val":
            q = q.filter(StandardDatasetImage.split == DatasetSplit.VAL)
        elif scope_norm == "test":
            q = q.filter(StandardDatasetImage.split == DatasetSplit.TEST)
        elif scope_norm != "all":
            raise ValidationError("scope must be one of: all, test, val, train")
        return q.order_by(StandardDatasetImage.path.asc()).all()

    def _read_status(self, job_id: str) -> Dict[str, Any]:
        return _read_json(self.status_path(job_id))

    def _update_status(self, job_id: str, patch: Dict[str, Any], *, bump_seq: bool = True) -> Dict[str, Any]:
        lock = _job_lock(job_id)
        with lock:
            current = self._read_status(job_id)
            current.update(dict(patch or {}))
            current["progress"] = max(0, min(100, int(current.get("progress") or 0)))
            current["processed"] = max(0, int(current.get("processed") or 0))
            current["total"] = max(0, int(current.get("total") or 0))
            if bump_seq:
                current["seq"] = int(current.get("seq") or 0) + 1
            _write_json_atomic(self.status_path(job_id), current)
            return current

    def _append_item(self, job_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        lock = _job_lock(job_id)
        with lock:
            status = self._read_status(job_id)
            rid = int(status.get("last_result_id") or 0) + 1
            row = dict(item or {})
            row["result_id"] = rid
            with open(self.results_path(job_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            status["last_result_id"] = rid
            status["seq"] = int(status.get("seq") or 0) + 1
            _write_json_atomic(self.status_path(job_id), status)
            return row

    def read_results_since(self, job_id: str, after_result_id: int = 0) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        p = self.results_path(job_id)
        if not p.exists():
            return out
        last = int(after_result_id or 0)
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    txt = line.strip()
                    if not txt:
                        continue
                    obj = json.loads(txt)
                    rid = int(obj.get("result_id") or 0)
                    if rid <= last:
                        continue
                    out.append(obj)
        except Exception:
            return out
        out.sort(key=lambda x: int(x.get("result_id") or 0))
        return out

    def _is_cancel_requested(self, job_id: str) -> bool:
        try:
            return bool(self._read_status(job_id).get("cancel_requested"))
        except Exception:
            return True

    def cancel_job(self, job_id: str) -> ModelEvaluationOut:
        st = self._update_status(job_id, {"cancel_requested": True}, bump_seq=True)
        if str(st.get("status")) == "queued":
            self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
        return self.get_job(job_id, include_items=False)

    def get_job(self, job_id: str, *, include_items: bool = True) -> ModelEvaluationOut:
        st = self._read_status(job_id)
        result = st.get("result") if isinstance(st.get("result"), dict) else {"metrics": None}
        if include_items:
            result = dict(result or {})
            result["items"] = self.read_results_since(job_id, after_result_id=0)
        payload = dict(st)
        payload["result"] = result
        return ModelEvaluationOut.model_validate(payload)

    def _run_job_thread(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            self._run_job(db, job_id)
        except Exception as e:
            try:
                self._update_status(
                    job_id,
                    {
                        "status": "failed",
                        "phase": "failed",
                        "progress": 100,
                        "error_message": f"{type(e).__name__}: {e}",
                    },
                    bump_seq=True,
                )
            except Exception:
                pass
        finally:
            db.close()

    def _run_job(self, db: Session, job_id: str) -> None:
        status = self._read_status(job_id)
        dataset = self._resolve_dataset(db, int(status["standard_dataset_id"]))
        root = resolve_storage_token(dataset.storage_path)
        rows = self._select_image_rows(db, dataset, scope=str(status.get("scope") or "all"))
        ctx = self._infer.resolve_model_context(db, model_version_id=int(status["model_version_id"]))
        class_names = read_class_names(root)

        total = len(rows)
        start = time.perf_counter()
        skipped = 0
        failed = 0
        evaluated = 0
        gt_by_image: dict[str, list[dict[str, Any]]] = {}
        pred_by_image: dict[str, list[dict[str, Any]]] = {}

        self._update_status(
            job_id,
            {"status": "running", "phase": "inferring", "total": total, "processed": 0, "progress": 0},
            bump_seq=True,
        )

        for idx, row in enumerate(rows, start=1):
            if self._is_cancel_requested(job_id):
                self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
                return

            rel_path = str(row.path or "")
            image_path = root / rel_path
            label_path = guess_label_path(root, rel_path)
            progress = int((idx / total) * 100) if total else 100
            base_item = {
                "filename": Path(rel_path).name,
                "image_path": rel_path,
            }

            if not image_path.exists() or not image_path.is_file():
                skipped += 1
                self._append_item(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "Image file not found"},
                )
                self._update_status(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            if not label_path.exists() or not label_path.is_file():
                skipped += 1
                self._append_item(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "YOLO label file not found"},
                )
                self._update_status(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            _w, _h, gt_boxes = read_yolo_boxes(root, rel_path, class_names)
            if not gt_boxes:
                skipped += 1
                self._append_item(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "No valid YOLO boxes"},
                )
                self._update_status(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            t0 = time.perf_counter()
            try:
                output = self._infer._run_by_engine(
                    engine=str(ctx.get("engine") or "ultralytics-yolo"),
                    weights_path=Path(str(ctx["weights_path"])),
                    image_path=image_path,
                    conf=float(status.get("conf") or 0.25),
                    iou=float(status.get("iou") or 0.5),
                    config_path=Path(str(ctx["config_path"])) if ctx.get("config_path") else None,
                )
                preds = output.get("predictions") if isinstance(output, dict) else []
                preds = preds if isinstance(preds, list) else []
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                image_id = rel_path
                gt_by_image[image_id] = gt_boxes
                pred_by_image[image_id] = preds
                evaluated += 1
                self._append_item(
                    job_id,
                    {
                        **base_item,
                        "status": "success",
                        "gt_count": len(gt_boxes),
                        "prediction_count": len(preds),
                        "inference_time_ms": elapsed_ms,
                    },
                )
            except Exception as e:
                failed += 1
                self._append_item(
                    job_id,
                    {
                        **base_item,
                        "status": "failed",
                        "gt_count": len(gt_boxes),
                        "prediction_count": 0,
                        "error_message": f"{type(e).__name__}: {e}",
                    },
                )

            self._update_status(job_id, {"processed": idx, "progress": progress}, bump_seq=True)

        if evaluated <= 0:
            raise ValidationError("No labeled images were available for evaluation")

        self._update_status(job_id, {"phase": "calculating", "progress": 99}, bump_seq=True)
        metrics = compute_detection_metrics(
            gt_by_image,
            pred_by_image,
            iou_threshold=float(status.get("iou") or 0.5),
            class_names=class_names,
        )
        metrics.update(
            {
                "evaluated_images": int(evaluated),
                "skipped_images": int(skipped),
                "failed_images": int(failed),
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
            }
        )
        self._update_status(
            job_id,
            {
                "status": "completed",
                "phase": "done",
                "progress": 100,
                "result": {"metrics": metrics},
                "error_message": None,
            },
            bump_seq=True,
        )
