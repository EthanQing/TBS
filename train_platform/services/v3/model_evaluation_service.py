from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.models.v3.enums import DatasetSplit, DatasetType, ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.jobs import JobNotFoundError, JobStatus, JobStore, JobStoreError, is_active_status
from train_platform.schemas.v3.model_evaluations import ModelEvaluationCreate, ModelEvaluationOut
from train_platform.services.v3.dataset_common import guess_label_path, read_class_names, read_yolo_boxes, resolve_storage_token
from train_platform.services.v3.inference_service import InferenceService
from train_platform.services.v3.model_evaluation_metrics import compute_detection_metrics
from train_platform.services.v3.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


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


def _is_valid_yolo_label(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in lines:
        parts = [p for p in line.strip().split() if p]
        if len(parts) < 5:
            continue
        try:
            int(float(parts[0]))
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
            return True
        except Exception:
            continue
    return False


class ModelEvaluationService:
    ACTIVE_STALE_AFTER = timedelta(hours=4)

    def __init__(self) -> None:
        self._infer = InferenceService()
        self._mv_svc = ModelVersionService()

    def jobs_root(self) -> Path:
        root = settings.temp_dir / "model_evaluations"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _store(self) -> JobStore:
        return JobStore(self.jobs_root())

    def job_dir(self, job_id: str) -> Path:
        return self._store().job_dir(job_id, create=True)

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _has_active_job(self) -> Optional[Dict[str, Any]]:
        now = _utcnow()
        try:
            statuses = self._store().list_statuses()
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation jobs: {exc}") from exc
        for data in statuses:
            if not is_active_status(data.get("status")):
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
                st = active.get("status") or JobStatus.RUNNING.value
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
            root = resolve_storage_token(dataset.storage_path)
            labeled_rows, skipped_images = self._select_labeled_image_rows(root, rows)
            if not labeled_rows:
                raise ValidationError("No labeled images were available for evaluation")

            job_id = self._new_job_id()
            self.job_dir(job_id)
            status: Dict[str, Any] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "phase": "preparing",
                "progress": 0,
                "processed": 0,
                "total": int(len(labeled_rows)),
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
                "skipped_images": int(skipped_images),
                "result": {"metrics": None},
                "error_message": None,
                "created_at": _to_iso(),
                "updated_at": _to_iso(),
            }
            status = self._store().create(job_id, status)

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

    def _select_labeled_image_rows(
        self,
        root: Path,
        rows: list[str],
    ) -> tuple[list[str], int]:
        labeled: list[str] = []
        skipped = 0
        for row in rows:
            rel_path = getattr(row, "path", row)
            rel_path = str(rel_path or "")
            if not rel_path:
                skipped += 1
                continue
            image_path = root / rel_path
            label_path = guess_label_path(root, rel_path)
            if not image_path.exists() or not image_path.is_file() or not label_path.exists() or not label_path.is_file():
                skipped += 1
                continue
            if not _is_valid_yolo_label(label_path):
                skipped += 1
                continue
            labeled.append(rel_path)
        return labeled, skipped

    def _write_ultralytics_eval_data_yaml(
        self,
        job_id: str,
        root: Path,
        rows: list[str],
        class_names: list[str],
    ) -> Path:
        job_root = self.job_dir(job_id)
        images_txt = job_root / "eval_images.txt"
        data_yaml = job_root / "eval_data.yaml"
        with images_txt.open("w", encoding="utf-8") as f:
            for row in rows:
                rel_path = getattr(row, "path", row)
                rel_path = str(rel_path or "")
                if not rel_path:
                    continue
                f.write((root / rel_path).resolve(strict=False).as_posix() + "\n")

            names: list[str] = list(class_names or [])
        if not names:
            max_class_id = -1
            for row in rows:
                rel_path = getattr(row, "path", row)
                label_path = guess_label_path(root, str(rel_path or ""))
                try:
                    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = [p for p in line.strip().split() if p]
                        if len(parts) >= 5:
                            max_class_id = max(max_class_id, int(float(parts[0])))
                except Exception:
                    continue
            names = [str(i) for i in range(max_class_id + 1)] if max_class_id >= 0 else ["0"]

        payload = {
            "path": root.resolve(strict=False).as_posix(),
            "train": images_txt.resolve(strict=False).as_posix(),
            "val": images_txt.resolve(strict=False).as_posix(),
            "test": images_txt.resolve(strict=False).as_posix(),
            "names": names,
            "nc": len(names),
        }
        data_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return data_yaml

    def read_results_since(self, job_id: str, after_result_id: int = 0) -> List[Dict[str, Any]]:
        try:
            return self._store().read_results_since(job_id, after_result_id=after_result_id)
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation job results: {exc}") from exc

    def _is_cancel_requested(self, job_id: str) -> bool:
        return bool(self._read_job_status(job_id).get("cancel_requested"))

    def _read_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            return self._store().read_status(job_id)
        except JobNotFoundError as exc:
            raise ValidationError("Evaluation job not found") from exc
        except JobStoreError as exc:
            raise ValidationError(f"Failed to read evaluation status: {exc}") from exc

    def cancel_job(self, job_id: str) -> ModelEvaluationOut:
        try:
            self._store().cancel(
                job_id,
                terminal_if=(JobStatus.QUEUED, JobStatus.RUNNING),
                terminal_patch={"phase": "cancelled", "error_message": None},
            )
        except JobStoreError as exc:
            raise ValidationError(f"Failed to cancel evaluation job: {exc}") from exc
        return self.get_job(job_id, include_items=False)

    def get_job(self, job_id: str, *, include_items: bool = True) -> ModelEvaluationOut:
        st = self._read_job_status(job_id)
        result = st.get("result") if isinstance(st.get("result"), dict) else {"metrics": None}
        if include_items:
            result = dict(result or {})
            result["items"] = self.read_results_since(job_id, after_result_id=0)
        payload = dict(st)
        payload["result"] = result
        return ModelEvaluationOut.model_validate(payload)

    def _run_job_thread(self, job_id: str) -> None:
        try:
            self._run_job(job_id)
        except Exception as e:
            try:
                self._store().update(
                    job_id,
                    {
                        "status": JobStatus.FAILED,
                        "phase": "failed",
                        "progress": 100,
                        "error_message": f"{type(e).__name__}: {e}",
                    },
                    bump_seq=True,
                )
            except Exception:
                pass

    def _prepare_job_snapshot(self, job_id: str) -> dict[str, Any]:
        status = self._read_job_status(job_id)
        with session_scope() as db:
            dataset = self._resolve_dataset(db, int(status["standard_dataset_id"]))
            root = resolve_storage_token(dataset.storage_path)
            rows = self._select_image_rows(db, dataset, scope=str(status.get("scope") or "all"))
            row_paths = [str(row.path or "") for row in rows]
            ctx = self._infer.resolve_model_context(db, model_version_id=int(status["model_version_id"]))
        return {
            "status": status,
            "root": root,
            "row_paths": row_paths,
            "model_context": ctx,
        }

    def _run_job(self, job_id: str) -> None:
        snapshot = self._prepare_job_snapshot(job_id)
        status = snapshot["status"]
        root = snapshot["root"]
        rows = snapshot["row_paths"]
        ctx = snapshot["model_context"]
        class_names = read_class_names(root)
        labeled_rows, skipped_by_label_scan = self._select_labeled_image_rows(root, rows)

        total = len(labeled_rows)
        start = time.perf_counter()
        skipped = int(status.get("skipped_images") or skipped_by_label_scan)
        failed = 0
        evaluated = 0
        gt_by_image: dict[str, list[dict[str, Any]]] = {}
        pred_by_image: dict[str, list[dict[str, Any]]] = {}

        if total <= 0:
            raise ValidationError("No labeled images were available for evaluation")

        self._store().update(
            job_id,
            {"status": JobStatus.RUNNING, "phase": "validating", "total": total, "processed": 0, "progress": 1},
            bump_seq=True,
        )

        if str(ctx.get("engine") or "").strip().lower() == "ultralytics-yolo":
            if self._is_cancel_requested(job_id):
                self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
                return
            data_yaml = self._write_ultralytics_eval_data_yaml(job_id, root, labeled_rows, class_names)
            self._store().update(
                job_id,
                {"phase": "calculating", "progress": 5},
                bump_seq=True,
            )
            metrics = self._infer.run_ultralytics_yolo_validation(
                weights_path=Path(str(ctx["weights_path"])),
                data_yaml=data_yaml,
                conf=float(status.get("conf") or 0.25),
                iou=float(status.get("iou") or 0.5),
            )
            if self._is_cancel_requested(job_id):
                self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
                return
            metrics.update(
                {
                    "evaluated_images": int(total),
                    "skipped_images": int(skipped),
                    "failed_images": 0,
                    "elapsed_ms": float(metrics.get("elapsed_ms") or round((time.perf_counter() - start) * 1000.0, 2)),
                }
            )
            self._store().update(
                job_id,
                {
                    "status": JobStatus.COMPLETED,
                    "phase": "done",
                    "progress": 100,
                    "processed": int(total),
                    "total": int(total),
                    "result": {"metrics": metrics},
                    "error_message": None,
                },
                bump_seq=True,
            )
            return

        self._store().update(
            job_id,
            {"phase": "inferring", "processed": 0, "progress": 1},
            bump_seq=True,
        )

        for idx, rel_path in enumerate(labeled_rows, start=1):
            if self._is_cancel_requested(job_id):
                self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
                return

            rel_path = str(rel_path or "")
            image_path = root / rel_path
            label_path = guess_label_path(root, rel_path)
            progress = int((idx / total) * 100) if total else 100
            base_item = {
                "filename": Path(rel_path).name,
                "image_path": rel_path,
            }

            if not image_path.exists() or not image_path.is_file():
                skipped += 1
                self._store().append_result(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "Image file not found"},
                )
                self._store().update(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            if not label_path.exists() or not label_path.is_file():
                skipped += 1
                self._store().append_result(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "YOLO label file not found"},
                )
                self._store().update(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            _w, _h, gt_boxes = read_yolo_boxes(root, rel_path, class_names)
            if not gt_boxes:
                skipped += 1
                self._store().append_result(
                    job_id,
                    {**base_item, "status": "skipped", "gt_count": 0, "prediction_count": 0, "error_message": "No valid YOLO boxes"},
                )
                self._store().update(job_id, {"processed": idx, "progress": progress}, bump_seq=True)
                continue

            t0 = time.perf_counter()
            try:
                if self._is_cancel_requested(job_id):
                    self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
                    return
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
                self._store().append_result(
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
                self._store().append_result(
                    job_id,
                    {
                        **base_item,
                        "status": "failed",
                        "gt_count": len(gt_boxes),
                        "prediction_count": 0,
                        "error_message": f"{type(e).__name__}: {e}",
                    },
                )

            if self._is_cancel_requested(job_id):
                self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
                return
            self._store().update(job_id, {"processed": idx, "progress": progress}, bump_seq=True)

        if evaluated <= 0:
            raise ValidationError("No labeled images were available for evaluation")

        if self._is_cancel_requested(job_id):
            self._store().update(job_id, {"status": JobStatus.CANCELLED, "phase": "cancelled"}, bump_seq=True)
            return

        self._store().update(job_id, {"phase": "calculating", "progress": 99}, bump_seq=True)
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
        self._store().update(
            job_id,
            {
                "status": JobStatus.COMPLETED,
                "phase": "done",
                "progress": 100,
                "result": {"metrics": metrics},
                "error_message": None,
            },
            bump_seq=True,
        )
