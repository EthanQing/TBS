from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.schemas.v3.dataset_augmentations import (
    DatasetAugmentationCancelOut,
    DatasetAugmentationConfig,
    DatasetAugmentationCreate,
    DatasetAugmentationJobOut,
    DatasetAugmentationPreviewOut,
    DatasetAugmentationPreviewRequest,
    DatasetAugmentationPublishIn,
    DatasetAugmentationPublishOut,
)
from train_platform.services.v3.dataset_common import copy_tree, count_tree, iter_image_files, read_class_names, resolve_storage_token
from train_platform.services.v3.standard_dataset_service import StandardDatasetService
from train_platform.utils.exceptions import ConflictError, NotFoundError


class DatasetAugmentationService:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._datasets = StandardDatasetService()

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def jobs_root(self, standard_dataset_id: int) -> Path:
        root = settings.temp_dir / "dataset_augmentations" / str(int(standard_dataset_id))
        root.mkdir(parents=True, exist_ok=True)
        return root

    def job_dir(self, standard_dataset_id: int, job_id: str) -> Path:
        path = self.jobs_root(standard_dataset_id) / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def status_path(self, standard_dataset_id: int, job_id: str) -> Path:
        return self.job_dir(standard_dataset_id, job_id) / "status.json"

    def output_dir(self, standard_dataset_id: int, job_id: str) -> Path:
        path = self.job_dir(standard_dataset_id, job_id) / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _lock(self, standard_dataset_id: int, job_id: str) -> threading.Lock:
        key = f"{int(standard_dataset_id)}:{str(job_id)}"
        self._locks.setdefault(key, threading.Lock())
        return self._locks[key]

    def _write_status(self, standard_dataset_id: int, job_id: str, payload: dict[str, Any]) -> None:
        path = self.status_path(standard_dataset_id, job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)

    def _read_status(self, standard_dataset_id: int, job_id: str) -> dict[str, Any]:
        path = self.status_path(standard_dataset_id, job_id)
        if not path.exists():
            raise NotFoundError("Dataset augmentation job not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise NotFoundError("Dataset augmentation job not found")
        return data

    def _static_temp_url(self, path: Path) -> str:
        rel = path.resolve(strict=False).relative_to(settings.temp_dir.resolve()).as_posix()
        return f"/static/temp/{rel}"

    def _dataset_root(self, db: Session, standard_dataset_id: int) -> tuple[StandardDataset, Path]:
        dataset = self._datasets.get_dataset(db, int(standard_dataset_id))
        root = resolve_storage_token(dataset.storage_path)
        return dataset, root

    def preview(self, db: Session, standard_dataset_id: int, payload: DatasetAugmentationPreviewRequest) -> DatasetAugmentationPreviewOut:
        _dataset, root = self._dataset_root(db, int(standard_dataset_id))
        images = iter_image_files(root)
        with_labels = 0
        for image in images:
            label = root / image.relative_to(root).with_suffix(".txt")
            if label.exists():
                with_labels += 1
        generated = len(images)
        if payload.slice.enabled:
            generated += len(images) * max(0, len(payload.slice.scales) - 1)
        if payload.rotate.enabled and payload.rotate.angles:
            generated += len(images) * len(payload.rotate.angles)
        if payload.translate.enabled and payload.translate.offsets:
            generated += len(images) * len(payload.translate.offsets)
        total_outputs = generated + (len(images) if payload.include_original else 0)
        return DatasetAugmentationPreviewOut(
            total_images=len(images),
            with_labels=with_labels,
            estimated_generated_outputs=generated,
            estimated_total_outputs=total_outputs,
            per_transform={
                "slice": len(images) * max(0, len(payload.slice.scales) - 1) if payload.slice.enabled else 0,
                "rotate": len(images) * len(payload.rotate.angles) if payload.rotate.enabled else 0,
                "translate": len(images) * len(payload.translate.offsets) if payload.translate.enabled else 0,
            },
            note="V3 currently materializes a cloned dataset as the augmentation output.",
        )

    def create_job(self, db: Session, standard_dataset_id: int, payload: DatasetAugmentationCreate) -> DatasetAugmentationJobOut:
        dataset, root = self._dataset_root(db, int(standard_dataset_id))
        job_id = uuid.uuid4().hex
        out_dir = self.output_dir(int(standard_dataset_id), job_id)
        copy_tree(root, out_dir)
        output_file_count, _ = count_tree(out_dir)
        status = {
            "job_id": job_id,
            "standard_dataset_id": int(standard_dataset_id),
            "status": "completed",
            "phase": "done",
            "progress": 100,
            "processed": len(iter_image_files(root)),
            "total": len(iter_image_files(root)),
            "seq": 1,
            "last_result_id": 0,
            "config": payload.model_dump(mode="json"),
            "result": {
                "output_url": self._static_temp_url(out_dir),
                "output_file_count": int(output_file_count),
                "generated_images": len(iter_image_files(root)),
                "generated_labels": 0,
                "published_standard_dataset_id": None,
            },
            "cancel_requested": False,
            "error_message": None,
            "created_at": self._utcnow().isoformat(),
            "updated_at": self._utcnow().isoformat(),
        }
        self._write_status(int(standard_dataset_id), job_id, status)
        return DatasetAugmentationJobOut.model_validate(status)

    def get_job(self, standard_dataset_id: int, job_id: str) -> DatasetAugmentationJobOut:
        return DatasetAugmentationJobOut.model_validate(self._read_status(int(standard_dataset_id), str(job_id)))

    def cancel_job(self, standard_dataset_id: int, job_id: str) -> DatasetAugmentationJobOut:
        with self._lock(int(standard_dataset_id), str(job_id)):
            payload = self._read_status(int(standard_dataset_id), str(job_id))
            payload["cancel_requested"] = True
            if payload.get("status") not in ("completed", "failed", "cancelled"):
                payload["status"] = "cancelled"
                payload["phase"] = "cancelled"
            payload["seq"] = int(payload.get("seq") or 0) + 1
            payload["updated_at"] = self._utcnow().isoformat()
            self._write_status(int(standard_dataset_id), str(job_id), payload)
        return DatasetAugmentationJobOut.model_validate(payload)

    def publish_job(
        self,
        db: Session,
        standard_dataset_id: int,
        job_id: str,
        payload: DatasetAugmentationPublishIn,
    ) -> DatasetAugmentationPublishOut:
        status = self._read_status(int(standard_dataset_id), str(job_id))
        if str(status.get("status") or "") != "completed":
            raise ConflictError("Only completed augmentation jobs can be published")
        source = self._datasets.get_dataset(db, int(standard_dataset_id))
        out_dir = self.output_dir(int(standard_dataset_id), str(job_id))
        target_name = f"{source.name}-aug-{str(job_id)[:8]}"
        published = self._datasets.materialize_from_source_tree(
            db,
            name=target_name,
            dataset_type=source.dataset_type,
            source_root=out_dir,
            description=payload.message or f"Augmentation output from {source.name}",
            source_type="augmentation_publish",
            publish_config={"source_standard_dataset_id": int(source.standard_dataset_id), "job_id": str(job_id)},
            created_by=payload.created_by,
        )
        status.setdefault("result", {})["published_standard_dataset_id"] = int(published.standard_dataset_id)
        status["seq"] = int(status.get("seq") or 0) + 1
        status["updated_at"] = self._utcnow().isoformat()
        self._write_status(int(standard_dataset_id), str(job_id), status)
        return DatasetAugmentationPublishOut(
            standard_dataset_id=int(published.standard_dataset_id),
            job_id=str(job_id),
            source_standard_dataset_id=int(source.standard_dataset_id),
        )

    def read_results_since(self, standard_dataset_id: int, job_id: str, *, after_result_id: int = 0) -> List[Dict[str, Any]]:
        return []
