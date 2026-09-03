from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal, session_scope
from train_platform.domains.datasets.illegal import versions as illegal_versions
from train_platform.domains.datasets.standard import content as standard_content
from train_platform.domains.datasets.standard import mounted as standard_mounted
from train_platform.models.v3.dataset_upload import DatasetUploadTask
from train_platform.platform.filesystem import extract_archive, remove_tree
from train_platform.utils.exceptions import NotFoundError, ValidationError


logger = logging.getLogger("train_platform.dataset_upload")


def _utcnow():
    return datetime.now(timezone.utc)


class DatasetUploadTaskService:
    """Own authoritative task status and dispatch prepared sources to owners."""

    def get_task(self, db: Session, task_id: str) -> DatasetUploadTask:
        task = db.query(DatasetUploadTask).filter(DatasetUploadTask.task_id == str(task_id)).first()
        if not task:
            raise NotFoundError("Dataset upload task not found")
        return task

    def _snapshot_task(self, task: DatasetUploadTask) -> dict[str, Any]:
        return {
            "task_id": str(task.task_id),
            "session_id": str(task.session_id or ""),
            "dataset_kind": str(task.dataset_kind or ""),
            "dataset_id": int(task.dataset_id),
            "mode": str(task.mode or "upload"),
            "source_path": str(task.source_path or ""),
            "source_type": str(task.source_type or ""),
            "created_by": task.created_by,
            "message": task.message,
        }

    def _get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        with session_scope() as db:
            return self._snapshot_task(self.get_task(db, task_id))

    def _update_task_by_id(
        self,
        task_id: str,
        *,
        status: str,
        stage: str,
        progress: int,
        error_message: str | None = None,
        processed_count: int | None = None,
        total_count: int | None = None,
        current_item: str | None = None,
        detail_message: str | None = None,
        finished: bool = False,
    ) -> None:
        with session_scope() as db:
            task = self.get_task(db, task_id)
            self._update_task(
                db,
                task,
                status=status,
                stage=stage,
                progress=progress,
                error_message=error_message,
                processed_count=processed_count,
                total_count=total_count,
                current_item=current_item,
                detail_message=detail_message,
                finished=finished,
            )

    def _prepare_source(self, task_id: str, snapshot: dict[str, Any]) -> tuple[Path, Path | None]:
        source = Path(snapshot["source_path"])
        source_type = snapshot["source_type"]
        if source_type == "zip":
            return self._extract_archive_for_task(task_id, source)
        if source_type not in {"dir", "dir_link"}:
            raise ValidationError(f"Unsupported dataset source type: {source_type}")
        if not source.exists() or not source.is_dir():
            raise NotFoundError("Dataset import source directory not found")
        return source, None

    def _run_standard(self, snapshot: dict[str, Any], source: Path) -> None:
        source_type = snapshot["source_type"]
        if source_type == "dir_link":
            self._update_task_by_id(
                snapshot["task_id"],
                status="linking",
                stage="linking",
                progress=30,
                detail_message="Preparing mounted standard dataset import",
            )
            importer = standard_mounted.import_mounted_source_tree
        else:
            self._update_task_by_id(
                snapshot["task_id"],
                status="validating",
                stage="validating",
                progress=75 if source_type == "zip" else 30,
                detail_message="Validating extracted standard dataset" if source_type == "zip" else "Validating standard dataset source",
            )
            importer = standard_content.import_source_tree
        with session_scope() as db:
            importer(
                db,
                int(snapshot["dataset_id"]),
                source,
                created_by=snapshot["created_by"],
                filename=Path(snapshot["source_path"]).name,
            )

    def _run_illegal(self, snapshot: dict[str, Any], source: Path, import_progress: Callable[..., None]) -> None:
        source_type = snapshot["source_type"]
        if source_type == "dir_link":
            self._update_task_by_id(
                snapshot["task_id"],
                status="linking",
                stage="linking",
                progress=30,
                detail_message="Preparing mounted illegal dataset import",
            )
            importer = illegal_versions.import_mounted_source_tree
        else:
            self._update_task_by_id(
                snapshot["task_id"],
                status="validating",
                stage="validating",
                progress=75 if source_type == "zip" else 30,
                detail_message="Validating extracted illegal dataset" if source_type == "zip" else "Validating illegal dataset source",
            )
            importer = illegal_versions.import_source_tree
        with session_scope() as db:
            importer(
                db,
                int(snapshot["dataset_id"]),
                source,
                message=snapshot["message"],
                created_by=snapshot["created_by"],
                append=(snapshot["mode"] == "append"),
                filename=Path(snapshot["source_path"]).name,
                progress_callback=import_progress,
            )

    def run_task(self, task_id: str) -> None:
        snapshot = self._get_task_snapshot(task_id)
        import_progress = self._make_import_progress_callback(task_id)
        staging: Path | None = None
        try:
            self._update_task_by_id(
                task_id,
                status="extracting",
                stage="extracting",
                progress=10,
                detail_message="Preparing dataset import",
            )
            logger.info(
                "Dataset upload task started task_id=%s session_id=%s",
                snapshot["task_id"],
                snapshot["session_id"],
            )
            source, staging = self._prepare_source(task_id, snapshot)
            if snapshot["dataset_kind"] == "standard":
                self._run_standard(snapshot, source)
            elif snapshot["dataset_kind"] == "illegal":
                self._run_illegal(snapshot, source, import_progress)
            else:
                raise ValidationError(f"Unsupported dataset kind: {snapshot['dataset_kind']}")
            self._update_task_by_id(
                task_id,
                status="done",
                stage="done",
                progress=100,
                detail_message="Dataset import completed",
                finished=True,
            )
            self._cleanup_uploaded_source(snapshot)
            logger.info("Dataset upload task finished task_id=%s", snapshot["task_id"])
        except Exception as exc:
            try:
                with session_scope() as db:
                    task = self.get_task(db, task_id)
                    self._update_task(
                        db,
                        task,
                        status="failed",
                        stage="failed",
                        progress=int(task.progress or 0),
                        error_message=str(exc),
                        detail_message="Dataset import failed",
                        finished=True,
                    )
            finally:
                logger.exception(
                    "Dataset upload task failed task_id=%s session_id=%s",
                    snapshot.get("task_id"),
                    snapshot.get("session_id"),
                )
        finally:
            if staging is not None:
                remove_tree(staging, ignore_errors=True)

    def _update_task(
        self,
        db: Session,
        task: DatasetUploadTask,
        *,
        status: str,
        stage: str,
        progress: int,
        error_message: str | None = None,
        processed_count: int | None = None,
        total_count: int | None = None,
        current_item: str | None = None,
        detail_message: str | None = None,
        finished: bool = False,
    ) -> None:
        task.status = status
        task.stage = stage
        task.progress = max(0, min(100, int(progress)))
        task.error_message = error_message
        if processed_count is not None:
            task.processed_count = max(0, int(processed_count or 0))
        if total_count is not None:
            task.total_count = max(0, int(total_count or 0))
        if current_item is not None:
            task.current_item = str(current_item or "")[:1000] or None
        if detail_message is not None:
            task.detail_message = str(detail_message or "") or None
        if finished:
            task.finished_at = _utcnow()
        db.commit()

    def _extract_archive_for_task(self, task_id: str, source: Path) -> tuple[Path, Path]:
        staging = settings.dataset_staging_dir / "upload-tasks" / str(task_id)
        extracted_dir = staging / "extracted"
        remove_tree(staging, ignore_errors=True)
        self._update_task_by_id(
            task_id,
            status="extracting",
            stage="extracting",
            progress=10,
            detail_message="Extracting dataset archive",
        )
        try:
            extracted_root = extract_archive(
                Path(source),
                extracted_dir,
                progress_callback=self._make_extract_progress_callback(task_id, start=10, end=70),
            )
            self._update_task_by_id(
                task_id,
                status="extracting",
                stage="extracting",
                progress=70,
                detail_message="Archive extraction completed",
            )
            return extracted_root, staging
        except Exception:
            remove_tree(staging, ignore_errors=True)
            raise

    def _make_extract_progress_callback(self, task_id: str, *, start: int, end: int):
        start = max(0, min(100, int(start)))
        end = max(start, min(100, int(end)))
        last_progress = {"value": start}

        def _callback(extracted_files: int, total_files: int, rel_path: str) -> None:
            total = max(0, int(total_files or 0))
            done = max(0, int(extracted_files or 0))
            if total <= 0:
                progress = end
            else:
                ratio = min(1.0, done / max(1, total))
                progress = start + int(round((end - start) * ratio))
            progress = max(start, min(end, progress))
            if progress <= int(last_progress["value"]):
                return
            last_progress["value"] = progress
            self._update_task_by_id(
                task_id,
                status="extracting",
                stage="extracting",
                progress=progress,
                processed_count=done,
                total_count=total,
                current_item=rel_path,
                detail_message=f"Extracted {done}/{total} files" if total else "Extracting files",
            )

        return _callback

    def _make_import_progress_callback(self, task_id: str) -> Callable[..., None]:
        last_snapshot: dict[str, Any] = {
            "progress": -1,
            "stage": "",
            "processed_count": None,
            "total_count": None,
            "current_item": None,
            "detail_message": None,
        }

        def _callback(progress: int, stage: str, detail: dict[str, Any] | None = None, **kwargs: Any) -> None:
            progress = max(0, min(99, int(progress)))
            detail_payload: dict[str, Any] = {}
            if isinstance(detail, dict):
                detail_payload.update(detail)
            detail_payload.update(kwargs)
            normalized_stage = str(stage or "validating")
            processed_count = detail_payload.get("processed_count")
            total_count = detail_payload.get("total_count")
            current_item = detail_payload.get("current_item")
            detail_message = detail_payload.get("detail_message")
            snapshot = {
                "progress": progress,
                "stage": normalized_stage,
                "processed_count": processed_count,
                "total_count": total_count,
                "current_item": current_item,
                "detail_message": detail_message,
            }
            if all(snapshot.get(key) == last_snapshot.get(key) for key in snapshot):
                return
            last_snapshot.update(snapshot)
            with SessionLocal() as progress_db:
                task = progress_db.query(DatasetUploadTask).filter(DatasetUploadTask.task_id == str(task_id)).first()
                if not task or str(task.status) in {"done", "failed", "cancelled"}:
                    return
                progress_value = max(progress, int(task.progress or 0))
                self._update_task(
                    progress_db,
                    task,
                    status=normalized_stage,
                    stage=normalized_stage,
                    progress=progress_value,
                    processed_count=processed_count,
                    total_count=total_count,
                    current_item=current_item,
                    detail_message=detail_message,
                )

        return _callback

    def _cleanup_uploaded_source(self, snapshot: dict[str, Any]) -> None:
        session_id = str(snapshot.get("session_id") or "")
        if session_id:
            remove_tree(settings.upload_sessions_dir / session_id, ignore_errors=True)


__all__ = ["DatasetUploadTaskService"]
