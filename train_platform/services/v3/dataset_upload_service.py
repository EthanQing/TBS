from __future__ import annotations

import logging
import math
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal, session_scope
from train_platform.domains.datasets.illegal.service import IllegalDatasetService
from train_platform.domains.datasets.illegal import versions as illegal_versions
from train_platform.domains.datasets.standard import StandardDatasetService
from train_platform.domains.datasets.standard.content import import_source_tree
from train_platform.domains.datasets.standard.mounted import import_mounted_source_tree
from train_platform.models.v3.dataset_upload import DatasetUploadSession, DatasetUploadTask
from train_platform.platform.filesystem import extract_archive, remove_tree
from train_platform.services.v3.dataset_import_service import DatasetImportService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


logger = logging.getLogger("train_platform.dataset_upload")

_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetUploadService:
    def _session_lock(self, session_id: str) -> threading.RLock:
        with _SESSION_LOCKS_GUARD:
            lock = _SESSION_LOCKS.get(str(session_id))
            if lock is None:
                lock = threading.RLock()
                _SESSION_LOCKS[str(session_id)] = lock
            return lock

    def _validate_kind(self, dataset_kind: str) -> str:
        kind = str(dataset_kind or "").strip().lower()
        if kind not in {"standard", "illegal"}:
            raise ValidationError("dataset_kind must be standard or illegal")
        return kind

    def _validate_mode(self, dataset_kind: str, mode: str | None) -> str:
        normalized = str(mode or "upload").strip().lower()
        if normalized not in {"upload", "append"}:
            raise ValidationError("mode must be upload or append")
        if dataset_kind == "standard" and normalized != "upload":
            raise ValidationError("Standard dataset only supports upload mode")
        return normalized

    def _ensure_dataset_exists(self, db: Session, dataset_kind: str, dataset_id: int) -> None:
        if dataset_kind == "standard":
            StandardDatasetService().get_dataset(db, int(dataset_id))
        else:
            IllegalDatasetService().get_dataset(db, int(dataset_id))

    def _session_root(self, session_id: str) -> Path:
        return settings.upload_sessions_dir / str(session_id)

    def _parts_root(self, session_id: str) -> Path:
        return self._session_root(session_id) / "parts"

    def _part_path(self, session_id: str, part_no: int) -> Path:
        return self._parts_root(session_id) / f"{int(part_no)}.part"

    def _session_out_uploaded_parts(self, row: DatasetUploadSession) -> list[int]:
        return sorted(int(x) for x in (row.uploaded_parts or []))

    def _refresh_uploaded_parts(self, row: DatasetUploadSession) -> list[int]:
        parts_dir = self._parts_root(row.session_id)
        if not parts_dir.exists():
            return self._session_out_uploaded_parts(row)
        uploaded: list[int] = []
        for path in parts_dir.glob("*.part"):
            try:
                part_no = int(path.stem)
            except Exception:
                continue
            if 1 <= part_no <= int(row.total_parts):
                uploaded.append(part_no)
        row.uploaded_parts = sorted(set(uploaded))
        return list(row.uploaded_parts or [])

    def create_session(
        self,
        db: Session,
        dataset_kind: str,
        dataset_id: int,
        *,
        filename: str,
        total_size: int,
        chunk_size: int | None = None,
        mode: str = "upload",
        created_by: str | None = None,
    ) -> DatasetUploadSession:
        kind = self._validate_kind(dataset_kind)
        mode = self._validate_mode(kind, mode)
        self._ensure_dataset_exists(db, kind, int(dataset_id))
        total_size = int(total_size)
        if total_size <= 0:
            raise ValidationError("total_size must be greater than 0")
        effective_chunk_size = int(chunk_size or settings.upload_chunk_size_mb * 1024 * 1024)
        if effective_chunk_size <= 0:
            raise ValidationError("chunk_size must be greater than 0")
        session_id = str(uuid.uuid4())
        total_parts = int(math.ceil(total_size / effective_chunk_size))
        row = DatasetUploadSession(
            session_id=session_id,
            dataset_kind=kind,
            dataset_id=int(dataset_id),
            mode=mode,
            filename=Path(str(filename or "dataset.zip")).name,
            total_size=total_size,
            chunk_size=effective_chunk_size,
            total_parts=total_parts,
            uploaded_parts=[],
            status="uploading",
            created_by=created_by,
            expires_at=_utcnow() + timedelta(hours=int(settings.upload_session_ttl_hours)),
        )
        db.add(row)
        self._parts_root(session_id).mkdir(parents=True, exist_ok=True)
        db.commit()
        db.refresh(row)
        logger.info("Created dataset upload session session_id=%s kind=%s dataset_id=%s", session_id, kind, dataset_id)
        return row

    def get_session(self, db: Session, dataset_kind: str, dataset_id: int, session_id: str) -> DatasetUploadSession:
        kind = self._validate_kind(dataset_kind)
        row = (
            db.query(DatasetUploadSession)
            .filter(
                DatasetUploadSession.session_id == str(session_id),
                DatasetUploadSession.dataset_kind == kind,
                DatasetUploadSession.dataset_id == int(dataset_id),
            )
            .first()
        )
        if not row:
            raise NotFoundError("Upload session not found")
        self._refresh_uploaded_parts(row)
        db.commit()
        db.refresh(row)
        return row

    def save_part(
        self,
        db: Session,
        dataset_kind: str,
        dataset_id: int,
        session_id: str,
        part_no: int,
        upload,
    ) -> dict[str, Any]:
        row = self.get_session(db, dataset_kind, dataset_id, session_id)
        if row.status not in {"uploading", "failed"}:
            raise ConflictError("Upload session is not accepting parts")
        part_no = int(part_no)
        if part_no < 1 or part_no > int(row.total_parts):
            raise ValidationError("part_no is outside session range")
        with self._session_lock(row.session_id):
            parts_dir = self._parts_root(row.session_id)
            parts_dir.mkdir(parents=True, exist_ok=True)
            target = self._part_path(row.session_id, part_no)
            tmp = target.with_suffix(".part.tmp")
            size = 0
            try:
                with tmp.open("wb") as f:
                    while True:
                        chunk = upload.file.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        f.write(chunk)
                expected = int(row.chunk_size) if part_no < int(row.total_parts) else int(row.total_size) - int(row.chunk_size) * (int(row.total_parts) - 1)
                if size != expected:
                    raise ValidationError(f"Chunk size mismatch: expected {expected}, got {size}")
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
                try:
                    upload.file.seek(0)
                except Exception:
                    pass
            row.status = "uploading"
            uploaded_parts = self._refresh_uploaded_parts(row)
            db.commit()
            return {
                "session_id": row.session_id,
                "part_no": part_no,
                "size": int(size),
                "uploaded_parts": uploaded_parts,
                "status": row.status,
            }

    def cancel_session(self, db: Session, dataset_kind: str, dataset_id: int, session_id: str) -> DatasetUploadSession:
        row = self.get_session(db, dataset_kind, dataset_id, session_id)
        with self._session_lock(row.session_id):
            row.status = "cancelled"
            remove_tree(self._session_root(row.session_id), ignore_errors=True)
            db.commit()
            db.refresh(row)
            return row

    def complete_session(
        self,
        db: Session,
        dataset_kind: str,
        dataset_id: int,
        session_id: str,
        *,
        message: str | None = None,
    ) -> DatasetUploadTask:
        row = self.get_session(db, dataset_kind, dataset_id, session_id)
        if row.status not in {"uploading", "failed"}:
            raise ConflictError("Upload session cannot be completed")
        with self._session_lock(row.session_id):
            row.status = "completing"
            db.commit()
            uploaded_parts = self._refresh_uploaded_parts(row)
            missing = [idx for idx in range(1, int(row.total_parts) + 1) if idx not in set(uploaded_parts)]
            if missing:
                row.status = "failed"
                row.error_message = f"Missing parts: {missing[:20]}"
                db.commit()
                raise ValidationError(row.error_message)
            archive_path = self._session_root(row.session_id) / "archive.zip"
            tmp_archive = archive_path.with_suffix(".zip.tmp")
            try:
                with tmp_archive.open("wb") as out:
                    for idx in range(1, int(row.total_parts) + 1):
                        with self._part_path(row.session_id, idx).open("rb") as part:
                            shutil.copyfileobj(part, out, length=1024 * 1024)
                actual_size = int(tmp_archive.stat().st_size)
                if actual_size != int(row.total_size):
                    raise ValidationError(f"Merged ZIP size mismatch: expected {row.total_size}, got {actual_size}")
                tmp_archive.replace(archive_path)
            except Exception as exc:
                tmp_archive.unlink(missing_ok=True)
                row.status = "failed"
                row.error_message = str(exc)
                db.commit()
                raise
            task = self._create_task(
                db,
                dataset_kind=row.dataset_kind,
                dataset_id=int(row.dataset_id),
                mode=row.mode,
                source_path=archive_path,
                source_type="zip",
                session_id=row.session_id,
                created_by=row.created_by,
                message=message,
            )
            row.status = "completed"
            row.error_message = None
            db.commit()
            db.refresh(task)
            logger.info("Completed upload session session_id=%s task_id=%s", row.session_id, task.task_id)
            return task

    def _create_task(
        self,
        db: Session,
        *,
        dataset_kind: str,
        dataset_id: int,
        mode: str,
        source_path: Path,
        source_type: str,
        session_id: str | None = None,
        created_by: str | None = None,
        message: str | None = None,
    ) -> DatasetUploadTask:
        task = DatasetUploadTask(
            task_id=str(uuid.uuid4()),
            dataset_kind=dataset_kind,
            dataset_id=int(dataset_id),
            session_id=session_id,
            mode=mode,
            source_path=str(Path(source_path).resolve(strict=False)),
            source_type=source_type,
            status="queued",
            stage="queued",
            progress=0,
            processed_count=0,
            total_count=0,
            created_by=created_by,
            message=message,
        )
        db.add(task)
        db.flush()
        return task

    def create_import_task(
        self,
        db: Session,
        dataset_kind: str,
        dataset_id: int,
        *,
        root_id: str = "default",
        rel_path: str,
        mode: str = "upload",
        storage_strategy: str = "copy",
        created_by: str | None = None,
        message: str | None = None,
    ) -> DatasetUploadTask:
        kind = self._validate_kind(dataset_kind)
        mode = self._validate_mode(kind, mode)
        self._ensure_dataset_exists(db, kind, int(dataset_id))
        strategy = str(storage_strategy or "copy").strip().lower()
        if strategy not in {"copy", "link"}:
            raise ValidationError("storage_strategy must be copy or link")
        _root_key, _root, source, _root_out = DatasetImportService().resolve_path(root_id, rel_path)
        if not source.exists():
            raise NotFoundError("Import path not found")
        source_type = "dir" if source.is_dir() else "zip"
        if source_type == "zip" and source.suffix.lower() != ".zip":
            raise ValidationError("Import file must be a ZIP archive or a directory")
        if strategy == "link":
            if source_type != "dir":
                raise ValidationError("Linked import only supports directories")
            source_type = "dir_link"
        task = self._create_task(
            db,
            dataset_kind=kind,
            dataset_id=int(dataset_id),
            mode=mode,
            source_path=source,
            source_type=source_type,
            created_by=created_by,
            message=message,
        )
        db.commit()
        db.refresh(task)
        logger.info("Created import task task_id=%s path=%s", task.task_id, source)
        return task

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

    def run_task(self, task_id: str) -> None:
        import_progress = self._make_import_progress_callback(task_id)
        snapshot = self._get_task_snapshot(task_id)
        try:
            self._update_task_by_id(task_id, status="extracting", stage="extracting", progress=10, detail_message="Preparing dataset import")
            source = Path(snapshot["source_path"])
            logger.info("Dataset upload task started task_id=%s session_id=%s", snapshot["task_id"], snapshot["session_id"])
            if snapshot["dataset_kind"] == "standard":
                if snapshot["source_type"] == "dir_link":
                    self._update_task_by_id(task_id, status="linking", stage="linking", progress=30, detail_message="Preparing mounted standard dataset import")
                    with session_scope() as db:
                        import_mounted_source_tree(
                            db,
                            int(snapshot["dataset_id"]),
                            source,
                            created_by=snapshot["created_by"],
                            filename=source.name,
                        )
                elif snapshot["source_type"] == "dir":
                    self._update_task_by_id(task_id, status="validating", stage="validating", progress=30, detail_message="Validating standard dataset source")
                    with session_scope() as db:
                        import_source_tree(
                            db,
                            int(snapshot["dataset_id"]),
                            source,
                            created_by=snapshot["created_by"],
                            filename=source.name,
                        )
                else:
                    extracted_root, staging = self._extract_archive_for_task(task_id, source)
                    try:
                        self._update_task_by_id(task_id, status="validating", stage="validating", progress=75, detail_message="Validating extracted standard dataset")
                        with session_scope() as db:
                            import_source_tree(
                                db,
                                int(snapshot["dataset_id"]),
                                extracted_root,
                                created_by=snapshot["created_by"],
                                filename=source.name,
                            )
                    finally:
                        remove_tree(staging, ignore_errors=True)
            else:
                if snapshot["source_type"] == "dir_link":
                    self._update_task_by_id(task_id, status="linking", stage="linking", progress=30, detail_message="Preparing mounted illegal dataset import")
                    with session_scope() as db:
                        illegal_versions.import_mounted_source_tree(
                            db,
                            int(snapshot["dataset_id"]),
                            source,
                            message=snapshot["message"],
                            created_by=snapshot["created_by"],
                            append=(snapshot["mode"] == "append"),
                            filename=source.name,
                            progress_callback=import_progress,
                        )
                elif snapshot["source_type"] == "dir":
                    self._update_task_by_id(task_id, status="validating", stage="validating", progress=30, detail_message="Validating illegal dataset source")
                    with session_scope() as db:
                        illegal_versions.import_source_tree(
                            db,
                            int(snapshot["dataset_id"]),
                            source,
                            message=snapshot["message"],
                            created_by=snapshot["created_by"],
                            append=(snapshot["mode"] == "append"),
                            filename=source.name,
                            progress_callback=import_progress,
                        )
                else:
                    extracted_root, staging = self._extract_archive_for_task(task_id, source)
                    try:
                        self._update_task_by_id(task_id, status="validating", stage="validating", progress=75, detail_message="Validating extracted illegal dataset")
                        with session_scope() as db:
                            illegal_versions.import_source_tree(
                                db,
                                int(snapshot["dataset_id"]),
                                extracted_root,
                                message=snapshot["message"],
                                created_by=snapshot["created_by"],
                                append=(snapshot["mode"] == "append"),
                                filename=source.name,
                                progress_callback=import_progress,
                            )
                    finally:
                        remove_tree(staging, ignore_errors=True)
            self._update_task_by_id(task_id, status="done", stage="done", progress=100, detail_message="Dataset import completed", finished=True)
            self._cleanup_task_source_by_snapshot(snapshot)
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
                logger.exception("Dataset upload task failed task_id=%s session_id=%s", snapshot.get("task_id"), snapshot.get("session_id"))

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
        self._update_task_by_id(task_id, status="extracting", stage="extracting", progress=10, detail_message="Extracting dataset archive")
        try:
            extracted_root = extract_archive(
                Path(source),
                extracted_dir,
                progress_callback=self._make_extract_progress_callback(task_id, start=10, end=70),
            )
            self._update_task_by_id(task_id, status="extracting", stage="extracting", progress=70, detail_message="Archive extraction completed")
            return extracted_root, staging
        except Exception:
            remove_tree(staging, ignore_errors=True)
            raise

    def _make_extract_progress_callback(
        self,
        task_id: str,
        *,
        start: int,
        end: int,
    ):
        start = max(0, min(100, int(start)))
        end = max(start, min(100, int(end)))
        last_progress = {"value": start}

        def _callback(extracted_files: int, total_files: int, _rel_path: str) -> None:
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
                current_item=_rel_path,
                detail_message=f"Extracted {done}/{total} files" if total else "Extracting files",
            )

        return _callback

    def _make_import_progress_callback(self, task_id: str) -> Callable[..., None]:
        last_snapshot: dict[str, Any] = {"progress": -1, "stage": "", "processed_count": None, "total_count": None, "current_item": None, "detail_message": None}

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

    def _cleanup_task_source(self, task: DatasetUploadTask) -> None:
        if task.session_id:
            remove_tree(self._session_root(task.session_id), ignore_errors=True)

    def _cleanup_task_source_by_snapshot(self, task: dict[str, Any]) -> None:
        session_id = str(task.get("session_id") or "")
        if session_id:
            remove_tree(self._session_root(session_id), ignore_errors=True)

    def cleanup_expired_sessions(self, db: Session) -> int:
        now = _utcnow()
        rows = (
            db.query(DatasetUploadSession)
            .filter(DatasetUploadSession.expires_at < now, DatasetUploadSession.status.in_(("uploading", "failed", "cancelled")))
            .all()
        )
        count = 0
        for row in rows:
            remove_tree(self._session_root(row.session_id), ignore_errors=True)
            row.status = "expired" if row.status != "cancelled" else row.status
            count += 1
        db.commit()
        return count
