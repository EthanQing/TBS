from __future__ import annotations

import json
import math
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.dataset_event import DatasetEvent
from train_platform.models.enums import DatasetType
from train_platform.schemas.v2.dataset_augmentations import (
    DatasetAugmentationConfig,
    DatasetAugmentationCreate,
    DatasetAugmentationJobOut,
    DatasetAugmentationPreviewOut,
    DatasetAugmentationPublishIn,
    DatasetAugmentationPublishOut,
)
from train_platform.services.dataset_service import DatasetService
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS
from train_platform.utils.path_utils import resolve_dataset_path


_LOCK_GUARD = threading.Lock()
_JOB_LOCKS: Dict[str, threading.Lock] = {}
_DATASET_LOCKS: Dict[int, threading.Lock] = {}


@dataclass
class _Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, float(self.x2) - float(self.x1))

    @property
    def h(self) -> float:
        return max(0.0, float(self.y2) - float(self.y1))

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class _SourceImage:
    rel_path: str
    image_path: Path
    label_path: Optional[Path]
    width: int
    height: int


def uuid4_short() -> str:
    import uuid

    return str(uuid.uuid4())


class DatasetAugmentationService:
    ACTIVE_STATUSES = {"queued", "running"}
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    ACTIVE_STALE_AFTER = timedelta(hours=12)
    RETAIN_DAYS = 7

    def __init__(self) -> None:
        self._dataset_svc = DatasetService()

    # --------------------
    # Paths / lock helpers
    # --------------------
    def jobs_root(self, dataset_id: int) -> Path:
        root = settings.temp_dir / "dataset_augmentations" / str(int(dataset_id))
        root.mkdir(parents=True, exist_ok=True)
        return root

    def job_dir(self, dataset_id: int, job_id: str) -> Path:
        d = self.jobs_root(dataset_id) / str(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status_path(self, dataset_id: int, job_id: str) -> Path:
        return self.job_dir(dataset_id, job_id) / "status.json"

    def results_path(self, dataset_id: int, job_id: str) -> Path:
        return self.job_dir(dataset_id, job_id) / "results.jsonl"

    def output_dir(self, dataset_id: int, job_id: str) -> Path:
        out = self.job_dir(dataset_id, job_id) / "output"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def output_images_dir(self, dataset_id: int, job_id: str) -> Path:
        d = self.output_dir(dataset_id, job_id) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output_labels_dir(self, dataset_id: int, job_id: str) -> Path:
        d = self.output_dir(dataset_id, job_id) / "labels"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _job_lock(self, dataset_id: int, job_id: str) -> threading.Lock:
        key = f"{int(dataset_id)}:{str(job_id)}"
        with _LOCK_GUARD:
            lock = _JOB_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _JOB_LOCKS[key] = lock
            return lock

    def _dataset_lock(self, dataset_id: int) -> threading.Lock:
        did = int(dataset_id)
        with _LOCK_GUARD:
            lock = _DATASET_LOCKS.get(did)
            if lock is None:
                lock = threading.Lock()
                _DATASET_LOCKS[did] = lock
            return lock

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _to_iso(self, dt: Optional[datetime] = None) -> str:
        return (dt or self._utcnow()).isoformat()

    def _parse_time(self, raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None

    # --------------------
    # File status helpers
    # --------------------
    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise NotFoundError("Augmentation job not found")
        last_err: Exception | None = None
        data = None
        for _ in range(5):
            try:
                data = json.loads(path.read_text(encoding="utf-8")) or {}
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.02)
            except Exception as e:
                last_err = e
                break
        if data is None:
            raise ValidationError(f"Failed to read augmentation status: {type(last_err).__name__}: {last_err}") from last_err
        if not isinstance(data, dict):
            raise ValidationError("Invalid augmentation status payload")
        return data

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        payload = dict(data or {})
        payload["updated_at"] = self._to_iso()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(path)

    def _read_status(self, dataset_id: int, job_id: str) -> Dict[str, Any]:
        return self._read_json(self.status_path(dataset_id, job_id))

    def _update_status(
        self,
        dataset_id: int,
        job_id: str,
        patch: Dict[str, Any],
        *,
        bump_seq: bool = True,
    ) -> Dict[str, Any]:
        lock = self._job_lock(dataset_id, job_id)
        with lock:
            cur = self._read_status(dataset_id, job_id)
            cur.update(dict(patch or {}))
            cur["progress"] = max(0, min(100, int(cur.get("progress") or 0)))
            cur["processed"] = max(0, int(cur.get("processed") or 0))
            cur["total"] = max(0, int(cur.get("total") or 0))
            if bump_seq:
                cur["seq"] = int(cur.get("seq") or 0) + 1
            self._write_json_atomic(self.status_path(dataset_id, job_id), cur)
            return cur

    def _append_result(self, dataset_id: int, job_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        lock = self._job_lock(dataset_id, job_id)
        with lock:
            st = self._read_status(dataset_id, job_id)
            rid = int(st.get("last_result_id") or 0) + 1
            payload = dict(row or {})
            payload["result_id"] = rid
            payload["processed_at"] = self._to_iso()
            with open(self.results_path(dataset_id, job_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            st["last_result_id"] = rid
            st["seq"] = int(st.get("seq") or 0) + 1
            self._write_json_atomic(self.status_path(dataset_id, job_id), st)
            return payload

    def read_results_since(self, dataset_id: int, job_id: str, *, after_result_id: int = 0) -> List[Dict[str, Any]]:
        path = self.results_path(dataset_id, job_id)
        if not path.exists() or not path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        after = max(0, int(after_result_id or 0))
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = int(obj.get("result_id") or 0)
                if rid <= after:
                    continue
                out.append(obj)
        out.sort(key=lambda x: int(x.get("result_id") or 0))
        return out

    def _cancel_requested(self, dataset_id: int, job_id: str) -> bool:
        try:
            st = self._read_status(dataset_id, job_id)
            return bool(st.get("cancel_requested"))
        except Exception:
            return True

    # --------------------
    # Public API
    # --------------------
    def preview(
        self,
        db: Session,
        dataset_id: int,
        payload: DatasetAugmentationConfig,
    ) -> DatasetAugmentationPreviewOut:
        cfg = DatasetAugmentationConfig.model_validate(payload)
        _ds, _ver, root = self._resolve_dataset_version_root(db, int(dataset_id), cfg.version_id)
        sources = self._collect_sources(root)

        per_transform = {"slice": 0, "rotate": 0, "translate": 0}
        est_generated = 0
        with_labels = 0
        for src in sources:
            if src.label_path and src.label_path.exists():
                with_labels += 1
            est = self._estimate_per_image(cfg, src.width, src.height)
            per_transform["slice"] += est["slice"]
            per_transform["rotate"] += est["rotate"]
            per_transform["translate"] += est["translate"]
            est_generated += est["slice"] + est["rotate"] + est["translate"]

        include_orig = int(len(sources)) if cfg.include_original else 0
        note = None
        if cfg.max_outputs_per_image:
            note = f"Per-image augmented outputs are capped at {int(cfg.max_outputs_per_image)}."

        return DatasetAugmentationPreviewOut(
            total_images=int(len(sources)),
            with_labels=int(with_labels),
            estimated_generated_outputs=int(est_generated),
            estimated_total_outputs=int(est_generated + include_orig),
            per_transform=per_transform,
            note=note,
        )

    def create_job(self, db: Session, dataset_id: int, payload: DatasetAugmentationCreate) -> DatasetAugmentationJobOut:
        cfg = DatasetAugmentationConfig.model_validate(payload)
        with self._dataset_lock(int(dataset_id)):
            active = self._find_active_job(int(dataset_id))
            if active is not None:
                jid = str(active.get("job_id") or "unknown")
                st = str(active.get("status") or "running")
                raise ConflictError(f"Another augmentation job is active (job_id={jid}, status={st})")

            _ds, ver, root = self._resolve_dataset_version_root(db, int(dataset_id), cfg.version_id)
            sources = self._collect_sources(root)
            if not sources:
                raise ValidationError("No image files found in selected dataset version")

            job_id = uuid4_short()
            output_url = self._static_temp_url(self.output_dir(int(dataset_id), job_id))
            status: Dict[str, Any] = {
                "job_id": job_id,
                "dataset_id": int(dataset_id),
                "version_id": int(ver.version_id),
                "status": "queued",
                "phase": "preparing",
                "progress": 0,
                "processed": 0,
                "total": int(len(sources)),
                "seq": 1,
                "last_result_id": 0,
                "cancel_requested": False,
                "error_message": None,
                "config": cfg.model_dump(mode="json"),
                "result": {
                    "output_url": output_url,
                    "output_file_count": 0,
                    "generated_images": 0,
                    "generated_labels": 0,
                    "published_version_id": None,
                },
                "created_at": self._to_iso(),
                "updated_at": self._to_iso(),
            }
            self._write_json_atomic(self.status_path(int(dataset_id), job_id), status)
            self._cleanup_old_jobs(int(dataset_id))

        t = threading.Thread(target=self._run_job, args=(int(dataset_id), str(job_id)), daemon=True)
        t.start()
        return self.get_job(int(dataset_id), str(job_id))

    def get_job(self, dataset_id: int, job_id: str) -> DatasetAugmentationJobOut:
        st = self._read_status(int(dataset_id), str(job_id))
        return DatasetAugmentationJobOut.model_validate(st)

    def cancel_job(self, dataset_id: int, job_id: str) -> DatasetAugmentationJobOut:
        st = self._update_status(int(dataset_id), str(job_id), {"cancel_requested": True}, bump_seq=True)
        if str(st.get("status") or "") == "queued":
            st = self._update_status(
                int(dataset_id),
                str(job_id),
                {"status": "cancelled", "phase": "cancelled", "progress": int(st.get("progress") or 0)},
                bump_seq=True,
            )
        return DatasetAugmentationJobOut.model_validate(st)

    def publish_job(
        self,
        db: Session,
        dataset_id: int,
        job_id: str,
        payload: DatasetAugmentationPublishIn,
    ) -> DatasetAugmentationPublishOut:
        st = self._read_status(int(dataset_id), str(job_id))
        if str(st.get("status") or "") != "completed":
            raise ConflictError("Only completed augmentation jobs can be published")

        output_dir = self.output_dir(int(dataset_id), str(job_id))
        if not output_dir.exists() or not output_dir.is_dir():
            raise NotFoundError("Augmentation output directory not found")

        result = st.get("result") if isinstance(st.get("result"), dict) else {}
        existing_vid = result.get("published_version_id")
        if existing_vid:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(existing_vid), DatasetVersion.dataset_id == int(dataset_id))
                .first()
            )
            ds = self._dataset_svc.get_dataset(db, int(dataset_id))
            if ver:
                return DatasetAugmentationPublishOut(
                    dataset_id=int(dataset_id),
                    job_id=str(job_id),
                    version_id=int(ver.version_id),
                    version=int(ver.version),
                    active_version_id=int(ds.active_version_id) if ds.active_version_id is not None else None,
                    activated=bool(ds.active_version_id == ver.version_id),
                )

        ds, ver = self._dataset_svc.create_version_from_directory(
            db,
            int(dataset_id),
            source_dir=output_dir,
            message=payload.message or f"augmentation:{job_id}",
            created_by=payload.created_by,
            activate=bool(payload.activate),
        )
        try:
            db.add(
                DatasetEvent(
                    dataset_id=int(dataset_id),
                    version_id=int(ver.version_id),
                    event_type="augmentation_published",
                    message=payload.message or "Augmentation output published as new dataset version",
                    created_by=payload.created_by,
                    data={
                        "job_id": str(job_id),
                        "activate": bool(payload.activate),
                        "published_version_id": int(ver.version_id),
                    },
                )
            )
            db.commit()
        except Exception:
            db.rollback()

        result = dict(result or {})
        result["published_version_id"] = int(ver.version_id)
        self._update_status(int(dataset_id), str(job_id), {"result": result}, bump_seq=True)
        return DatasetAugmentationPublishOut(
            dataset_id=int(dataset_id),
            job_id=str(job_id),
            version_id=int(ver.version_id),
            version=int(ver.version),
            active_version_id=int(ds.active_version_id) if ds.active_version_id is not None else None,
            activated=bool(ds.active_version_id == ver.version_id),
        )

    # --------------------
    # Runtime worker
    # --------------------
    def _run_job(self, dataset_id: int, job_id: str) -> None:
        db = SessionLocal()
        try:
            st = self._update_status(
                int(dataset_id),
                str(job_id),
                {"status": "running", "phase": "preparing", "progress": 1, "error_message": None},
                bump_seq=True,
            )
            cfg = DatasetAugmentationConfig.model_validate(st.get("config") or {})
            ds, _ver, root = self._resolve_dataset_version_root(db, int(dataset_id), cfg.version_id)

            self._update_status(int(dataset_id), str(job_id), {"phase": "scanning", "progress": 5}, bump_seq=True)
            sources = self._collect_sources(root)
            if not sources:
                raise ValidationError("No image files found in selected dataset version")
            self._update_status(
                int(dataset_id),
                str(job_id),
                {"total": int(len(sources)), "processed": 0, "phase": "augmenting", "progress": 10},
                bump_seq=True,
            )

            names = self._resolve_class_names(root, ds, sources)
            generated_images = 0
            generated_labels = 0

            for idx, src in enumerate(sources, start=1):
                if self._cancel_requested(int(dataset_id), str(job_id)):
                    self._update_status(
                        int(dataset_id),
                        str(job_id),
                        {"status": "cancelled", "phase": "cancelled", "progress": int((idx - 1) / len(sources) * 100)},
                        bump_seq=True,
                    )
                    return

                summary = self._augment_one_source(
                    dataset_id=int(dataset_id),
                    job_id=str(job_id),
                    source_idx=idx,
                    source=src,
                    cfg=cfg,
                )
                generated_images += int(summary.get("generated_images") or 0)
                generated_labels += int(summary.get("generated_labels") or 0)
                self._append_result(
                    int(dataset_id),
                    str(job_id),
                    {
                        "source_image": src.rel_path,
                        "generated": int(summary.get("generated_images") or 0),
                        "by_transform": summary.get("by_transform") or {},
                    },
                )

                prog = 10 + int((idx / len(sources)) * 85)
                self._update_status(
                    int(dataset_id),
                    str(job_id),
                    {"processed": int(idx), "total": int(len(sources)), "phase": "augmenting", "progress": min(95, prog)},
                    bump_seq=True,
                )

            self._update_status(int(dataset_id), str(job_id), {"phase": "finalizing", "progress": 96}, bump_seq=True)
            out_dir = self.output_dir(int(dataset_id), str(job_id))
            self._write_output_metadata(out_dir, names)

            output_file_count = 0
            for p in out_dir.rglob("*"):
                if p.is_file():
                    output_file_count += 1

            result = {
                "output_url": self._static_temp_url(out_dir),
                "output_file_count": int(output_file_count),
                "generated_images": int(generated_images),
                "generated_labels": int(generated_labels),
                "published_version_id": None,
            }
            self._update_status(
                int(dataset_id),
                str(job_id),
                {
                    "status": "completed",
                    "phase": "done",
                    "progress": 100,
                    "processed": int(len(sources)),
                    "total": int(len(sources)),
                    "result": result,
                },
                bump_seq=True,
            )
        except Exception as e:
            self._update_status(
                int(dataset_id),
                str(job_id),
                {
                    "status": "failed",
                    "phase": "failed",
                    "progress": 100,
                    "error_message": f"{type(e).__name__}: {e}",
                },
                bump_seq=True,
            )
        finally:
            db.close()

    def _augment_one_source(
        self,
        *,
        dataset_id: int,
        job_id: str,
        source_idx: int,
        source: _SourceImage,
        cfg: DatasetAugmentationConfig,
    ) -> Dict[str, Any]:
        out_images = self.output_images_dir(dataset_id, job_id)
        out_labels = self.output_labels_dir(dataset_id, job_id)
        ext = source.image_path.suffix.lower() or ".jpg"
        base = f"{int(source_idx):06d}_{source.image_path.stem}"

        boxes = self._read_yolo_boxes(source.label_path, source.width, source.height)
        transform_limit = max(1, int(cfg.max_outputs_per_image))
        generated_aug = 0
        generated_total = 0
        by_transform = {"original": 0, "slice": 0, "rotate": 0, "translate": 0}

        if cfg.include_original:
            name = f"{base}__aug_original{ext}"
            dst_img = out_images / name
            shutil.copy2(source.image_path, dst_img)
            dst_lbl = out_labels / (Path(name).stem + ".txt")
            if source.label_path and source.label_path.exists():
                shutil.copy2(source.label_path, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            generated_total += 1
            by_transform["original"] += 1

        with Image.open(source.image_path) as img0:
            img = img0.copy()

        if cfg.slice.enabled and cfg.slice.scales:
            for sid, (patch, patch_boxes) in enumerate(self._iter_slice_outputs(img, boxes, cfg), start=1):
                if generated_aug >= transform_limit:
                    break
                if not patch_boxes:
                    continue
                self._save_sample(out_images, out_labels, patch, patch_boxes, f"{base}__aug_slice_{sid}{ext}")
                generated_aug += 1
                generated_total += 1
                by_transform["slice"] += 1

        if cfg.rotate.enabled and cfg.rotate.angles and generated_aug < transform_limit:
            for rid, (rot_img, rot_boxes) in enumerate(self._iter_rotate_outputs(img, boxes, cfg), start=1):
                if generated_aug >= transform_limit:
                    break
                if not rot_boxes:
                    continue
                self._save_sample(out_images, out_labels, rot_img, rot_boxes, f"{base}__aug_rotate_{rid}{ext}")
                generated_aug += 1
                generated_total += 1
                by_transform["rotate"] += 1

        if cfg.translate.enabled and cfg.translate.offsets and generated_aug < transform_limit:
            for tid, (tr_img, tr_boxes) in enumerate(self._iter_translate_outputs(img, boxes, cfg), start=1):
                if generated_aug >= transform_limit:
                    break
                if not tr_boxes:
                    continue
                self._save_sample(out_images, out_labels, tr_img, tr_boxes, f"{base}__aug_translate_{tid}{ext}")
                generated_aug += 1
                generated_total += 1
                by_transform["translate"] += 1

        return {
            "generated_images": int(generated_total),
            "generated_labels": int(generated_total),
            "by_transform": by_transform,
        }

    # --------------------
    # Dataset source helpers
    # --------------------
    def _resolve_dataset_version_root(
        self,
        db: Session,
        dataset_id: int,
        version_id: int | None,
    ) -> Tuple[Dataset, DatasetVersion, Path]:
        ds = db.query(Dataset).filter(Dataset.dataset_id == int(dataset_id)).first()
        if not ds:
            raise NotFoundError("Dataset not found")
        if ds.dataset_type != DatasetType.DETECTION:
            raise ValidationError("Manual augmentation currently supports detection datasets only")
        if str(getattr(ds, "format", "") or "").lower() == "illegal":
            raise ValidationError("Illegal dataset must be converted before augmentation")

        ver: Optional[DatasetVersion] = None
        if version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(version_id), DatasetVersion.dataset_id == int(dataset_id))
                .first()
            )
            if not ver:
                raise NotFoundError("Dataset version not found")
        elif ds.active_version_id is not None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.version_id == int(ds.active_version_id), DatasetVersion.dataset_id == int(dataset_id))
                .first()
            )
        if ver is None:
            ver = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == int(dataset_id))
                .order_by(DatasetVersion.version.desc(), DatasetVersion.version_id.desc())
                .first()
            )
        if ver is None:
            raise ValidationError("Dataset has no available version")

        root = resolve_dataset_path(ver.snapshot_path or ds.storage_path)
        if not root.exists() or not root.is_dir():
            raise ValidationError(f"Dataset version path does not exist: {root}")
        return ds, ver, root

    def _collect_sources(self, root: Path) -> List[_SourceImage]:
        out: List[_SourceImage] = []
        skip_dirs = {".versions", ".thumbnails", "__pycache__"}
        all_images: List[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                rel_parts = p.relative_to(root).parts
            except Exception:
                continue
            if any(part in skip_dirs for part in rel_parts):
                continue
            all_images.append(p)
        all_images.sort(key=lambda x: x.as_posix().lower())

        for img_path in all_images:
            rel = img_path.relative_to(root).as_posix()
            label_path = self._guess_label_path(root, img_path)
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            out.append(
                _SourceImage(
                    rel_path=rel,
                    image_path=img_path,
                    label_path=label_path if (label_path and label_path.exists()) else None,
                    width=int(w),
                    height=int(h),
                )
            )
        return out

    def _guess_label_path(self, root: Path, image_path: Path) -> Optional[Path]:
        try:
            rel = image_path.relative_to(root)
        except Exception:
            return None
        parts = list(rel.parts)
        for i, part in enumerate(parts):
            if part.lower() == "images":
                parts2 = list(parts)
                parts2[i] = "labels"
                candidate = root / Path(*parts2)
                candidate = candidate.with_suffix(".txt")
                if candidate.exists():
                    return candidate
                break
        fallback = root / "labels" / f"{Path(rel).stem}.txt"
        if fallback.exists():
            return fallback
        return None

    # --------------------
    # Preview estimation
    # --------------------
    def _estimate_per_image(self, cfg: DatasetAugmentationConfig, width: int, height: int) -> Dict[str, int]:
        slice_count = 0
        if cfg.slice.enabled:
            for s in cfg.slice.scales:
                scale = max(32, int(s))
                x_positions = self._sliding_positions(width, scale, cfg.slice.overlap)
                y_positions = self._sliding_positions(height, scale, cfg.slice.overlap)
                slice_count += int(len(x_positions) * len(y_positions))
        rotate_count = len(cfg.rotate.angles) if cfg.rotate.enabled else 0
        translate_count = len(cfg.translate.offsets) if cfg.translate.enabled else 0

        remaining = max(1, int(cfg.max_outputs_per_image))
        kept_slice = min(slice_count, remaining)
        remaining -= kept_slice
        kept_rotate = min(rotate_count, max(0, remaining))
        remaining -= kept_rotate
        kept_translate = min(translate_count, max(0, remaining))
        return {"slice": int(kept_slice), "rotate": int(kept_rotate), "translate": int(kept_translate)}

    # --------------------
    # Transform primitives
    # --------------------
    def _read_yolo_boxes(self, label_path: Optional[Path], width: int, height: int) -> List[_Box]:
        if label_path is None or not label_path.exists() or not label_path.is_file():
            return []
        out: List[_Box] = []
        try:
            text = label_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx = float(parts[1]) * width
                cy = float(parts[2]) * height
                bw = float(parts[3]) * width
                bh = float(parts[4]) * height
            except Exception:
                continue
            box = _Box(class_id=cls, x1=cx - bw / 2.0, y1=cy - bh / 2.0, x2=cx + bw / 2.0, y2=cy + bh / 2.0)
            clipped = self._clip_box(box, width, height)
            if clipped is not None and clipped.area > 0:
                out.append(clipped)
        return out

    def _clip_box(self, box: _Box, width: int, height: int) -> Optional[_Box]:
        x1 = max(0.0, min(float(width), float(box.x1)))
        y1 = max(0.0, min(float(height), float(box.y1)))
        x2 = max(0.0, min(float(width), float(box.x2)))
        y2 = max(0.0, min(float(height), float(box.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return _Box(class_id=int(box.class_id), x1=x1, y1=y1, x2=x2, y2=y2)

    def _keep_box(self, *, old_box: _Box, new_box: _Box, min_area_ratio: float, min_visibility: float, min_pixel_size: int) -> bool:
        if old_box.area <= 0:
            return False
        if new_box.w < float(min_pixel_size) or new_box.h < float(min_pixel_size):
            return False
        ratio = new_box.area / old_box.area
        return ratio >= float(min_area_ratio) and ratio >= float(min_visibility)

    def _sliding_positions(self, total: int, size: int, overlap: float) -> List[int]:
        if total <= size:
            return [0]
        step = max(1, int(round(size * (1.0 - float(overlap)))))
        out = [0]
        pos = 0
        while True:
            nxt = pos + step
            if nxt + size >= total:
                break
            out.append(int(nxt))
            pos = nxt
        last = max(0, int(total - size))
        if out[-1] != last:
            out.append(last)
        return out

    def _iter_slice_outputs(
        self,
        img: Image.Image,
        boxes: List[_Box],
        cfg: DatasetAugmentationConfig,
    ) -> Iterable[Tuple[Image.Image, List[_Box]]]:
        w, h = img.size
        for s in cfg.slice.scales:
            scale = max(32, int(s))
            xs = self._sliding_positions(w, scale, cfg.slice.overlap)
            ys = self._sliding_positions(h, scale, cfg.slice.overlap)
            for y0 in ys:
                for x0 in xs:
                    x1 = min(w, x0 + scale)
                    y1 = min(h, y0 + scale)
                    patch = img.crop((x0, y0, x1, y1))
                    pb: List[_Box] = []
                    for b in boxes:
                        inter = _Box(
                            class_id=b.class_id,
                            x1=max(b.x1, x0),
                            y1=max(b.y1, y0),
                            x2=min(b.x2, x1),
                            y2=min(b.y2, y1),
                        )
                        inter = self._clip_box(inter, x1, y1)
                        if inter is None:
                            continue
                        shifted = _Box(
                            class_id=inter.class_id,
                            x1=inter.x1 - x0,
                            y1=inter.y1 - y0,
                            x2=inter.x2 - x0,
                            y2=inter.y2 - y0,
                        )
                        shifted = self._clip_box(shifted, int(x1 - x0), int(y1 - y0))
                        if shifted is None:
                            continue
                        if not self._keep_box(
                            old_box=b,
                            new_box=shifted,
                            min_area_ratio=cfg.slice.min_area_ratio,
                            min_visibility=cfg.slice.min_visibility,
                            min_pixel_size=cfg.slice.min_pixel_size,
                        ):
                            continue
                        pb.append(shifted)
                    yield patch, pb

    def _iter_rotate_outputs(
        self,
        img: Image.Image,
        boxes: List[_Box],
        cfg: DatasetAugmentationConfig,
    ) -> Iterable[Tuple[Image.Image, List[_Box]]]:
        w, h = img.size
        cx, cy = w / 2.0, h / 2.0
        border = self._border_color(img.mode, int(cfg.rotate.border_value))
        for angle in cfg.rotate.angles:
            rad = math.radians(float(angle))
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            rot = img.rotate(float(angle), expand=True, fillcolor=border)
            rw, rh = rot.size
            ncx, ncy = rw / 2.0, rh / 2.0
            rb: List[_Box] = []
            for b in boxes:
                points = [(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)]
                tx: List[float] = []
                ty: List[float] = []
                for px, py in points:
                    qx = (px - cx) * cos_a - (py - cy) * sin_a + ncx
                    qy = (px - cx) * sin_a + (py - cy) * cos_a + ncy
                    tx.append(qx)
                    ty.append(qy)
                nb = _Box(class_id=b.class_id, x1=min(tx), y1=min(ty), x2=max(tx), y2=max(ty))
                nb = self._clip_box(nb, rw, rh)
                if nb is None:
                    continue
                if not self._keep_box(
                    old_box=b,
                    new_box=nb,
                    min_area_ratio=cfg.slice.min_area_ratio,
                    min_visibility=cfg.slice.min_visibility,
                    min_pixel_size=cfg.slice.min_pixel_size,
                ):
                    continue
                rb.append(nb)
            yield rot, rb

    def _iter_translate_outputs(
        self,
        img: Image.Image,
        boxes: List[_Box],
        cfg: DatasetAugmentationConfig,
    ) -> Iterable[Tuple[Image.Image, List[_Box]]]:
        w, h = img.size
        border = self._border_color(img.mode, int(cfg.translate.border_value))
        for off in cfg.translate.offsets:
            dx = int(round(float(off.dx) * w))
            dy = int(round(float(off.dy) * h))
            canvas = Image.new(img.mode, (w, h), border)
            canvas.paste(img, (dx, dy))
            tb: List[_Box] = []
            for b in boxes:
                nb = _Box(class_id=b.class_id, x1=b.x1 + dx, y1=b.y1 + dy, x2=b.x2 + dx, y2=b.y2 + dy)
                nb = self._clip_box(nb, w, h)
                if nb is None:
                    continue
                if not self._keep_box(
                    old_box=b,
                    new_box=nb,
                    min_area_ratio=cfg.slice.min_area_ratio,
                    min_visibility=cfg.slice.min_visibility,
                    min_pixel_size=cfg.slice.min_pixel_size,
                ):
                    continue
                tb.append(nb)
            yield canvas, tb

    def _border_color(self, mode: str, v: int):
        x = max(0, min(255, int(v)))
        mode_u = str(mode or "").upper()
        if mode_u in {"L", "P", "1", "I", "F"}:
            return x
        if mode_u in {"RGBA", "LA"}:
            return (x, x, x, 255)
        return (x, x, x)

    def _save_sample(
        self,
        out_images: Path,
        out_labels: Path,
        image: Image.Image,
        boxes: List[_Box],
        filename: str,
    ) -> None:
        img_path = out_images / filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(img_path)

        lbl_path = out_labels / (Path(filename).stem + ".txt")
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        iw, ih = image.size
        lines: List[str] = []
        for b in boxes:
            c = (b.x1 + b.x2) / 2.0 / iw
            d = (b.y1 + b.y2) / 2.0 / ih
            w = b.w / iw
            h = b.h / ih
            lines.append(f"{int(b.class_id)} {c:.6f} {d:.6f} {w:.6f} {h:.6f}")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

    # --------------------
    # Metadata / class helpers
    # --------------------
    def _resolve_class_names(self, root: Path, ds: Dataset, sources: List[_SourceImage]) -> List[str]:
        yaml_path = find_yolo_dataset_yaml(root, dataset_name=str(getattr(ds, "name", "") or "") or None)
        if yaml_path is not None and yaml_path.exists():
            try:
                cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                cfg = {}
            if isinstance(cfg, dict):
                names = self._normalize_names(cfg.get("names"), cfg.get("nc"))
                if names:
                    return names

        for name in ("classes.txt", "class_names.txt", "obj.names", "names.txt"):
            p = root / name
            if p.exists() and p.is_file():
                lines = [x.strip() for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
                if lines:
                    return lines

        max_cls = -1
        for src in sources:
            for b in self._read_yolo_boxes(src.label_path, src.width, src.height):
                max_cls = max(max_cls, int(b.class_id))
        if max_cls < 0:
            return ["class_0"]
        return [f"class_{i}" for i in range(max_cls + 1)]

    def _normalize_names(self, names_raw: Any, nc_raw: Any) -> List[str]:
        names: List[str] = []
        if isinstance(names_raw, dict):
            pairs: List[Tuple[int, str]] = []
            for k, v in names_raw.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                val = str(v).strip()
                if not val:
                    continue
                pairs.append((idx, val))
            if pairs:
                max_idx = max(i for i, _ in pairs)
                names = [f"class_{i}" for i in range(max_idx + 1)]
                for i, val in pairs:
                    names[i] = val
        elif isinstance(names_raw, list):
            names = [str(x).strip() for x in names_raw if str(x).strip()]

        if not names:
            try:
                nc = int(nc_raw)
            except Exception:
                nc = 0
            if nc > 0:
                names = [f"class_{i}" for i in range(nc)]
        return names

    def _write_output_metadata(self, out_dir: Path, names: List[str]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        classes = [str(x).strip() for x in (names or []) if str(x).strip()]
        if not classes:
            classes = ["class_0"]
        (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        (out_dir / "class_names.txt").write_text("\n".join(classes), encoding="utf-8")
        cfg = {
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": int(len(classes)),
            "names": classes,
        }
        with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # --------------------
    # Active-job / cleanup
    # --------------------
    def _find_active_job(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        root = self.jobs_root(int(dataset_id))
        now = self._utcnow()
        for status_file in root.glob("*/status.json"):
            try:
                data = self._read_json(status_file)
            except Exception:
                continue
            status = str(data.get("status") or "").strip().lower()
            if status not in self.ACTIVE_STATUSES:
                continue
            updated = self._parse_time(data.get("updated_at") or data.get("created_at"))
            if updated and (now - updated) > self.ACTIVE_STALE_AFTER:
                continue
            return data
        return None

    def _cleanup_old_jobs(self, dataset_id: int) -> None:
        root = self.jobs_root(int(dataset_id))
        cutoff = self._utcnow() - timedelta(days=self.RETAIN_DAYS)
        for status_file in root.glob("*/status.json"):
            try:
                st = self._read_json(status_file)
            except Exception:
                continue
            status = str(st.get("status") or "").strip().lower()
            if status not in self.TERMINAL_STATUSES:
                continue
            updated = self._parse_time(st.get("updated_at") or st.get("created_at"))
            if updated is None or updated >= cutoff:
                continue
            try:
                shutil.rmtree(status_file.parent, ignore_errors=True)
            except Exception:
                continue

    # --------------------
    # Utilities
    # --------------------
    def _static_temp_url(self, path: Path) -> Optional[str]:
        try:
            rel = path.resolve(strict=False).relative_to(settings.temp_dir.resolve())
        except Exception:
            return None
        return f"/static/temp/{rel.as_posix()}"
