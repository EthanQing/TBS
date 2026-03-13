from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.enums import ModelStage, TrainingRunStatus
from train_platform.models.model_registry import ModelVersion
from train_platform.models.training_run import TrainingRun
from train_platform.schemas.v2.inference_jobs import (
    InferenceJobCreate,
    InferenceJobOut,
    InferenceModelCandidate,
)
from train_platform.services.inference_service import InferenceService
from train_platform.services.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_temp_path

_LOCKS_GUARD = threading.Lock()
_JOB_LOCKS: Dict[str, threading.Lock] = {}
_CREATE_JOB_PROCESS_LOCK = threading.Lock()


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
        raise ValidationError("Job not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValidationError(f"Failed to read job status: {type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("Invalid status payload")
    return data


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
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


class InferenceJobService:
    ACTIVE_STATUSES = {"queued", "running"}
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    ACTIVE_STALE_AFTER = timedelta(hours=2)
    CREATE_JOB_LOCK_TIMEOUT_SEC = float(os.getenv("INFERENCE_JOB_CREATE_LOCK_TIMEOUT_SEC", "5"))

    def __init__(self) -> None:
        self._infer = InferenceService()
        self._mv_svc = ModelVersionService()

    def jobs_root(self) -> Path:
        root = settings.temp_dir / "inference_jobs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _create_job_lock_path(self) -> Path:
        return self.jobs_root() / ".create_job.lock"

    def _acquire_create_job_lock(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        lock_path = self._create_job_lock_path()

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{os.getpid()}\n{time.time()}\n")
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ConflictError("Another inference job request is being processed, please retry")
                # stale lock recovery
                try:
                    if lock_path.exists():
                        age = time.time() - float(lock_path.stat().st_mtime)
                        if age > max(30.0, float(timeout_sec) * 3.0):
                            lock_path.unlink(missing_ok=True)
                            continue
                except Exception:
                    pass
                time.sleep(0.05)

    def _release_create_job_lock(self) -> None:
        try:
            self._create_job_lock_path().unlink(missing_ok=True)
        except Exception:
            pass

    def job_dir(self, job_id: str) -> Path:
        d = self.jobs_root() / str(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def results_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "results.jsonl"

    def _resolve_architecture(self, db: Session, run: TrainingRun | None) -> ModelArchitecture | None:
        if not run:
            return None
        return db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()

    def _resolve_run(self, db: Session, run_id: str | None) -> TrainingRun | None:
        if not run_id:
            return None
        return db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()

    def _resolve_model_version(self, db: Session, model_version_id: int) -> ModelVersion:
        row = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if not row:
            raise NotFoundError("Model version not found")
        return row

    def _weights_ext_ok(self, engine: str, weights_path: Path) -> bool:
        ext = weights_path.suffix.lower()
        if engine == "paddle-det":
            return ext == ".pdparams"
        return ext in {".pt", ".pth"}

    def _resolve_paddle_config(self, arch: ModelArchitecture | None) -> Optional[Path]:
        if not arch:
            return None
        params = arch.default_params if isinstance(arch.default_params, dict) else {}
        raw = params.get("config_path")
        if not raw:
            return None

        txt = str(raw).strip().replace("\\", "/")
        if not txt:
            return None
        p = Path(txt)
        if p.is_absolute() and p.exists():
            return p.resolve(strict=False)

        candidates = [
            (settings.paddle_det_dir / txt).resolve(strict=False),
            (settings.home_dir / txt).resolve(strict=False),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _build_candidate(
        self,
        *,
        source: str,
        model_version: ModelVersion | None,
        run: TrainingRun,
        arch: ModelArchitecture | None,
    ) -> Optional[InferenceModelCandidate]:
        engine = str(getattr(arch, "engine", "") or "ultralytics-yolo").strip().lower()
        family = str(getattr(arch, "family", "") or "").strip() or None
        variant = str(getattr(arch, "variant", "") or "").strip() or None

        weights_rel = None
        if model_version and model_version.weights_path:
            weights_rel = model_version.weights_path
        elif run.result:
            weights_rel = run.result.best_weights_path or run.result.last_weights_path
        if not weights_rel:
            return None

        from train_platform.utils.path_utils import resolve_training_path

        weights_abs = resolve_training_path(weights_rel)
        if not weights_abs.exists() or not weights_abs.is_file():
            return None
        if not self._weights_ext_ok(engine, weights_abs):
            return None

        config_path = None
        if engine == "paddle-det":
            cfg = self._resolve_paddle_config(arch)
            if cfg is None:
                return None
            config_path = str(cfg)

        label_parts = []
        if family:
            label_parts.append(family)
        if variant:
            label_parts.append(variant)
        if model_version and model_version.version:
            label_parts.append(f"v:{model_version.version}")
        else:
            label_parts.append(f"run:{str(run.run_id)[:8]}")
        label = " / ".join(label_parts)

        created_at = None
        if model_version and getattr(model_version, "created_at", None):
            created_at = model_version.created_at
        elif getattr(run, "finished_at", None):
            created_at = run.finished_at
        else:
            created_at = getattr(run, "created_at", None)

        return InferenceModelCandidate(
            source="model_version" if source == "model_version" else "training_run",
            model_version_id=int(model_version.model_version_id) if model_version else None,
            run_id=str(run.run_id),
            project_id=int(run.project_id),
            architecture_id=int(run.architecture_id),
            engine=engine,
            family=family,
            variant=variant,
            version=str(model_version.version) if model_version else None,
            label=label,
            weights_path=str(weights_rel),
            config_path=config_path,
            inferable=True,
            created_at=created_at,
        )

    def list_inferable_models(self, db: Session, *, project_id: int | None = None) -> List[InferenceModelCandidate]:
        q_runs = db.query(TrainingRun).filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
        if project_id is not None:
            q_runs = q_runs.filter(TrainingRun.project_id == int(project_id))
        runs = q_runs.all()
        run_map = {str(r.run_id): r for r in runs}
        arch_ids = {int(r.architecture_id) for r in runs}
        arch_rows = []
        if arch_ids:
            arch_rows = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id.in_(sorted(arch_ids))).all()
        arch_map = {int(a.architecture_id): a for a in arch_rows}

        q_mvs = db.query(ModelVersion)
        if project_id is not None:
            q_mvs = q_mvs.filter(ModelVersion.project_id == int(project_id))
        mvs = q_mvs.order_by(ModelVersion.created_at.desc()).all()

        out: List[InferenceModelCandidate] = []
        run_with_mv: set[str] = set()

        for mv in mvs:
            run = run_map.get(str(mv.run_id))
            if not run:
                continue
            arch = arch_map.get(int(run.architecture_id))
            cand = self._build_candidate(source="model_version", model_version=mv, run=run, arch=arch)
            if not cand:
                continue
            out.append(cand)
            run_with_mv.add(str(run.run_id))

        for run in runs:
            rid = str(run.run_id)
            if rid in run_with_mv:
                continue
            arch = arch_map.get(int(run.architecture_id))
            cand = self._build_candidate(source="training_run", model_version=None, run=run, arch=arch)
            if not cand:
                continue
            out.append(cand)

        out.sort(key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return out

    def _new_job_id(self) -> str:
        return uuid.uuid4().hex

    def _has_active_job(self) -> Optional[Dict[str, Any]]:
        root = self.jobs_root()
        now = _utcnow()
        for status_file in root.glob("*/status.json"):
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

    def _ensure_model_version_for_payload(
        self, db: Session, *, model_version_id: int | None, run_id: str | None
    ) -> int:
        if model_version_id is not None:
            mv = self._resolve_model_version(db, int(model_version_id))
            return int(mv.model_version_id)

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

        run = self._resolve_run(db, rid)
        if not run:
            raise NotFoundError("Training run not found")
        if run.status != TrainingRunStatus.COMPLETED:
            raise ConflictError("Only completed runs can be used for inference")

        base = f"run-{rid[:8]}"
        for i in range(1, 200):
            version = base if i == 1 else f"{base}-{i}"
            try:
                mv = self._mv_svc.register_from_run(
                    db,
                    run_id=rid,
                    version=version,
                    stage=ModelStage.DEVELOPMENT,
                    description="Auto-created for inference jobs",
                )
                return int(mv.model_version_id)
            except ConflictError:
                continue
        raise ConflictError("Failed to auto-register model version for run")

    def _normalize_inputs(self, payload: InferenceJobCreate) -> tuple[list[str], Optional[str]]:
        if payload.mode == "video":
            return [], str(payload.video_token or "").strip()

        tokens = [str(x).strip() for x in payload.input_tokens if str(x).strip()]
        if not tokens:
            raise ValidationError("No input tokens provided")
        if payload.mode == "image":
            tokens = tokens[:1]
        return tokens, None

    def create_job(self, db: Session, payload: InferenceJobCreate) -> InferenceJobOut:
        with _CREATE_JOB_PROCESS_LOCK:
            self._acquire_create_job_lock(self.CREATE_JOB_LOCK_TIMEOUT_SEC)
            try:
                active = self._has_active_job()
                if active:
                    jid = active.get("job_id") or "unknown"
                    st = active.get("status") or "running"
                    raise ConflictError(f"Another inference job is active (job_id={jid}, status={st})")

                mv_id = self._ensure_model_version_for_payload(
                    db, model_version_id=payload.model_version_id, run_id=payload.run_id
                )
                ctx = self._infer.resolve_model_context(db, model_version_id=mv_id)

                tokens, video_token = self._normalize_inputs(payload)
                mode = str(payload.mode)
                total = len(tokens)
                if mode == "video":
                    total = self._probe_total_frames(video_token)

                job_id = self._new_job_id()
                self.job_dir(job_id)

                status: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": "queued",
                    "phase": "preparing",
                    "mode": mode,
                    "progress": 0,
                    "processed": 0,
                    "total": int(total),
                    "seq": 1,
                    "last_result_id": 0,
                    "model_version_id": int(mv_id),
                    "run_id": str(ctx.get("run_id") or payload.run_id or ""),
                    "engine": str(ctx.get("engine") or ""),
                    "family": str(ctx.get("family") or "") or None,
                    "variant": str(ctx.get("variant") or "") or None,
                    "conf": float(payload.conf),
                    "iou": float(payload.iou),
                    "show_labels": bool(payload.show_labels),
                    "show_confidence": bool(payload.show_confidence),
                    "input_tokens": tokens,
                    "video_token": video_token,
                    "cancel_requested": False,
                    "result": {"mode": mode},
                    "error_message": None,
                    "created_at": _to_iso(),
                    "updated_at": _to_iso(),
                }
                _write_json_atomic(self.status_path(job_id), status)
            finally:
                self._release_create_job_lock()

        t = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        t.start()
        return self.get_job(job_id, include_items=False)

    def _probe_total_frames(self, video_token: str | None) -> int:
        raw = str(video_token or "").strip()
        if not raw:
            return 0
        try:
            import cv2
        except Exception:
            return 0
        path = resolve_temp_path(raw)
        if not path.exists():
            return 0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return max(0, total)
        finally:
            cap.release()

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
            st = self._read_status(job_id)
        except Exception:
            return True
        return bool(st.get("cancel_requested"))

    def cancel_job(self, job_id: str) -> InferenceJobOut:
        st = self._update_status(job_id, {"cancel_requested": True}, bump_seq=True)
        if str(st.get("status")) == "queued":
            st = self._update_status(
                job_id,
                {"status": "cancelled", "phase": "cancelled", "progress": int(st.get("progress") or 0)},
                bump_seq=True,
            )
        return self.get_job(job_id, include_items=False)

    def get_job(self, job_id: str, *, include_items: bool = True) -> InferenceJobOut:
        st = self._read_status(job_id)
        result = st.get("result") if isinstance(st.get("result"), dict) else {"mode": st.get("mode")}
        if include_items and str(st.get("mode")) in {"image", "batch"}:
            result = dict(result or {})
            result.setdefault("mode", st.get("mode"))
            items = self.read_results_since(job_id, after_result_id=0)
            result["items"] = items

        payload = dict(st)
        payload["result"] = result
        return InferenceJobOut.model_validate(payload)

    def _run_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            st = self._update_status(
                job_id,
                {"status": "running", "phase": "preparing", "progress": 0, "error_message": None},
                bump_seq=True,
            )
            if self._is_cancel_requested(job_id):
                self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
                return

            mode = str(st.get("mode") or "")
            if mode in {"image", "batch"}:
                self._run_image_job(db, job_id)
            elif mode == "video":
                self._run_video_job(db, job_id)
            else:
                raise ValidationError(f"Unsupported mode: {mode}")

            fin = self._read_status(job_id)
            if str(fin.get("status")) == "running":
                self._update_status(
                    job_id,
                    {"status": "completed", "phase": "done", "progress": 100, "error_message": None},
                    bump_seq=True,
                )
        except Exception as e:
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
        finally:
            db.close()

    def _static_temp_url(self, path: Path) -> Optional[str]:
        try:
            rel = path.resolve(strict=False).relative_to(settings.temp_dir.resolve())
            return f"/static/temp/{rel.as_posix()}"
        except Exception:
            return None

    def _draw_predictions(
        self,
        image: Any,
        predictions: List[Dict[str, Any]],
        *,
        show_labels: bool,
        show_confidence: bool,
    ) -> None:
        try:
            import cv2
        except Exception:
            return

        for pred in predictions:
            box = pred.get("xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(x))) for x in box]
            except Exception:
                continue
            cls_id = int(pred.get("class_id") or -1)
            color = (
                int((37 * (cls_id + 3)) % 255),
                int((17 * (cls_id + 7)) % 255),
                int((29 * (cls_id + 11)) % 255),
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if not show_labels and not show_confidence:
                continue
            parts = []
            if show_labels:
                parts.append(str(pred.get("class_name") or pred.get("class_id") or "obj"))
            if show_confidence:
                try:
                    parts.append(f"{float(pred.get('confidence') or 0):.3f}")
                except Exception:
                    pass
            if not parts:
                continue
            text = " ".join(parts)
            cv2.putText(
                image,
                text,
                (max(0, x1), max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

    def _render_image_result(
        self,
        job_id: str,
        *,
        source_path: Path,
        predictions: List[Dict[str, Any]],
        show_labels: bool,
        show_confidence: bool,
    ) -> Optional[str]:
        try:
            import cv2
        except Exception:
            return None
        img = cv2.imread(str(source_path))
        if img is None:
            return None
        self._draw_predictions(
            img,
            predictions,
            show_labels=show_labels,
            show_confidence=show_confidence,
        )

        out_dir = self.job_dir(job_id) / "output" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = source_path.stem or "image"
        name = f"{stem}_{int(time.time() * 1000)}.jpg"
        out_path = out_dir / name
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            return None
        return self._static_temp_url(out_path)

    def _run_image_job(self, db: Session, job_id: str) -> None:
        st = self._read_status(job_id)
        tokens = [str(x).strip() for x in (st.get("input_tokens") or []) if str(x).strip()]
        total = len(tokens)
        show_labels = bool(st.get("show_labels", True))
        show_conf = bool(st.get("show_confidence", True))
        mv_id = int(st.get("model_version_id"))
        conf = float(st.get("conf") or 0.5)
        iou = float(st.get("iou") or 0.45)

        self._update_status(job_id, {"phase": "inferring", "total": total, "processed": 0, "progress": 0}, bump_seq=True)

        for idx, token in enumerate(tokens, start=1):
            if self._is_cancel_requested(job_id):
                self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
                return

            info = self._infer.run_inference_output(
                db,
                model_version_id=mv_id,
                input_path=token,
                conf=conf,
                iou=iou,
            )
            output = info.get("output") if isinstance(info.get("output"), dict) else {}
            err = info.get("error_message")
            preds = output.get("predictions") if isinstance(output, dict) else []
            predictions = preds if isinstance(preds, list) else []

            src_path = resolve_temp_path(token)
            src_url = self._static_temp_url(src_path) if src_path.exists() else None
            out_url = None
            if src_path.exists() and predictions:
                out_url = self._render_image_result(
                    job_id,
                    source_path=src_path,
                    predictions=predictions,
                    show_labels=show_labels,
                    show_confidence=show_conf,
                )

            item = {
                "filename": Path(token).name,
                "token": token,
                "status": "failed" if err else "success",
                "detections": int(len(predictions)),
                "inference_time_ms": info.get("inference_time_ms"),
                "source_url": src_url,
                "output_url": out_url,
                "output": output if output else None,
                "error_message": str(err) if err else None,
            }
            self._append_item(job_id, item)

            progress = int((idx / total) * 100) if total > 0 else 100
            self._update_status(
                job_id,
                {
                    "processed": idx,
                    "total": total,
                    "progress": progress,
                    "phase": "inferring" if idx < total else "finalizing",
                },
                bump_seq=True,
            )

        self._update_status(
            job_id,
            {"result": {"mode": st.get("mode"), "items": []}, "phase": "finalizing"},
            bump_seq=True,
        )

    def _run_video_job(self, db: Session, job_id: str) -> None:
        st = self._read_status(job_id)
        video_token = str(st.get("video_token") or "").strip()
        mv_id = int(st.get("model_version_id"))
        conf = float(st.get("conf") or 0.5)
        iou = float(st.get("iou") or 0.45)
        show_labels = bool(st.get("show_labels", True))
        show_conf = bool(st.get("show_confidence", True))

        if not video_token:
            raise ValidationError("Missing video token")

        video_path = resolve_temp_path(video_token)
        if not video_path.exists() or not video_path.is_file():
            raise ValidationError(f"Video file not found: {video_token}")

        try:
            import cv2
        except Exception as e:
            raise ValidationError(f"OpenCV is required for video inference: {e}") from e

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValidationError(f"Failed to open video: {video_token}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0:
            fps = 25.0
        first_frame = None
        if width <= 0 or height <= 0:
            ok0, frame0 = cap.read()
            if not ok0:
                cap.release()
                raise ValidationError("Video has no readable frames")
            first_frame = frame0
            h0, w0 = frame0.shape[:2]
            width, height = int(w0), int(h0)

        out_dir = self.job_dir(job_id) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / "output.mp4"
        tmp_dir = self.job_dir(job_id) / "work"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        frame_tmp = tmp_dir / "frame.jpg"

        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (max(1, width), max(1, height)),
        )
        if not writer.isOpened():
            cap.release()
            raise ValidationError("Failed to create output video writer")

        start_t = time.perf_counter()
        processed = 0
        self._update_status(
            job_id,
            {"phase": "inferring", "total": max(0, total_frames), "processed": 0, "progress": 0},
            bump_seq=True,
        )

        try:
            if first_frame is not None:
                cv2.imwrite(str(frame_tmp), first_frame)
                info0 = self._infer.run_inference_output(
                    db,
                    model_version_id=mv_id,
                    input_path=str(frame_tmp),
                    conf=conf,
                    iou=iou,
                )
                output0 = info0.get("output") if isinstance(info0.get("output"), dict) else {}
                preds0 = output0.get("predictions") if isinstance(output0, dict) else []
                pred_list0 = preds0 if isinstance(preds0, list) else []
                if pred_list0:
                    self._draw_predictions(
                        first_frame,
                        pred_list0,
                        show_labels=show_labels,
                        show_confidence=show_conf,
                    )
                writer.write(first_frame)
                processed += 1
                progress0 = int((processed / total_frames) * 100) if total_frames > 0 else 0
                self._update_status(
                    job_id,
                    {
                        "processed": processed,
                        "total": max(total_frames, processed),
                        "progress": max(0, min(99, progress0)),
                        "phase": "inferring",
                    },
                    bump_seq=True,
                )

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self._is_cancel_requested(job_id):
                    self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
                    return

                cv2.imwrite(str(frame_tmp), frame)
                info = self._infer.run_inference_output(
                    db,
                    model_version_id=mv_id,
                    input_path=str(frame_tmp),
                    conf=conf,
                    iou=iou,
                )
                output = info.get("output") if isinstance(info.get("output"), dict) else {}
                preds = output.get("predictions") if isinstance(output, dict) else []
                predictions = preds if isinstance(preds, list) else []
                if predictions:
                    self._draw_predictions(
                        frame,
                        predictions,
                        show_labels=show_labels,
                        show_confidence=show_conf,
                    )
                writer.write(frame)

                processed += 1
                progress = int((processed / total_frames) * 100) if total_frames > 0 else 0
                self._update_status(
                    job_id,
                    {
                        "processed": processed,
                        "total": max(total_frames, processed),
                        "progress": max(0, min(99, progress)),
                        "phase": "inferring",
                    },
                    bump_seq=True,
                )
        finally:
            cap.release()
            writer.release()
            try:
                frame_tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass

        if self._is_cancel_requested(job_id):
            self._update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
            return

        elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        video_url = self._static_temp_url(out_video)
        if not video_url:
            raise ValidationError("Failed to resolve output video URL")
        if not out_video.exists() or out_video.stat().st_size <= 0:
            raise ValidationError("Output video was not generated")

        self._update_status(
            job_id,
            {
                "phase": "finalizing",
                "progress": 100,
                "processed": processed,
                "total": max(total_frames, processed),
                "result": {
                    "mode": "video",
                    "video": {
                        "output_url": video_url,
                        "total_frames": max(total_frames, processed),
                        "processed_frames": processed,
                        "fps": round(float(fps), 3),
                        "width": int(width) if width > 0 else None,
                        "height": int(height) if height > 0 else None,
                        "total_time_ms": elapsed_ms,
                    },
                },
            },
            bump_seq=True,
        )

    def list_jobs_for_debug(self) -> List[Dict[str, Any]]:
        out = []
        for status_file in self.jobs_root().glob("*/status.json"):
            try:
                out.append(_read_json(status_file))
            except Exception:
                continue
        return out
