from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import ModelStage, TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.schemas.v3.inference_jobs import (
    InferenceJobCreate,
    InferenceJobOut,
    InferenceModelCandidate,
)
from train_platform.services.v3.inference_service import InferenceService
from train_platform.services.v3.model_version_service import ModelVersionService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

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

    def _worker_url_for_engine(self, engine: str) -> str:
        e = str(engine or "").strip().lower()
        if e == "paddle-det":
            return os.getenv("PADDLE_INFERENCE_WORKER_URL", "http://127.0.0.1:18003").rstrip("/")
        return os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")

    def _internal_request_headers(self) -> Dict[str, str]:
        token = str(settings.internal_api_token or "").strip()
        if not token:
            return {}
        return {"X-Internal-Token": token}

    def _dispatch_job_to_worker(self, job_id: str, *, status: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        engine = str(ctx.get("engine") or status.get("engine") or "ultralytics-yolo").strip().lower()
        worker_url = self._worker_url_for_engine(engine)
        timeout = float(os.getenv("INFERENCE_JOB_WORKER_DISPATCH_TIMEOUT", "10"))

        payload: Dict[str, Any] = {
            "job_id": job_id,
            "mode": str(status.get("mode") or "image"),
            "weights_path": str(ctx.get("weights_path") or ""),
            "input_tokens": list(status.get("input_tokens") or []),
            "video_token": status.get("video_token"),
            "conf": float(status.get("conf") or 0.5),
            "iou": float(status.get("iou") or 0.45),
            "show_labels": bool(status.get("show_labels", True)),
            "show_confidence": bool(status.get("show_confidence", True)),
        }
        if engine == "paddle-det":
            payload["config_path"] = str(ctx.get("config_path") or "")

        resp = requests.post(
            f"{worker_url}/internal/inference-jobs/run",
            json=payload,
            timeout=timeout,
            headers=self._internal_request_headers(),
        )
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        if resp.status_code != 200:
            msg = ""
            if isinstance(data, dict):
                msg = str(data.get("error") or data.get("detail") or "").strip()
            raise RuntimeError(msg or f"Inference worker error {resp.status_code}: {resp.text}")
        worker_status = str((data or {}).get("status") or "").strip().lower()
        worker_error = str((data or {}).get("error") or "").strip()
        if worker_status not in {"started", "ok"}:
            raise RuntimeError(worker_error or f"Inference worker returned status={worker_status or 'unknown'}")

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
                    total = 0

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

        try:
            self._dispatch_job_to_worker(job_id, status=status, ctx=ctx)
        except Exception as e:
            self._update_status(
                job_id,
                {
                    "status": "failed",
                    "phase": "failed",
                    "progress": 100,
                    "error_message": f"Failed to dispatch inference job to worker: {type(e).__name__}: {e}",
                },
                bump_seq=True,
            )
        return self.get_job(job_id, include_items=False)

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

    def list_jobs_for_debug(self) -> List[Dict[str, Any]]:
        out = []
        for status_file in self.jobs_root().glob("*/status.json"):
            try:
                out.append(_read_json(status_file))
            except Exception:
                continue
        return out
