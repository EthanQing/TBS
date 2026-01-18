from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, TextIO

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.enums import DeploymentStatus, LogLevel, TrainingRunStatus
from train_platform.models.training_run import TrainingRun, TrainingRunArtifact, TrainingRunEvent, TrainingRunResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_remove_dir(path: Path) -> None:
    import shutil

    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _add_event(db: Session, run_id: str, event_type: str, message: str, *, level: LogLevel = LogLevel.INFO) -> None:
    db.add(TrainingRunEvent(run_id=run_id, level=level, event_type=event_type, message=message))


def _spawn_training_subprocess(run_id: str, *, stdout_f: TextIO, stderr_f: TextIO) -> subprocess.Popen:
    args = [sys.executable, "-m", "train_platform.workers.training.train_entry", "--run-id", run_id]

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f, creationflags=creationflags)

    return subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f, start_new_session=True)


def _terminate_process_tree(proc: subprocess.Popen, *, timeout_sec: int = 20) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _index_completion_artifacts(db: Session, run_id: str) -> None:
    """
    Index common Ultralytics artifacts for UI download/inspection.

    This function is idempotent.
    """
    base = settings.training_dir
    run_dir = base / run_id

    db.query(TrainingRunArtifact).filter(TrainingRunArtifact.run_id == run_id).delete()

    candidates: list[tuple[str, str, Path]] = [
        ("weights", "best.pt", run_dir / "weights" / "best.pt"),
        ("weights", "last.pt", run_dir / "weights" / "last.pt"),
        ("csv", "results.csv", run_dir / "results.csv"),
        ("config", "args.yaml", run_dir / "args.yaml"),
        ("config", "results.yaml", run_dir / "results.yaml"),
        ("log", "train.stdout.log", run_dir / "logs" / "train.stdout.log"),
        ("log", "train.stderr.log", run_dir / "logs" / "train.stderr.log"),
    ]

    plot_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    for name in plot_names:
        candidates.append(("plot", name, run_dir / name))

    for kind, name, abs_path in candidates:
        if not abs_path.exists() or not abs_path.is_file():
            continue
        try:
            rel = abs_path.relative_to(base).as_posix()
        except Exception:
            rel = str(abs_path)

        size_bytes = None
        try:
            size_bytes = int(abs_path.stat().st_size)
        except Exception:
            size_bytes = None

        db.add(
            TrainingRunArtifact(
                run_id=run_id,
                kind=kind,
                name=name,
                path=rel,
                size_bytes=size_bytes,
            )
        )

    # Update/Upsert TrainingRunResult for compatibility with model registry.
    res = db.query(TrainingRunResult).filter(TrainingRunResult.run_id == run_id).first()
    if not res:
        res = TrainingRunResult(run_id=run_id)
        db.add(res)

    res.results_dir = run_id

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    res.best_weights_path = best.relative_to(base).as_posix() if best.exists() else None
    res.last_weights_path = last.relative_to(base).as_posix() if last.exists() else None

    size_source = best if best.exists() else last if last.exists() else None
    if size_source and size_source.exists():
        try:
            res.model_size_mb = round(size_source.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass


@dataclass
class RunningJob:
    run_id: str
    proc: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    stdout_f: TextIO
    stderr_f: TextIO


class DbQueueWorker:
    def __init__(self) -> None:
        self.worker_id = os.getenv("WORKER_ID") or uuid.uuid4().hex
        self.poll_interval = float(os.getenv("WORKER_POLL_INTERVAL", "2"))
        self.heartbeat_interval = float(os.getenv("WORKER_HEARTBEAT_INTERVAL", "5"))
        self.stale_after = int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120"))

        self._running: Optional[RunningJob] = None
        self._last_heartbeat_at: Optional[datetime] = None

    def run_forever(self) -> None:
        print(f"[worker] starting worker_id={self.worker_id}", flush=True)
        settings.ensure_dirs()
        while True:
            try:
                self.tick()
            except Exception as e:
                print(f"[worker] tick error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            time.sleep(self.poll_interval)

    def tick(self) -> None:
        if self._running is not None:
            self._tick_running()
            return
        self._try_start_next_run()

    def _tick_running(self) -> None:
        assert self._running is not None
        run_id = self._running.run_id

        db = SessionLocal()
        try:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                _terminate_process_tree(self._running.proc)
                self._cleanup_running()
                return

            # Heartbeat
            now = _utcnow()
            if self._last_heartbeat_at is None or (now - self._last_heartbeat_at).total_seconds() >= self.heartbeat_interval:
                run.heartbeat_at = now
                db.commit()
                self._last_heartbeat_at = now

            # Cancel / delete request handling
            cancel_requested = bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
            if cancel_requested and self._running.proc.poll() is None:
                _add_event(db, run_id, "cancel", "Terminating training subprocess due to cancel/delete request")
                db.commit()
                _terminate_process_tree(self._running.proc)

            rc = self._running.proc.poll()
            if rc is None:
                return

            # Subprocess ended
            run.finished_at = now

            delete_requested = run.delete_requested_at is not None
            if delete_requested:
                run.status = TrainingRunStatus.DELETED
                run.hidden = True
                _add_event(db, run_id, "deleted", "Run marked as deleted")
            elif run.cancel_requested_at is not None:
                run.status = TrainingRunStatus.CANCELLED
                _add_event(db, run_id, "cancelled", "Run cancelled")
            elif rc == 0:
                run.status = TrainingRunStatus.COMPLETED
                _add_event(db, run_id, "completed", "Run completed")
            else:
                run.status = TrainingRunStatus.FAILED
                run.error_message = f"Training subprocess exited with code {rc}"
                _add_event(db, run_id, "failed", run.error_message, level=LogLevel.ERROR)

            # Index artifacts (best-effort) and persist results.
            try:
                _index_completion_artifacts(db, run_id)
            except Exception:
                pass

            # Release claim
            run.worker_id = None
            run.claimed_at = None
            run.pid = None
            run.heartbeat_at = None

            db.commit()

            # Optionally cleanup files if deleted
            if delete_requested:
                _safe_remove_dir(settings.training_dir / run_id)

        finally:
            db.close()
            self._cleanup_running()

    def _cleanup_running(self) -> None:
        if self._running is None:
            return
        try:
            self._running.stdout_f.close()
        except Exception:
            pass
        try:
            self._running.stderr_f.close()
        except Exception:
            pass
        self._running = None
        self._last_heartbeat_at = None

    def _try_start_next_run(self) -> None:
        db = SessionLocal()
        try:
            self._reconcile_stale_claims(db)

            now = _utcnow()
            q = (
                db.query(TrainingRun)
                .filter(TrainingRun.status == TrainingRunStatus.QUEUED)
                .filter(TrainingRun.queued_at.isnot(None))
                .filter(TrainingRun.claimed_at.is_(None))
                .filter(TrainingRun.hidden == False)  # noqa: E712
                .order_by(TrainingRun.queued_at.asc())
            )

            # Best-effort row locking for multi-worker.
            try:
                q = q.with_for_update(skip_locked=True)
            except Exception:
                pass

            run = q.first()
            if not run:
                return

            # Prepare log files
            run_dir = settings.training_dir / run.run_id
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            stdout_path = logs_dir / "train.stdout.log"
            stderr_path = logs_dir / "train.stderr.log"

            stdout_f = open(stdout_path, "a", encoding="utf-8", buffering=1)
            stderr_f = open(stderr_path, "a", encoding="utf-8", buffering=1)

            proc = _spawn_training_subprocess(run.run_id, stdout_f=stdout_f, stderr_f=stderr_f)

            run.claimed_at = now
            run.worker_id = self.worker_id
            run.pid = int(proc.pid)
            run.heartbeat_at = now
            run.started_at = now
            run.status = TrainingRunStatus.RUNNING
            run.run_dir = run.run_dir or run.run_id

            _add_event(db, run.run_id, "started", f"Run started by worker {self.worker_id}")
            db.commit()

            self._running = RunningJob(
                run_id=run.run_id,
                proc=proc,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                stdout_f=stdout_f,
                stderr_f=stderr_f,
            )
            self._last_heartbeat_at = now

        finally:
            db.close()

    def _reconcile_stale_claims(self, db: Session) -> None:
        now = _utcnow()
        threshold = now - timedelta(seconds=self.stale_after)

        stale_queued = (
            db.query(TrainingRun)
            .filter(TrainingRun.status == TrainingRunStatus.QUEUED)
            .filter(TrainingRun.queued_at.isnot(None))
            .filter(TrainingRun.worker_id.isnot(None))
            .filter(
                (TrainingRun.heartbeat_at.is_(None) & (TrainingRun.claimed_at < threshold))
                | (TrainingRun.heartbeat_at < threshold)
            )
            .all()
        )
        for run in stale_queued:
            _add_event(db, run.run_id, "requeue", "Released stale claim; re-queued for another worker")
            run.worker_id = None
            run.claimed_at = None
            run.pid = None
            run.heartbeat_at = None

        stale_running = (
            db.query(TrainingRun)
            .filter(TrainingRun.status == TrainingRunStatus.RUNNING)
            .filter(TrainingRun.worker_id.isnot(None))
            .filter(
                (TrainingRun.heartbeat_at.is_(None) & (TrainingRun.started_at < threshold))
                | (TrainingRun.heartbeat_at < threshold)
            )
            .all()
        )
        for run in stale_running:
            run.status = TrainingRunStatus.FAILED
            run.finished_at = now
            run.error_message = "Worker heartbeat lost; marking as failed"
            _add_event(db, run.run_id, "failed", run.error_message, level=LogLevel.ERROR)
            run.worker_id = None
            run.claimed_at = None
            run.pid = None
            run.heartbeat_at = None

        if stale_queued or stale_running:
            db.commit()


def main() -> None:
    DbQueueWorker().run_forever()


if __name__ == "__main__":
    main()

