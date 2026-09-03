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
from train_platform.core.license import assert_valid_license
from train_platform.db.session import SessionLocal
from train_platform.domains.training.runs import finalize_execution, mark_started, release_stale_claim, touch_heartbeat
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun
from train_platform.services.v3.alarm_service import AlarmService
from train_platform.utils.training_params import parse_visible_host_gpu_ids, worker_can_run_device


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_remove_dir(path: Path) -> None:
    import shutil

    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


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


def _parse_worker_engines(raw: Optional[str]) -> Optional[set[str]]:
    """
    Parse WORKER_ENGINES env var.

    Examples:
      - "ultralytics-yolo,paddle-det"
      - "all" / "*" (or empty) => no filtering.
    """
    value = (raw or "").strip()
    if not value or value in {"*", "all", "ALL", "All"}:
        return None
    engines = {x.strip().lower() for x in value.split(",") if x.strip()}
    return engines or None



@dataclass
class RunningJob:
    run_id: str
    proc: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    stdout_f: TextIO
    stderr_f: TextIO


class DbQueueWorker:
    def __init__(
        self,
        *,
        worker_id: Optional[str] = None,
        allowed_engines: Optional[set[str]] = None,
    ) -> None:
        self.worker_id = worker_id or os.getenv("WORKER_ID") or uuid.uuid4().hex
        self.poll_interval = float(os.getenv("WORKER_POLL_INTERVAL", "2"))
        self.heartbeat_interval = float(os.getenv("WORKER_HEARTBEAT_INTERVAL", "5"))
        self.stale_after = int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120"))
        self.allowed_engines = (
            allowed_engines
            if allowed_engines is not None
            else _parse_worker_engines(os.getenv("WORKER_ENGINES"))
        )
        self.visible_host_gpu_ids = parse_visible_host_gpu_ids()

        self._running: Optional[RunningJob] = None
        self._last_heartbeat_at: Optional[datetime] = None

    def run_forever(self) -> None:
        engines_text = ",".join(sorted(self.allowed_engines)) if self.allowed_engines else "*"
        print(f"[worker] starting worker_id={self.worker_id} engines={engines_text}", flush=True)
        settings.ensure_dirs()
        while True:
            try:
                self.tick()
            except Exception as e:
                print(f"[worker] tick error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            time.sleep(self.poll_interval)

    def tick(self) -> None:
        assert_valid_license()
        if self._running is not None:
            self._tick_running()
            return
        self._try_start_next_run()

    def _tick_running(self) -> None:
        assert self._running is not None
        run_id = self._running.run_id

        db = SessionLocal()
        should_cleanup = False
        try:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                _terminate_process_tree(self._running.proc)
                should_cleanup = True
                return

            now = _utcnow()
            if self._last_heartbeat_at is None or (now - self._last_heartbeat_at).total_seconds() >= self.heartbeat_interval:
                if touch_heartbeat(db, run_id, execution_owner=self.worker_id, heartbeat_at=now):
                    self._last_heartbeat_at = now

            cancel_requested = bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
            if cancel_requested and self._running.proc.poll() is None:
                _terminate_process_tree(self._running.proc)

            rc = self._running.proc.poll()
            if rc is None:
                return

            result = finalize_execution(
                db,
                run_id,
                exit_code=int(rc),
                error_message=f"Training subprocess exited with code {rc}" if rc != 0 else None,
            )
            should_cleanup = True
            if result.changed:
                AlarmService.try_evaluate_training_rules(db, run_ids=[str(result.run_id)])

            if result.status == TrainingRunStatus.DELETED:
                _safe_remove_dir(settings.training_dir / run_id)
        finally:
            db.close()
            if should_cleanup:
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
                .join(ModelArchitecture, TrainingRun.architecture_id == ModelArchitecture.architecture_id)
                .filter(TrainingRun.status == TrainingRunStatus.QUEUED)
                .filter(TrainingRun.queued_at.isnot(None))
                .filter(TrainingRun.claimed_at.is_(None))
                .filter(TrainingRun.hidden == False)  # noqa: E712
                .order_by(TrainingRun.queued_at.asc())
            )
            if self.allowed_engines:
                q = q.filter(ModelArchitecture.engine.in_(sorted(self.allowed_engines)))

            # Best-effort row locking for multi-worker.
            try:
                q = q.with_for_update(skip_locked=True)
            except Exception:
                pass

            run = None
            for candidate in q.limit(50).all():
                device_spec = getattr(getattr(candidate, "parameters", None), "device", "auto")
                if worker_can_run_device(device_spec, self.visible_host_gpu_ids):
                    run = candidate
                    break
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

            try:
                started = mark_started(
                    db,
                    run.run_id,
                    worker_id=self.worker_id,
                    pid=int(proc.pid),
                    started_at=now,
                )
            except Exception:
                _terminate_process_tree(proc)
                stdout_f.close()
                stderr_f.close()
                raise
            AlarmService.try_evaluate_training_rules(db, run_ids=[str(started.run_id)])

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
        changed_ids: list[str] = []
        for run in stale_queued:
            release_stale_claim(db, str(run.run_id))
            changed_ids.append(str(run.run_id))

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
            result = finalize_execution(
                db,
                str(run.run_id),
                exit_code=1,
                error_message="Worker heartbeat lost; marking as failed",
            )
            if result.changed:
                changed_ids.append(str(result.run_id))

        if changed_ids:
            AlarmService.try_evaluate_training_rules(db, run_ids=changed_ids)



def main() -> None:
    assert_valid_license()
    DbQueueWorker().run_forever()


if __name__ == "__main__":
    main()

