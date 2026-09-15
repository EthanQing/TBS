from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

import psutil
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.db.session import SessionLocal
from train_platform.domains.training.runs import finalize_execution, mark_started, release_stale_claim, touch_heartbeat
from train_platform.domains.monitoring.alarms.training import evaluate_training_alerts_best_effort
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun
from train_platform.domains.training.parameters import (
    extract_selected_gpu_ids,
    parse_visible_host_gpu_ids,
    worker_can_run_device,
)
from train_platform.platform.runtime.ultralytics_ddp import (
    UltralyticsDDPCleanupIncomplete,
    UltralyticsDDPError,
    terminate_registered_processes,
)
from train_platform.platform.runtime import process_scope


CUSTOM_CANCEL_FALLBACK_SECONDS = 10.0
ULTRALYTICS_DDP_CANCEL_FALLBACK_SECONDS = 15.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_remove_dir(path: Path) -> None:
    import shutil

    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _write_execution_record(run_root: Path, run_id: str, owner: Mapping[str, Any]) -> None:
    runtime_dir = Path(run_root) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / "execution.json"
    temporary = runtime_dir / "execution.json.tmp"
    temporary.write_text(
        json.dumps({"run_id": str(run_id), "execution_owner": dict(owner)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_execution_record(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / "runtime" / "execution.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid execution record: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("execution_owner"), dict):
        raise ValueError(f"invalid execution record: {path}")
    return value


def _identity_is_live(identity: Mapping[str, Any]) -> bool | None:
    if process_scope.compare_process_scope(
        process_scope.identity_process_scope(identity), process_scope.get_process_scope()
    ) != "same":
        return None
    try:
        process = psutil.Process(int(identity["pid"]))
        if float(process.create_time()) != float(identity["create_time"]):
            return False
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError, KeyError, TypeError, ValueError):
        return None


def _spawn_training_subprocess(run_id: str, *, stdout_f: TextIO, stderr_f: TextIO) -> subprocess.Popen:
    args = [sys.executable, "-m", "train_platform.workers.training.train_entry", "--run-id", run_id]
    env = os.environ.copy()
    # Redirected Python streams choose their own encoding, independent of the log file handles.
    env["PYTHONIOENCODING"] = "utf-8"

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f, env=env, creationflags=creationflags)

    return subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f, env=env, start_new_session=True)


def _terminate_process_tree(proc: subprocess.Popen, *, timeout_sec: int = 20) -> None:
    if proc.poll() is not None:
        return

    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        descendants = []

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            deadline = time.time() + float(timeout_sec)
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    if descendants:
        try:
            for child in descendants:
                try:
                    if child.is_running():
                        child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            _, alive = psutil.wait_procs(descendants, timeout=2.0)
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            psutil.wait_procs(alive, timeout=2.0)
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


@contextmanager
def worker_shutdown_signals():
    previous = {}

    def request_shutdown(signum, frame):
        raise SystemExit(0)

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                old_handler = signal.getsignal(sig)
                signal.signal(sig, request_shutdown)
                previous[sig] = old_handler
            except (ValueError, OSError):
                pass
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@dataclass
class RunningJob:
    run_id: str
    engine: str
    proc: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    stdout_f: TextIO
    stderr_f: TextIO
    cancel_seen_at: Optional[datetime] = None
    guard_create_time: float = 0.0
    ultralytics_ddp: bool = False
    execution_owner: dict[str, Any] | None = None


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
        self._gpu_resource_reporter = None

    def start_resource_reporter(self) -> None:
        try:
            if self._gpu_resource_reporter is None:
                from train_platform.workers.gpu_resource_reporter import GpuResourceReporter
                allowed_engines = self.allowed_engines
                if allowed_engines is None:
                    from train_platform.domains.training.frameworks import list_plugins

                    allowed_engines = {
                        plugin.plugin_id for plugin in list_plugins() if plugin.implemented
                    }
                self._gpu_resource_reporter = GpuResourceReporter(
                    worker_id=self.worker_id,
                    allowed_engines=allowed_engines,
                )
            self._gpu_resource_reporter.start()
        except Exception as exc:
            print(f"[worker] GPU resource reporter start failed: {exc}", file=sys.stderr, flush=True)

    def stop_resource_reporter(self) -> None:
        if self._gpu_resource_reporter is not None:
            self._gpu_resource_reporter.stop()

    def run_forever(self) -> None:
        engines_text = ",".join(sorted(self.allowed_engines)) if self.allowed_engines else "*"
        print(f"[worker] starting worker_id={self.worker_id} engines={engines_text}", flush=True)
        settings.ensure_dirs()
        self.start_resource_reporter()
        try:
            with worker_shutdown_signals():
                while True:
                    try:
                        self.tick()
                    except Exception as e:
                        print(f"[worker] tick error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    time.sleep(self.poll_interval)
        finally:
            self.stop_resource_reporter()

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
        registered_cleanup_done = False
        try:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                self._validate_running_ddp_scope()
                _terminate_process_tree(self._running.proc)
                self._cleanup_registered_ddp()
                should_cleanup = True
                return

            now = _utcnow()
            if self._last_heartbeat_at is None or (now - self._last_heartbeat_at).total_seconds() >= self.heartbeat_interval:
                if touch_heartbeat(
                    db,
                    run_id,
                    execution_owner=self.worker_id,
                    expected_pid=int(self._running.proc.pid),
                    heartbeat_at=now,
                ):
                    self._last_heartbeat_at = now

            cancel_requested = bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
            if cancel_requested and self._running.proc.poll() is None:
                if self._running.engine == "custom-source" or self._running.ultralytics_ddp:
                    grace_seconds = (
                        ULTRALYTICS_DDP_CANCEL_FALLBACK_SECONDS
                        if self._running.ultralytics_ddp
                        else CUSTOM_CANCEL_FALLBACK_SECONDS
                    )
                    if self._running.cancel_seen_at is None:
                        self._running.cancel_seen_at = now
                    elif (now - self._running.cancel_seen_at).total_seconds() >= grace_seconds:
                        self._validate_running_ddp_scope()
                        _terminate_process_tree(self._running.proc)
                        self._cleanup_registered_ddp()
                        registered_cleanup_done = True
                else:
                    _terminate_process_tree(self._running.proc)

            rc = self._running.proc.poll()
            if rc is None:
                return

            if not registered_cleanup_done:
                self._cleanup_registered_ddp()

            result = finalize_execution(
                db,
                run_id,
                exit_code=int(rc),
                expected_pid=int(self._running.proc.pid),
                error_message=f"Training subprocess exited with code {rc}" if rc != 0 else None,
            )
            should_cleanup = True
            if result.changed:
                evaluate_training_alerts_best_effort(db, run_ids=[str(result.run_id)])

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

    def _validate_running_ddp_scope(self) -> None:
        if self._running is None or not self._running.ultralytics_ddp:
            return
        owner = self._running.execution_owner
        if not isinstance(owner, dict):
            raise UltralyticsDDPCleanupIncomplete(
                "running DDP job has no execution owner",
                run_id=self._running.run_id, attempt_id="unknown",
                execution_owner={}, survivors=[],
            )
        scope_status = process_scope.compare_process_scope(
            owner.get("process_scope"), process_scope.get_process_scope()
        )
        if scope_status != "same":
            raise UltralyticsDDPCleanupIncomplete(
                f"running DDP execution process scope is {scope_status}",
                run_id=self._running.run_id, attempt_id="unknown",
                execution_owner=owner, survivors=[],
            )
        run_root = settings.training_dir / self._running.run_id
        record = _read_execution_record(run_root)
        if record is None:
            _write_execution_record(run_root, self._running.run_id, owner)

    def _cleanup_registered_ddp(self) -> None:
        if self._running is None or not self._running.ultralytics_ddp:
            return
        self._validate_running_ddp_scope()
        owner = self._running.execution_owner
        assert isinstance(owner, dict)
        survivors = terminate_registered_processes(
            settings.training_dir / self._running.run_id,
            run_id=self._running.run_id,
            owner=owner,
            grace_seconds=2.0,
        )
        if survivors:
            pids = ",".join(str(process.pid) for process in survivors)
            raise UltralyticsDDPError(f"Registered training processes are still alive: run_id={self._running.run_id} pids={pids}")

    def _cleanup_stale_ddp(self, run: TrainingRun) -> bool:
        run_id = str(run.run_id)
        run_root = settings.training_dir / run_id
        ddp_root = run_root / "runtime" / "ddp"
        try:
            execution = _read_execution_record(run_root)
        except ValueError as exc:
            print(f"[worker] stale DDP cleanup deferred run_id={run_id}: {exc}", file=sys.stderr, flush=True)
            return False

        owner: dict[str, Any] | None = None
        if execution is not None:
            candidate = execution["execution_owner"]
            if (
                execution.get("run_id") != run_id
                or candidate.get("guard_pid") != run.pid
                or candidate.get("worker_id") != str(run.worker_id or "")
            ):
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: execution record does not match claim", file=sys.stderr, flush=True)
                return False
            owner = dict(candidate)

        contexts: list[tuple[str, dict[str, Any]]] = []
        for attempt in ddp_root.glob("*") if ddp_root.is_dir() else []:
            context_path = attempt / "context.json"
            try:
                context = json.loads(context_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # An unreadable attempt cannot prove that its execution has exited.
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: unreadable context {context_path}", file=sys.stderr, flush=True)
                return False
            if not isinstance(context, dict):
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: invalid context {context_path}", file=sys.stderr, flush=True)
                return False
            if context.get("run_id") != run_id:
                continue
            context_owner = context.get("execution_owner")
            if not isinstance(context_owner, dict):
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: context owner missing", file=sys.stderr, flush=True)
                return False
            if context_owner.get("worker_id") != str(run.worker_id or "") or context_owner.get("guard_pid") != run.pid:
                continue
            if context.get("attempt_id") != attempt.name:
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: context attempt mismatch", file=sys.stderr, flush=True)
                return False
            contexts.append((attempt.name, context))

        if owner is None:
            owners = {
                json.dumps(context["execution_owner"], sort_keys=True): context["execution_owner"]
                for _, context in contexts
            }
            if len(owners) != 1:
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: execution scope unavailable", file=sys.stderr, flush=True)
                return False
            owner = dict(next(iter(owners.values())))

        scope_status = process_scope.compare_process_scope(
            owner.get("process_scope"), process_scope.get_process_scope()
        )
        if scope_status != "same":
            print(f"[worker] stale DDP cleanup deferred run_id={run_id}: process scope {scope_status}", file=sys.stderr, flush=True)
            return False

        guard_identity = {"pid": owner.get("guard_pid"), "create_time": owner.get("guard_create_time"), "process_scope": owner.get("process_scope")}
        guard_live = _identity_is_live(guard_identity)
        if guard_live is not False:
            print(f"[worker] stale DDP cleanup deferred run_id={run_id}: guard state {'live' if guard_live else 'unknown'}", file=sys.stderr, flush=True)
            return False
        matching = [
            (attempt_id, context)
            for attempt_id, context in contexts
            if all(
                context["execution_owner"].get(key) == owner.get(key)
                for key in ("guard_pid", "guard_create_time", "worker_id")
            )
        ]
        for _, context in matching:
            context_scope_status = process_scope.compare_process_scope(
                context["execution_owner"].get("process_scope"), process_scope.get_process_scope()
            )
            if context_scope_status != "same":
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: context process scope {context_scope_status}", file=sys.stderr, flush=True)
                return False
        if execution is None and not matching:
            print(f"[worker] stale DDP cleanup deferred run_id={run_id}: no matching scoped attempt", file=sys.stderr, flush=True)
            return False
        for attempt_id, context in matching:
            supervisor = context.get("supervisor")
            supervisor_live = _identity_is_live(supervisor) if isinstance(supervisor, Mapping) else None
            if supervisor_live is not False:
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: supervisor state {'live' if supervisor_live else 'unknown'}", file=sys.stderr, flush=True)
                return False
            try:
                survivors = terminate_registered_processes(
                    run_root,
                    run_id=run_id,
                    owner=owner,
                    attempt_id=attempt_id,
                    grace_seconds=2.0,
                )
            except (UltralyticsDDPCleanupIncomplete, OSError, ValueError) as exc:
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: {exc}", file=sys.stderr, flush=True)
                return False
            if survivors:
                print(f"[worker] stale DDP cleanup deferred run_id={run_id}: registered processes remain", file=sys.stderr, flush=True)
                return False
        return True

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
            q = q.filter(~TrainingRun.resource_request.has())

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

            engine = str(getattr(run.architecture, "engine", "") or "").strip().lower()
            ultralytics_ddp = (
                engine == "ultralytics-yolo"
                and len(extract_selected_gpu_ids(getattr(run.parameters, "device", "auto"))) > 1
            )
            current_scope = process_scope.get_process_scope()
            if ultralytics_ddp and current_scope is None:
                raise RuntimeError("Ultralytics DDP requires a verifiable process scope")

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
                guard_create_time = float(psutil.Process(proc.pid).create_time())
                execution_owner = {
                    "guard_pid": int(proc.pid),
                    "guard_create_time": guard_create_time,
                    "worker_id": self.worker_id,
                    "process_scope": current_scope,
                }
                _write_execution_record(run_dir, str(run.run_id), execution_owner)
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
            evaluate_training_alerts_best_effort(db, run_ids=[str(started.run_id)])

            self._running = RunningJob(
                run_id=run.run_id,
                engine=engine,
                proc=proc,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                stdout_f=stdout_f,
                stderr_f=stderr_f,
                guard_create_time=guard_create_time,
                ultralytics_ddp=ultralytics_ddp,
                execution_owner=execution_owner,
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
            engine = str(getattr(getattr(run, "architecture", None), "engine", "") or "").strip().lower()
            device = getattr(getattr(run, "parameters", None), "device", "auto")
            if engine == "ultralytics-yolo" and len(extract_selected_gpu_ids(device)) > 1:
                if not self._cleanup_stale_ddp(run):
                    continue
            result = finalize_execution(
                db,
                str(run.run_id),
                exit_code=1,
                expected_pid=int(run.pid) if run.pid is not None else None,
                error_message="Worker heartbeat lost; marking as failed",
            )
            if result.changed:
                changed_ids.append(str(result.run_id))

        if changed_ids:
            evaluate_training_alerts_best_effort(db, run_ids=changed_ids)



def main() -> None:
    assert_valid_license()
    DbQueueWorker().run_forever()


if __name__ == "__main__":
    main()

