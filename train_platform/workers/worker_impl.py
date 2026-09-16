from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

import psutil
from sqlalchemy.orm import Session
from sqlalchemy import text

from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.db.session import SessionLocal
from train_platform.domains.training.runs import finalize_execution, mark_started, release_stale_claim, touch_heartbeat
from train_platform.domains.monitoring.alarms.training import evaluate_training_alerts_best_effort
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import LogLevel, TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEvent
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
        json.dumps({"run_id": str(run_id), "allocation_id": owner.get("allocation_id"),
                    "execution_owner": dict(owner)}, ensure_ascii=False, indent=2) + "\n",
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


def _spawn_training_subprocess(run_id: str, *, stdout_f: TextIO, stderr_f: TextIO,
                               allocation_id: str | None = None, worker_instance_id: str | None = None,
                               assigned_gpu_uuids: list[str] | None = None) -> subprocess.Popen:
    args = [sys.executable, "-m", "train_platform.workers.training.train_entry", "--run-id", run_id]
    if allocation_id:
        args += ["--allocation-id", allocation_id, "--worker-instance-id", str(worker_instance_id)]
    env = os.environ.copy()
    # Redirected Python streams choose their own encoding, independent of the log file handles.
    env["PYTHONIOENCODING"] = "utf-8"
    if allocation_id:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned_gpu_uuids or [])
        env["TRAIN_PLATFORM_ALLOCATION_ID"] = allocation_id

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
    allocation_id: str | None = None
    worker_instance_id: str | None = None
    gpu_count: int = 0
    last_heartbeat_at: Optional[datetime] = None
    cleanup_state: str = "running"
    cleanup_future: Future | None = None
    cancel_future: Future | None = None
    cleanup_owner: dict[str, Any] | None = None
    process_watch_stop: Any = None
    process_watch_thread: Any = None


class RecoveredProcess:
    def __init__(self, pid: int, create_time: float, exit_code: int | None):
        self.pid = int(pid)
        self.create_time = float(create_time)
        self.exit_code = exit_code

    def poll(self):
        state = _identity_is_live({"pid": self.pid, "create_time": self.create_time,
                                   "process_scope": process_scope.get_process_scope()})
        return None if state is True else self.exit_code if state is False else None

    def terminate(self):
        psutil.Process(self.pid).terminate()


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

        self._running_jobs: dict[str, RunningJob] = {}
        self._cleanup_executor = ThreadPoolExecutor(max_workers=max(2, getattr(settings, "worker_max_concurrent_trainings", 2)))
        self._gpu_resource_reporter = None
        # The cursor survives polling ticks so a temporarily blocked queue head
        # cannot monopolize every bounded scheduling pass.
        from train_platform.domains.training.resources.allocator import AllocationScanCursor
        self._allocation_scan_cursor = AllocationScanCursor()

    def _record_cleanup_pending(self, allocation: Mapping[str, Any], reason: str, error: str) -> None:
        from train_platform.models.v3.gpu_allocation import GpuAllocation
        db = SessionLocal()
        try:
            run = db.query(TrainingRun).filter_by(run_id=allocation["run_id"]).with_for_update().first()
            locked = db.query(GpuAllocation).filter_by(
                allocation_id=allocation["allocation_id"],
            ).with_for_update().first()
            owner = allocation.get("owner")
            if (locked is None or run is None or locked.execution_owner != owner
                    or run.current_allocation_id != allocation["allocation_id"]):
                db.rollback()
                return
            details = {"cleanup_status": reason, "error": error}
            if run.resource_wait_reason != "cleanup_pending" or (run.resource_wait_details or {}) != details:
                run.resource_wait_reason = "cleanup_pending"
                run.resource_wait_details = details
                db.add(TrainingRunEvent(run_id=run.run_id, level=LogLevel.INFO,
                                        event_type="cleanup_pending", message=reason, data=details))
            db.commit()
        finally:
            db.close()

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
            self._cleanup_executor.shutdown(wait=False, cancel_futures=False)

    def tick(self) -> None:
        assert_valid_license()
        try:
            self._reconcile_managed_allocations()
        except Exception as exc:
            print(f"[worker] allocation reconciliation deferred: {exc}", file=sys.stderr, flush=True)
        for key, job in list(self._running_jobs.items()):
            try:
                self._tick_running(job)
            except Exception as exc:
                print(f"[worker] job maintenance error allocation={key}: {exc}", file=sys.stderr, flush=True)
        self._publish_running_task_count()
        while self.available_training_slots() > 0:
            if not self._try_start_next_run():
                break

    def has_running_jobs(self) -> bool:
        return bool(self._running_jobs)

    def running_job_count(self) -> int:
        return len(self._running_jobs)

    def available_training_slots(self) -> int:
        limit = getattr(settings, "worker_max_concurrent_trainings", 2) if self._managed_scheduling_required() else 1
        return max(0, limit - len(self._running_jobs))

    def _publish_running_task_count(self) -> None:
        instance_id = getattr(self._gpu_resource_reporter, "instance_id", None)
        if not instance_id:
            return
        from train_platform.models.v3.gpu_resource import GpuWorkerInstance
        db = SessionLocal()
        try:
            worker = db.get(GpuWorkerInstance, instance_id)
            if worker is not None and hasattr(worker, "running_task_count"):
                worker.running_task_count = len(self._running_jobs)
                db.commit()
        finally:
            db.close()

    def _reconcile_managed_allocations(self) -> None:
        from train_platform.models.v3.gpu_allocation import GpuAllocation
        from train_platform.domains.training.resources.lifecycle import revoke_unactivated_allocation
        node_id = getattr(self._gpu_resource_reporter, "node_id", None)
        if not node_id:
            return
        own_instance_id = getattr(self._gpu_resource_reporter, "instance_id", None)
        db = SessionLocal()
        try:
            allocations = db.query(GpuAllocation).filter(
                GpuAllocation.node_id == node_id,
                GpuAllocation.state.in_(("reserved", "starting", "running", "releasing")),
            ).all()
            snapshots = [{
                "allocation_id": item.allocation_id, "run_id": item.run_id,
                "worker_instance_id": item.worker_instance_id, "authorization_state": item.authorization_state,
                "deadline": item.launch_deadline_at, "owner": dict(item.execution_owner) if item.execution_owner else None,
                "launcher": dict(item.launcher_identity) if item.launcher_identity else None,
                "exit_code": item.exit_code, "gpu_count": len(item.devices),
                "assigned_gpu_uuids": [device.gpu_uuid for device in sorted(item.devices, key=lambda row: row.ordinal)],
                "cleanup_owner": (item.request_snapshot or {}).get("legacy_execution_owner"),
            } for item in allocations]
        finally:
            db.close()
        for allocation in snapshots:
            if allocation["allocation_id"] in self._running_jobs:
                continue
            launcher = allocation["launcher"]
            launcher_live = _identity_is_live(launcher) if launcher else None
            own = allocation["worker_instance_id"] == own_instance_id
            if not own and launcher_live is not False:
                continue
            owner = allocation["owner"]
            if not owner:
                deadline = allocation["deadline"] if allocation["deadline"].tzinfo else allocation["deadline"].replace(tzinfo=timezone.utc)
                if allocation["authorization_state"] == "issued" and _utcnow() > deadline and (own or launcher_live is False):
                    write_db = SessionLocal()
                    try:
                        revoke_unactivated_allocation(write_db, allocation["allocation_id"], run_id=allocation["run_id"],
                                                      worker_instance_id=allocation["worker_instance_id"],
                                                      reason="unactivated allocation launch deadline expired")
                        write_db.commit()
                    finally:
                        write_db.close()
                continue
            if process_scope.compare_process_scope(owner.get("process_scope"), process_scope.get_process_scope()) != "same":
                self._record_cleanup_pending(allocation, "process_scope_unconfirmed",
                                             "execution process scope cannot be confirmed on this worker")
                continue
            live = _identity_is_live({"pid": owner.get("guard_pid"), "create_time": owner.get("guard_create_time"),
                                      "process_scope": owner.get("process_scope")})
            if live is None:
                self._record_cleanup_pending(allocation, "process_scope_unconfirmed",
                                             "supervisor process identity cannot be confirmed")
                continue
            run_dir = settings.training_dir / allocation["run_id"]
            logs_dir = run_dir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = logs_dir / "train.stdout.log"; stderr_path = logs_dir / "train.stderr.log"
            exit_code = allocation["exit_code"] if allocation["exit_code"] is not None else 1
            job = RunningJob(
                    run_id=allocation["run_id"], allocation_id=allocation["allocation_id"],
                    worker_instance_id=allocation["worker_instance_id"], engine="unknown",
                    proc=RecoveredProcess(owner["guard_pid"], owner["guard_create_time"], exit_code),
                    stdout_path=stdout_path, stderr_path=stderr_path,
                    stdout_f=open(stdout_path, "a", encoding="utf-8", buffering=1),
                    stderr_f=open(stderr_path, "a", encoding="utf-8", buffering=1),
                    guard_create_time=float(owner["guard_create_time"]), execution_owner=owner,
                    cleanup_owner=allocation["cleanup_owner"],
                    gpu_count=allocation["gpu_count"], ultralytics_ddp=allocation["gpu_count"] > 1,
                )
            if live:
                from train_platform.platform.runtime.execution_processes import register_execution_process, start_descendant_registration
                try:
                    register_execution_process(
                        settings.training_dir / allocation["run_id"], run_id=allocation["run_id"],
                        allocation_id=allocation["allocation_id"], execution_owner=owner,
                        pid=int(owner["guard_pid"]), role="supervisor",
                        assigned_gpu_uuids=allocation["assigned_gpu_uuids"], expected_identity={
                            "pid": owner["guard_pid"], "create_time": owner["guard_create_time"],
                            "process_scope": owner["process_scope"],
                        },
                    )
                except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
                    self._record_cleanup_pending(allocation, "registration_temporarily_failed", str(exc))
                job.process_watch_stop, job.process_watch_thread = start_descendant_registration(
                    settings.training_dir / allocation["run_id"], run_id=allocation["run_id"],
                    allocation_id=allocation["allocation_id"], execution_owner=owner,
                    supervisor_pid=int(owner["guard_pid"]), assigned_gpu_uuids=allocation["assigned_gpu_uuids"],
                )
            self._running_jobs[allocation["allocation_id"]] = job

    def _tick_running(self, job: RunningJob) -> None:
        run_id = job.run_id

        db = SessionLocal()
        should_cleanup = False
        registered_cleanup_done = False
        try:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                if job.allocation_id:
                    raise RuntimeError("managed execution lost its task record; allocation retained for reconciliation")
                _terminate_process_tree(job.proc)
                should_cleanup = True
                return

            now = _utcnow()
            if job.allocation_id and job.execution_owner is None:
                from train_platform.models.v3.gpu_allocation import GpuAllocation
                allocation = db.get(GpuAllocation, job.allocation_id)
                if allocation and allocation.execution_owner:
                    job.execution_owner = dict(allocation.execution_owner)
                    job.guard_create_time = float(job.execution_owner["guard_create_time"])
            if job.last_heartbeat_at is None or (now - job.last_heartbeat_at).total_seconds() >= self.heartbeat_interval:
                if touch_heartbeat(
                    db,
                    run_id,
                    execution_owner=(job.execution_owner or {}).get("worker_id", self.worker_id) if job.allocation_id else self.worker_id,
                    expected_pid=int(job.proc.pid),
                    heartbeat_at=now,
                    allocation_id=job.allocation_id,
                    expected_create_time=job.guard_create_time if job.allocation_id else None,
                ):
                    job.last_heartbeat_at = now
                    if job.allocation_id:
                        from train_platform.models.v3.gpu_allocation import GpuAllocation
                        allocation = db.get(GpuAllocation, job.allocation_id)
                        if allocation and allocation.state != "released":
                            allocation.heartbeat_at = now
                            db.commit()

            cancel_requested = bool(run.cancel_requested_at is not None or run.delete_requested_at is not None)
            if cancel_requested and job.proc.poll() is None:
                if job.cancel_future is not None:
                    if job.cancel_future.done():
                        try:
                            job.cancel_future.result()
                        finally:
                            job.cancel_future = None
                    return
                if job.engine == "custom-source" or job.ultralytics_ddp:
                    grace_seconds = (
                        ULTRALYTICS_DDP_CANCEL_FALLBACK_SECONDS
                        if job.ultralytics_ddp
                        else CUSTOM_CANCEL_FALLBACK_SECONDS
                    )
                    if job.cancel_seen_at is None:
                        job.cancel_seen_at = now
                    elif (now - job.cancel_seen_at).total_seconds() >= grace_seconds:
                        db.close()
                        job.cancel_future = self._cleanup_executor.submit(self._terminate_running_job, job)
                        return
                else:
                    db.close()
                    job.cancel_future = self._cleanup_executor.submit(self._terminate_running_job, job)
                    return

            rc = job.proc.poll()
            if rc is None:
                return

            if job.allocation_id:
                if job.cleanup_future is None:
                    job.cleanup_future = self._cleanup_executor.submit(self._finish_managed_job, job, int(rc))
                    return
                if not job.cleanup_future.done():
                    return
                try:
                    cleanup_complete, managed_status, managed_changed = job.cleanup_future.result()
                except Exception:
                    job.cleanup_future = None
                    raise
                if not cleanup_complete:
                    job.cleanup_future = None
                    return
                from types import SimpleNamespace
                result = SimpleNamespace(changed=managed_changed, run_id=run_id, status=managed_status)
            else:
                if not registered_cleanup_done:
                    self._cleanup_registered_ddp(job)
                result = finalize_execution(db, run_id, exit_code=int(rc), expected_pid=int(job.proc.pid),
                    error_message=f"Training subprocess exited with code {rc}" if rc != 0 else None)
            should_cleanup = True
            if result.changed:
                evaluate_training_alerts_best_effort(db, run_ids=[str(result.run_id)])

            if result.status == TrainingRunStatus.DELETED:
                _safe_remove_dir(settings.training_dir / run_id)
        finally:
            db.close()
            if should_cleanup:
                self._cleanup_running(job)
    def _cleanup_running(self, job: RunningJob) -> None:
        if job.process_watch_stop is not None:
            job.process_watch_stop.set()
        if job.process_watch_thread is not None:
            job.process_watch_thread.join(timeout=2.0)
        try:
            job.stdout_f.close()
        except Exception:
            pass
        try:
            job.stderr_f.close()
        except Exception:
            pass
        self._running_jobs.pop(job.allocation_id or job.run_id, None)

    def _terminate_running_job(self, job: RunningJob) -> None:
        owner = job.execution_owner
        if owner and process_scope.compare_process_scope(owner.get("process_scope"), process_scope.get_process_scope()) != "same":
            raise RuntimeError("refusing to terminate execution in a different process scope")
        if owner:
            live = _identity_is_live({"pid": owner.get("guard_pid"),
                                      "create_time": owner.get("guard_create_time"),
                                      "process_scope": owner.get("process_scope")})
            if live is False:
                return
            if live is None:
                raise RuntimeError("execution process identity cannot be verified")
        _terminate_process_tree(job.proc)

    def _finish_managed_job(self, job: RunningJob, rc: int) -> tuple[bool, TrainingRunStatus | None, bool]:
        from train_platform.domains.training.resources.lifecycle import record_execution_result, request_releasing, finish_allocation
        from train_platform.models.v3.gpu_allocation import GpuAllocation
        from train_platform.platform.runtime.execution_processes import cleanup_registered_execution
        if job.process_watch_stop is not None:
            job.process_watch_stop.set()
        if job.process_watch_thread is not None:
            job.process_watch_thread.join(timeout=2.0)
            if job.process_watch_thread.is_alive():
                if job.execution_owner:
                    self._record_cleanup_pending({
                        "allocation_id": job.allocation_id, "run_id": job.run_id,
                        "owner": job.execution_owner,
                    }, "registration_temporarily_failed", "descendant registration scan is still running")
                return False, None, False
        db = SessionLocal()
        assigned_gpu_uuids: list[str] = []
        try:
            allocation = db.get(GpuAllocation, job.allocation_id)
            if allocation and allocation.state == "released":
                run = db.get(TrainingRun, job.run_id)
                return True, run.status if run else None, False
            owner = allocation.execution_owner if allocation else None
            if allocation is not None:
                assigned_gpu_uuids = [device.gpu_uuid for device in sorted(allocation.devices, key=lambda row: row.ordinal)]
            if not owner:
                from train_platform.domains.training.resources.lifecycle import revoke_unactivated_allocation
                released = revoke_unactivated_allocation(
                    db, job.allocation_id, run_id=job.run_id,
                    worker_instance_id=job.worker_instance_id,
                    reason="training subprocess exited before activation",
                )
                db.commit()
                return released, None, False
            job.execution_owner = dict(owner)
            if allocation.exit_code is None:
                record_execution_result(db, job.allocation_id, owner, rc, f"Training subprocess exited with code {rc}" if rc else None)
            request_releasing(db, job.allocation_id, owner)
            db.commit()
        finally:
            db.close()
        ddp_cleanup_error: Exception | None = None
        if job.ultralytics_ddp:
            try:
                self._cleanup_registered_ddp(job)
            except (UltralyticsDDPCleanupIncomplete, RuntimeError, OSError, psutil.Error) as exc:
                ddp_cleanup_error = exc
        try:
            proof = cleanup_registered_execution(
                settings.training_dir / job.run_id, run_id=job.run_id,
                allocation_id=job.allocation_id, execution_owner=owner,
                assigned_gpu_uuids=assigned_gpu_uuids,
            )
        except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
            proof = {"complete": False, "survivors": [], "unknown": [],
                     "error": str(exc), "registration_errors": [
                         {"stage": "registration_temporarily_failed", "error": str(exc)}
                     ]}
        if ddp_cleanup_error is not None:
            proof = dict(proof)
            proof["complete"] = False
            proof["error"] = str(ddp_cleanup_error)
            proof["registration_errors"] = [
                *(proof.get("registration_errors") or []),
                {"stage": "ddp_cleanup", "error": str(ddp_cleanup_error)},
            ]
        if not proof["complete"]:
            pending_reason = "registration_temporarily_failed"
            if proof.get("survivors"):
                pending_reason = "processes_still_alive"
            elif proof.get("unknown") or proof.get("unconfirmed_sessions"):
                pending_reason = "process_scope_unconfirmed"
            elif any(item.get("stage") == "registration_io_error"
                     for item in proof.get("registration_errors", [])):
                pending_reason = "registration_temporarily_failed"
            elif any(item.get("stage") == "base_registration"
                     for item in proof.get("registration_errors", [])):
                pending_reason = "base_registration_missing"
            elif proof.get("recovered_registration"):
                pending_reason = "recovered_registration_waiting_cleanup"
            unresolved_by_key = {}
            for item in proof.get("registration_errors", []):
                value = {
                    "stage": item.get("stage"), "error": item.get("error"),
                    "target": {key: (item.get("target") or {}).get(key)
                               for key in ("run_id", "allocation_id", "execution_owner", "pid", "create_time",
                                           "process_scope", "sid", "pgid", "assigned_gpu_uuids")
                               if (item.get("target") or {}).get(key) is not None},
                }
                unresolved_by_key[json.dumps(value, sort_keys=True, default=str)] = value
            details = {
                "cleanup_status": pending_reason,
                "survivors": proof.get("survivors", []),
                "unknown": proof.get("unknown", []),
                "unconfirmed_sessions": proof.get("unconfirmed_sessions", []),
                "error": proof.get("error"),
                "recovered_registration": bool(proof.get("recovered_registration")),
                "unresolved_errors": [unresolved_by_key[key] for key in sorted(unresolved_by_key)],
            }
            status_db = SessionLocal()
            try:
                run = status_db.query(TrainingRun).filter_by(run_id=job.run_id).with_for_update().first()
                current = status_db.get(GpuAllocation, job.allocation_id)
                if (run is not None and current is not None
                        and run.current_allocation_id == job.allocation_id
                        and current.execution_owner == owner and current.state == "releasing"):
                    previous = run.resource_wait_details or {}
                    if (run.resource_wait_reason != "cleanup_pending" or previous != details):
                        run.resource_wait_reason = "cleanup_pending"
                        run.resource_wait_details = details
                        status_db.add(TrainingRunEvent(
                            run_id=job.run_id, level=LogLevel.INFO,
                            event_type="cleanup_pending", message=pending_reason, data=details,
                        ))
                    status_db.commit()
            finally:
                status_db.close()
            return False, None, False
        db = SessionLocal()
        try:
            finish_allocation(db, job.allocation_id, owner, proof)
            db.commit()
            run = db.get(TrainingRun, job.run_id)
            status = run.status if run else None
        finally:
            db.close()
        if status == TrainingRunStatus.COMPLETED:
            from train_platform.domains.training.runs import index_completion_artifacts
            artifact_db = SessionLocal()
            try:
                index_completion_artifacts(artifact_db, job.run_id)
                artifact_db.commit()
            except Exception as exc:
                artifact_db.rollback()
                print(f"[worker] artifact indexing failed run_id={job.run_id}: {exc}", file=sys.stderr, flush=True)
            finally:
                artifact_db.close()
        return True, status, True

    def _validate_running_ddp_scope(self, job: RunningJob) -> None:
        owner = job.execution_owner
        if not isinstance(owner, dict):
            raise UltralyticsDDPCleanupIncomplete("running DDP job has no execution owner", run_id=job.run_id,
                                                  attempt_id="unknown", execution_owner={}, survivors=[])
        status = process_scope.compare_process_scope(owner.get("process_scope"), process_scope.get_process_scope())
        if status != "same":
            raise UltralyticsDDPCleanupIncomplete(f"running DDP execution process scope is {status}", run_id=job.run_id,
                                                  attempt_id="unknown", execution_owner=owner, survivors=[])

    def _cleanup_registered_ddp(self, job: RunningJob) -> None:
        if not job.ultralytics_ddp:
            return
        self._validate_running_ddp_scope(job)
        survivors = terminate_registered_processes(settings.training_dir / job.run_id, run_id=job.run_id,
                                                     owner=job.cleanup_owner or job.execution_owner, grace_seconds=2.0)
        if survivors:
            raise UltralyticsDDPError(f"Registered training processes are still alive: pids={','.join(str(x.pid) for x in survivors)}")

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

    def _try_start_next_run(self) -> bool:
        managed_mode = self._managed_scheduling_required()
        if managed_mode:
            if not getattr(settings, "gpu_scheduler_enabled", False):
                return False
            if self._gpu_resource_reporter is None:
                return False
            if self._try_start_managed():
                return True
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
            if managed_mode:
                q = q.filter(TrainingRun.parameters.has(device="cpu"))

            # Best-effort row locking for multi-worker.
            try:
                q = q.with_for_update(skip_locked=True)
            except Exception:
                pass

            run = None
            for candidate in q.limit(50).all():
                device_spec = getattr(getattr(candidate, "parameters", None), "device", "auto")
                if managed_mode and str(device_spec or "auto").strip().lower() != "cpu":
                    continue
                if worker_can_run_device(device_spec, self.visible_host_gpu_ids):
                    run = candidate
                    break
            if not run:
                return False

            if not managed_mode:
                from train_platform.models.v3.gpu_allocation import GpuNodeSchedulingState
                node_id = getattr(self._gpu_resource_reporter, "node_id", None) or getattr(settings, "gpu_node_id", None)
                if node_id:
                    node = db.query(GpuNodeSchedulingState).filter_by(node_id=node_id).with_for_update().first()
                    if node is not None and node.managed:
                        return False

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

            job = RunningJob(
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
            job.last_heartbeat_at = now
            self._running_jobs[job.run_id] = job
            return True

        finally:
            db.close()

    def _managed_scheduling_required(self) -> bool:
        if getattr(settings, "gpu_scheduler_enabled", False):
            return True
        node_id = getattr(self._gpu_resource_reporter, "node_id", None) or getattr(settings, "gpu_node_id", None)
        if not node_id:
            return False
        from train_platform.models.v3.gpu_allocation import GpuNodeSchedulingState
        db = SessionLocal()
        try:
            state = db.get(GpuNodeSchedulingState, node_id)
            return bool(state and state.managed)
        finally:
            db.close()

    def _try_start_managed(self) -> bool:
        from train_platform.domains.training.resources.allocator import reserve_next
        from train_platform.domains.training.resources.lifecycle import issue_start_authorization, revoke_unactivated_allocation
        db = SessionLocal()
        allocation = None
        proc = None
        stdout_f = stderr_f = None
        try:
            instance_id = self._gpu_resource_reporter.instance_id
            self._adopt_proven_legacy_executions(instance_id)
            decision = None
            for _ in range(50):
                if db.get_bind().dialect.name == "mysql":
                    db.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
                decision = reserve_next(
                    db, instance_id, scan_cursor=self._allocation_scan_cursor,
                    scheduler_enabled=settings.gpu_scheduler_enabled,
                    shared_execution_enabled=settings.gpu_shared_execution_enabled,
                    node_defaults={"max_shared_tasks_per_device": settings.gpu_max_shared_tasks_per_device,
                                   "memory_safety_mib": settings.gpu_memory_safety_mib},
                    stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
                    start_timeout_seconds=settings.gpu_allocation_start_timeout_seconds,
                )
                # Release locks and preserve the wait reason after every attempted
                # candidate. The next query resumes strictly after this row.
                db.commit()
                if decision.allocation is not None:
                    break
                if decision.reason_code == "scan_exhausted":
                    return False
            if decision is None or decision.allocation is None:
                return False
            allocation = decision.allocation
            allocation_id = allocation.allocation_id
            run_id = allocation.run_id
            assigned = [item.gpu_uuid for item in sorted(allocation.devices, key=lambda item: item.ordinal)]
            sharing = allocation.request_snapshot["sharing"]
            engine = str(db.get(TrainingRun, run_id).architecture.engine).lower()
            db.commit()
            deadline = _utcnow() + timedelta(seconds=settings.gpu_allocation_start_timeout_seconds)
            issue_start_authorization(db, allocation_id, allocation.launcher_identity or {}, deadline)
            db.commit()
            run_dir = settings.training_dir / run_id
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = logs_dir / "train.stdout.log"
            stderr_path = logs_dir / "train.stderr.log"
            stdout_f = open(stdout_path, "a", encoding="utf-8", buffering=1)
            stderr_f = open(stderr_path, "a", encoding="utf-8", buffering=1)
            proc = _spawn_training_subprocess(
                run_id, stdout_f=stdout_f, stderr_f=stderr_f,
                allocation_id=allocation_id, worker_instance_id=instance_id,
                assigned_gpu_uuids=assigned,
            )
            create_time = float(psutil.Process(proc.pid).create_time())
            self._running_jobs[allocation_id] = RunningJob(
                run_id=run_id, allocation_id=allocation_id, worker_instance_id=instance_id,
                engine=engine, proc=proc, stdout_path=stdout_path, stderr_path=stderr_path,
                stdout_f=stdout_f, stderr_f=stderr_f, guard_create_time=create_time,
                gpu_count=len(assigned), ultralytics_ddp=engine == "ultralytics-yolo" and len(assigned) > 1,
                last_heartbeat_at=_utcnow(),
            )
            return True
        except Exception as exc:
            db.rollback()
            if proc is not None:
                # Activation may already have committed. Keep supervising the
                # child; only the ordinary cleanup flow may release its budget.
                self._running_jobs[allocation_id] = RunningJob(
                    run_id=run_id, allocation_id=allocation_id, worker_instance_id=instance_id,
                    engine=engine, proc=proc, stdout_path=stdout_path, stderr_path=stderr_path,
                    stdout_f=stdout_f, stderr_f=stderr_f, gpu_count=len(assigned),
                    ultralytics_ddp=engine == "ultralytics-yolo" and len(assigned) > 1,
                )
                print(f"[worker] launch identity pending reconciliation: {exc}", file=sys.stderr, flush=True)
                return True
            if allocation is not None:
                try:
                    revoke_unactivated_allocation(db, allocation.allocation_id, run_id=allocation.run_id,
                                                  worker_instance_id=allocation.worker_instance_id,
                                                  reason=f"launch failed: {exc}")
                    db.commit()
                except Exception:
                    db.rollback()
            for handle in (stdout_f, stderr_f):
                if handle:
                    handle.close()
            print(f"[worker] managed launch failed: {exc}", file=sys.stderr, flush=True)
            return False
        finally:
            db.close()

    def _adopt_proven_legacy_executions(self, worker_instance_id: str) -> None:
        from train_platform.domains.training.resources.allocator import adopt_legacy_execution
        from train_platform.platform.runtime.cuda_devices import cuda_environment_fingerprint
        read_db = SessionLocal()
        try:
            candidates = [(str(run.run_id), str(run.worker_id or "")) for run in read_db.query(TrainingRun).filter(
                TrainingRun.status == TrainingRunStatus.RUNNING,
                TrainingRun.current_allocation_id.is_(None),
            ).all()]
        finally:
            read_db.close()
        for run_id, _worker_id in candidates:
            run_dir = settings.training_dir / run_id
            records: list[dict[str, Any]] = []
            try:
                execution = _read_execution_record(run_dir)
                if execution:
                    records.append(execution)
                for path in (run_dir / "runtime" / "ddp").glob("*/context.json"):
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        records.append(value)
            except (OSError, ValueError):
                continue
            for record in records:
                owner = record.get("execution_owner")
                assigned = record.get("assigned_gpu_uuids")
                if not isinstance(owner, dict) or not isinstance(assigned, list) or not assigned:
                    continue
                if _identity_is_live({"pid": owner.get("guard_pid"), "create_time": owner.get("guard_create_time"),
                                      "process_scope": owner.get("process_scope")}) is not True:
                    continue
                write_db = SessionLocal()
                try:
                    decision = adopt_legacy_execution(
                        write_db, run_id=run_id, worker_instance_id=worker_instance_id,
                        execution_owner=owner, assigned_gpu_uuids=[str(item) for item in assigned],
                        cuda_environment_fingerprint=cuda_environment_fingerprint(),
                        stale_after_seconds=settings.gpu_inventory_stale_after_seconds,
                    )
                    registration = None
                    if decision.allocation:
                        registration = (
                            decision.allocation.allocation_id,
                            dict(decision.allocation.execution_owner or {}),
                            [str(item) for item in assigned],
                        )
                    write_db.commit()
                    if registration:
                        from train_platform.platform.runtime.execution_processes import register_execution_process, start_descendant_registration
                        allocation_id, managed_owner, assigned_uuids = registration
                        register_execution_process(
                            settings.training_dir / run_id, run_id=run_id, allocation_id=allocation_id,
                            execution_owner=managed_owner, pid=int(managed_owner["guard_pid"]), role="supervisor",
                            assigned_gpu_uuids=assigned_uuids,
                        )
                        watch_stop, watch_thread = start_descendant_registration(
                            settings.training_dir / run_id, run_id=run_id, allocation_id=allocation_id,
                            execution_owner=managed_owner, supervisor_pid=int(managed_owner["guard_pid"]),
                            assigned_gpu_uuids=assigned_uuids,
                        )
                        existing = self._running_jobs.pop(run_id, None)
                        if existing is not None:
                            existing.allocation_id = allocation_id
                            existing.worker_instance_id = worker_instance_id
                            existing.execution_owner = managed_owner
                            existing.cleanup_owner = owner
                            existing.process_watch_stop = watch_stop
                            existing.process_watch_thread = watch_thread
                            self._running_jobs[allocation_id] = existing
                        break
                except Exception:
                    write_db.rollback()
                finally:
                    write_db.close()

    def _reconcile_stale_claims(self, db: Session) -> None:
        now = _utcnow()
        threshold = now - timedelta(seconds=self.stale_after)

        stale_queued = (
            db.query(TrainingRun)
            .filter(TrainingRun.status == TrainingRunStatus.QUEUED)
            .filter(TrainingRun.queued_at.isnot(None))
            .filter(TrainingRun.worker_id.isnot(None))
            .filter(TrainingRun.current_allocation_id.is_(None))
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
            .filter(TrainingRun.current_allocation_id.is_(None))
            .filter(
                (TrainingRun.heartbeat_at.is_(None) & (TrainingRun.started_at < threshold))
                | (TrainingRun.heartbeat_at < threshold)
            )
            .all()
        )
        for run in stale_running:
            if self._managed_scheduling_required():
                continue
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

