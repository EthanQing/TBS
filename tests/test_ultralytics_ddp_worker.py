from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_platform.workers import worker_impl as worker


LOCAL_SCOPE = {"boot_id": "test-boot", "pid_namespace": {"device": 1, "inode": 2}}


@pytest.fixture(autouse=True)
def local_process_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))


def setup_worker(monkeypatch, *, exited=False):
    now = datetime.now(timezone.utc)
    process = SimpleNamespace(pid=101, returncode=0 if exited else None)
    process.poll = lambda: process.returncode
    job = worker.RunningJob(
        run_id="run", engine="ultralytics-yolo", proc=process,
        stdout_path=Path("stdout"), stderr_path=Path("stderr"),
        stdout_f=StringIO(), stderr_f=StringIO(),
        guard_create_time=123.5, ultralytics_ddp=True,
        execution_owner={"guard_pid": 101, "guard_create_time": 123.5, "worker_id": "worker", "process_scope": LOCAL_SCOPE},
    )
    instance = worker.DbQueueWorker(worker_id="worker")
    instance._running = job
    instance._last_heartbeat_at = now
    run = SimpleNamespace(cancel_requested_at=now, delete_requested_at=None)
    db = SimpleNamespace(close=lambda: None)
    db.query = lambda *args: db
    db.filter = lambda *args: db
    db.first = lambda: run
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(worker, "_utcnow", lambda: now)
    monkeypatch.setattr(worker, "touch_heartbeat", lambda *args, **kwargs: True)
    events = []

    def terminate(proc):
        events.append("terminate")
        proc.returncode = -15

    def cleanup(*args, **kwargs):
        assert kwargs["owner"] == {"guard_pid": 101, "guard_create_time": 123.5, "worker_id": "worker", "process_scope": LOCAL_SCOPE}
        events.append("cleanup")
        return []

    def finalize(*args, **kwargs):
        events.append("finalize")
        assert kwargs["expected_pid"] == 101
        return SimpleNamespace(changed=False, status=worker.TrainingRunStatus.CANCELLED)

    monkeypatch.setattr(worker, "_terminate_process_tree", terminate)
    monkeypatch.setattr(worker, "terminate_registered_processes", cleanup)
    monkeypatch.setattr(worker, "finalize_execution", finalize)
    return instance, job, events, now


def test_worker_gives_supervisor_grace_before_forced_cleanup(monkeypatch):
    instance, job, events, now = setup_worker(monkeypatch)
    instance._tick_running()
    assert job.cancel_seen_at == now
    assert events == []
    job.cancel_seen_at = now - timedelta(seconds=worker.ULTRALYTICS_DDP_CANCEL_FALLBACK_SECONDS)
    instance._tick_running()
    assert events == ["terminate", "cleanup", "finalize"]
    assert instance._running is None


def test_worker_cleans_registered_orphans_after_supervisor_exit(monkeypatch):
    instance, _, events, _ = setup_worker(monkeypatch, exited=True)
    instance._tick_running()
    assert events == ["cleanup", "finalize"]


def test_worker_does_not_finalize_while_registered_process_survives(monkeypatch):
    instance, _, events, _ = setup_worker(monkeypatch, exited=True)
    monkeypatch.setattr(worker, "terminate_registered_processes", lambda *args, **kwargs: [SimpleNamespace(pid=404)])
    with pytest.raises(worker.UltralyticsDDPError, match="404"):
        instance._tick_running()
    assert events == []
    assert instance._running is not None


@pytest.mark.parametrize("outcome", ["survivor", "unknown"])
def test_worker_retries_cleanup_before_releasing_job(monkeypatch, outcome):
    instance, job, events, _ = setup_worker(monkeypatch, exited=True)
    attempts = []

    def cleanup(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            if outcome == "unknown":
                raise worker.UltralyticsDDPCleanupIncomplete(
                    "unknown process", run_id="run", attempt_id="attempt",
                    execution_owner=kwargs["owner"], survivors=[],
                )
            return [SimpleNamespace(pid=404)]
        return []

    monkeypatch.setattr(worker, "terminate_registered_processes", cleanup)
    with pytest.raises(worker.UltralyticsDDPError):
        instance._tick_running()
    assert instance._running is job
    assert not job.stdout_f.closed and not job.stderr_f.closed
    assert events == []
    instance._tick_running()
    assert events == ["finalize"]
    assert instance._running is None
    assert job.stdout_f.closed and job.stderr_f.closed


@pytest.mark.parametrize("state", ["live", "unknown", "survivor", "clean", "corrupt"])
def test_stale_ddp_requires_confirmed_cleanup(tmp_path, monkeypatch, state):
    import json

    instance = worker.DbQueueWorker(worker_id="worker")
    run = SimpleNamespace(run_id="run", pid=101, worker_id="worker")
    owner = {"guard_pid": 101, "guard_create_time": 123.5, "worker_id": "worker", "process_scope": LOCAL_SCOPE}
    attempt = tmp_path / "run" / "runtime" / "ddp" / "attempt"
    attempt.mkdir(parents=True)
    context = {
        "run_id": "run", "attempt_id": "attempt", "execution_owner": owner,
        "supervisor": {"pid": 101, "create_time": 123.5},
    }
    (attempt / "context.json").write_text("{" if state == "corrupt" else json.dumps(context))
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker, "_identity_is_live", lambda identity: {"live": True, "unknown": None}.get(state, False))
    calls = []

    def cleanup(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["owner"] == owner
        assert kwargs["attempt_id"] == "attempt"
        return [SimpleNamespace(pid=404)] if state == "survivor" else []

    monkeypatch.setattr(worker, "terminate_registered_processes", cleanup)
    assert instance._cleanup_stale_ddp(run) is (state == "clean")
    assert len(calls) == (1 if state in {"survivor", "clean"} else 0)
