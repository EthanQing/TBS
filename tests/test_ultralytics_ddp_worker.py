from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_platform.workers import worker_impl as worker


def setup_worker(monkeypatch, *, exited=False):
    now = datetime.now(timezone.utc)
    process = SimpleNamespace(pid=101, returncode=0 if exited else None)
    process.poll = lambda: process.returncode
    job = worker.RunningJob(
        run_id="run", engine="ultralytics-yolo", proc=process,
        stdout_path=Path("stdout"), stderr_path=Path("stderr"),
        stdout_f=StringIO(), stderr_f=StringIO(),
        guard_create_time=123.5, ultralytics_ddp=True,
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
        assert kwargs["owner"] == {"guard_pid": 101, "guard_create_time": 123.5, "worker_id": "worker"}
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
