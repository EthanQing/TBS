from contextlib import contextmanager
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

from train_platform.platform.runtime.gpu_probe import GpuProbeResult
from train_platform.workers import gpu_resource_reporter as reporter


def test_reporter_retries_registration_and_probe_failure(monkeypatch):
    registrations = []
    reports = []
    stopped = []
    heartbeats = []
    sessions = []
    probe_calls = []
    completed = threading.Event()

    @contextmanager
    def session_scope():
        db = object()
        sessions.append(db)
        yield db

    def register(db, **kwargs):
        registrations.append(kwargs)
        if len(registrations) == 1:
            raise RuntimeError("transient registration outage")

    def probe():
        probe_calls.append(1)
        if len(probe_calls) == 1:
            raise RuntimeError("transient probe error")
        return GpuProbeResult("empty", "nvml", datetime.now(timezone.utc), complete=True)

    def save(db, instance_id, result):
        reports.append((db, instance_id, result))
        if result.status == "empty":
            completed.set()

    monkeypatch.setattr(reporter, "settings", SimpleNamespace(
        gpu_inventory_enabled=True, gpu_inventory_interval_seconds=0.01, gpu_node_id=None))
    monkeypatch.setattr(reporter, "session_scope", session_scope)
    monkeypatch.setattr(reporter, "register_worker_instance", register)
    monkeypatch.setattr(reporter, "save_inventory", save)
    monkeypatch.setattr(reporter, "mark_worker_stopped", lambda db, instance_id: stopped.append(instance_id))
    monkeypatch.setattr(reporter.process_scope, "get_process_scope", lambda: {"boot_id": "test"})
    monkeypatch.setattr(reporter, "probe_gpus", probe)
    # Heartbeats use the same explicit independent-session seam.
    monkeypatch.setattr(reporter, "update_worker_heartbeat", lambda *a, **kw: heartbeats.append(a), raising=False)
    instance = reporter.GpuResourceReporter(worker_id="worker", allowed_engines={"ultralytics-yolo"})
    instance.start()
    instance.start()
    try:
        assert completed.wait(3), "reporter did not recover after transient errors"
    finally:
        instance.stop()
    assert {r["instance_id"] for r in registrations} == {instance.instance_id}
    assert len(registrations) == 2
    assert registrations[0]["started_at"] == registrations[1]["started_at"]
    assert registrations[1]["node_id"] is None
    assert registrations[1]["process_scope"] == {"boot_id": "test"}
    assert registrations[1]["allowed_engines"] == ["ultralytics-yolo"]
    assert any(result.status == "failed" for _, _, result in reports)
    assert len({id(db) for db in sessions}) == len(sessions)
    assert stopped == [instance.instance_id]
    assert heartbeats


def test_disabled_reporter_never_creates_session(monkeypatch):
    monkeypatch.setattr(reporter, "settings", SimpleNamespace(gpu_inventory_enabled=False))
    calls = []
    monkeypatch.setattr(reporter, "session_scope", lambda: calls.append(1))
    instance = reporter.GpuResourceReporter(worker_id="worker", allowed_engines={"paddle-det"})
    instance.start()
    instance.stop()
    assert calls == []


def test_worker_signal_context_restores_handlers(monkeypatch):
    from train_platform.workers import worker_impl as worker
    import pytest

    previous = object()
    installed = {}
    monkeypatch.setattr(worker.signal, "getsignal", lambda sig: previous)
    monkeypatch.setattr(worker.signal, "signal", lambda sig, handler: installed.update({sig: handler}))
    with pytest.raises(SystemExit):
        with worker.worker_shutdown_signals():
            installed[worker.signal.SIGTERM](worker.signal.SIGTERM, None)
    assert all(handler is previous for handler in installed.values())


def test_worker_signal_context_can_run_outside_main_thread(monkeypatch):
    from train_platform.workers import worker_impl as worker

    def cannot_install(*args):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(worker.signal, "signal", cannot_install)
    with worker.worker_shutdown_signals():
        pass
