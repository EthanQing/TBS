import os
from types import SimpleNamespace

import pytest

from train_platform.platform.runtime import ultralytics_ddp as ddp
from train_platform.platform.runtime import process_scope
from train_platform.workers.training import train_entry_impl as entry


LOCAL_SCOPE = {"boot_id": "test-boot", "pid_namespace": {"device": 1, "inode": 2}}


@pytest.fixture(autouse=True)
def local_process_scope(monkeypatch):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)


@pytest.mark.parametrize("outcome", ["incomplete", "failed", "cancelled"])
def test_entry_defers_finalization_only_for_unfinished_cleanup(monkeypatch, outcome):
    original = RuntimeError("rank failed")
    error = (
        ddp.UltralyticsDDPCleanupIncomplete(
            "cleanup incomplete",
            run_id="run", attempt_id="attempt",
            execution_owner={"guard_pid": 123, "guard_create_time": 12.0, "worker_id": "worker", "process_scope": LOCAL_SCOPE},
            survivors=[{"pid": 456, "create_time": 34.0}], original_error=original,
        )
        if outcome == "incomplete"
        else ddp.UltralyticsDDPCancelled("cancelled") if outcome == "cancelled" else original
    )
    opened = []
    finalized = []
    monkeypatch.setattr(entry, "assert_valid_license", lambda: None)
    monkeypatch.setattr(entry, "_wait_for_execution_guard_pid", lambda *args, **kwargs: 123)

    def session():
        opened.append(True)
        if len(opened) == 1:
            raise error
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(entry, "SessionLocal", session)
    monkeypatch.setattr(entry, "finalize_execution", lambda *args, **kwargs: finalized.append(kwargs))
    assert entry.main(["--run-id", "run"]) == (0 if outcome == "cancelled" else 1)
    if outcome == "incomplete":
        assert finalized == []
        assert len(opened) == 1
    else:
        assert len(finalized) == 1
        assert finalized[0]["expected_pid"] == 123


@pytest.mark.parametrize("still_alive", [False, True])
@pytest.mark.parametrize("training_error", [None, "registration", "exit", "marker"])
def test_runtime_rechecks_candidates_after_later_cleanup_round(monkeypatch, tmp_path, still_alive, training_error):
    class Process:
        pid = 91001
        live = True

        def create_time(self):
            return 12.0

        def is_running(self):
            return self.live

        def status(self):
            return ddp.psutil.STATUS_RUNNING if self.live else ddp.psutil.STATUS_ZOMBIE

        def children(self, **kwargs):
            return []

        def terminate(self):
            pass

        def kill(self):
            pass

    process = Process()
    code = 7 if training_error == "exit" else 0
    launcher = SimpleNamespace(pid=process.pid, poll=lambda: code, wait=lambda **kwargs: code)
    monkeypatch.setattr(ddp.subprocess, "Popen", lambda *args, **kwargs: launcher)
    monkeypatch.setattr(ddp.psutil, "Process", lambda *args, **kwargs: process)
    monkeypatch.setattr(ddp.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(ddp, "terminate_registered_processes", lambda *args, **kwargs: [process])
    original = OSError("registration write failed")
    if training_error == "registration":
        monkeypatch.setattr(ddp, "register_process", lambda *args, **kwargs: (_ for _ in ()).throw(original))

    if training_error == "marker":
        write_json = ddp._atomic_json
        def fail_marker(path, value):
            if path.name == "cleanup-pending.json":
                raise OSError("marker write failed")
            write_json(path, value)
        monkeypatch.setattr(ddp, "_atomic_json", fail_marker)

    def wait(processes, **kwargs):
        process.live = still_alive
        return ([], list(processes)) if still_alive else (list(processes), [])

    monkeypatch.setattr(ddp.psutil, "wait_procs", wait)
    context = {
        "run_id": "run", "run_root": str(tmp_path), "world_size": 2,
        "cuda_visible_devices": "0,1", "execution_owner": {"guard_pid": os.getpid(), "process_scope": LOCAL_SCOPE},
    }

    def execute():
        ddp.run_ultralytics_ddp(context, cancel_requested=lambda: False, upsert_epoch_metrics=lambda *args: None)

    if still_alive:
        with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete) as caught:
            execute()
        assert caught.value.run_id == "run"
        assert caught.value.survivors[0]["pid"] == process.pid
        if training_error == "registration":
            assert caught.value.original_error is original
        if training_error == "exit":
            assert "code 7" in str(caught.value.original_error)
        if training_error == "marker":
            assert any("marker write failed" in str(e) for e in caught.value.cleanup_errors)
    elif training_error == "exit":
        with pytest.raises(ddp.UltralyticsDDPError, match="code 7"):
            execute()
    elif training_error == "registration":
        with pytest.raises(OSError) as caught:
            execute()
        assert caught.value is original
    else:
        execute()


@pytest.mark.parametrize("broken", ["context", "registration", "pending"])
def test_registered_cleanup_cannot_succeed_with_unreadable_identity(tmp_path, broken):
    import json

    attempt = tmp_path / "runtime" / "ddp" / "attempt"
    registrations = attempt / "processes"
    registrations.mkdir(parents=True)
    context = {"run_id": "run", "attempt_id": "attempt", "execution_owner": {"guard_pid": 123, "process_scope": LOCAL_SCOPE}}
    (attempt / "context.json").write_text(json.dumps(context))
    target = {
        "context": attempt / "context.json",
        "registration": registrations / "rank-0.json",
        "pending": attempt / "cleanup-pending.json",
    }[broken]
    target.write_text("{")
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete):
        ddp.terminate_registered_processes(tmp_path, run_id="run", owner={"guard_pid": 123, "process_scope": LOCAL_SCOPE})


def test_unknown_process_is_preserved_for_worker_retry(tmp_path, monkeypatch):
    import json

    attempt = tmp_path / "runtime" / "ddp" / "attempt"
    registrations = attempt / "processes"
    registrations.mkdir(parents=True)
    scope = {"run_id": "run", "attempt_id": "attempt", "execution_owner": {"guard_pid": 123, "process_scope": LOCAL_SCOPE}}
    (attempt / "context.json").write_text(json.dumps(scope))
    (registrations / "rank-0.json").write_text(json.dumps({**scope, "pid": 404, "create_time": 10.0}))
    monkeypatch.setattr(ddp.psutil, "Process", lambda pid: (_ for _ in ()).throw(ddp.psutil.AccessDenied(pid)))
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete) as caught:
        ddp.terminate_registered_processes(tmp_path, run_id="run", owner=scope["execution_owner"])
    assert caught.value.survivors[0]["state"] == "unknown"
    pending = json.loads((attempt / "cleanup-pending.json").read_text())
    assert pending["survivors"][0]["pid"] == 404
    monkeypatch.setattr(ddp.psutil, "Process", lambda pid: (_ for _ in ()).throw(ddp.psutil.NoSuchProcess(pid)))
    assert ddp.terminate_registered_processes(tmp_path, run_id="run", owner=scope["execution_owner"]) == []
    assert not (attempt / "cleanup-pending.json").exists()
