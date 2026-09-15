from types import SimpleNamespace

import pytest

from train_platform.models.v3.gpu_allocation import GpuAllocation
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.runtime import execution_processes
from train_platform.domains.training.resources import lifecycle
from train_platform.workers.training import train_entry_impl as entry


SCOPE = {"boot_id": "boot", "pid_namespace": {"device": 1, "inode": 2}}


class Query:
    def __init__(self, value): self.value = value
    def options(self, *args): return self
    def filter(self, *args): return self
    def filter_by(self, **kwargs): return self
    def first(self): return self.value


class Db:
    def __init__(self, run, allocation, log):
        self.run, self.allocation, self.log = run, allocation, log
    def query(self, model): return Query(self.run if model is TrainingRun else self.allocation if model is GpuAllocation else None)
    def commit(self): self.log.append("commit")
    def rollback(self): self.log.append("rollback")
    def close(self): self.log.append("close")


@pytest.mark.parametrize("failure_stage", ["register", "activate"])
def test_managed_start_failure_before_commit_cleans_prepared_owner(tmp_path, monkeypatch, failure_stage):
    log = []
    run = SimpleNamespace(parameters=object(), standard_dataset=object(), architecture=object())
    allocation = SimpleNamespace(
        worker_id="worker", devices=[SimpleNamespace(gpu_uuid="GPU-a", ordinal=0, reserved_memory_mib=1)],
        request_snapshot={"sharing": "exclusive"},
    )
    monkeypatch.setattr(entry, "assert_valid_license", lambda: None)
    monkeypatch.setattr(entry, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(entry, "SessionLocal", lambda: Db(run, allocation, log))
    monkeypatch.setattr(entry.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 1.0))
    monkeypatch.setattr(entry.process_scope, "get_process_scope", lambda: SCOPE)

    def register(*args, **kwargs):
        log.append(("register", kwargs["execution_owner"]))
        if failure_stage == "register": raise OSError("registration failed")

    def activate(*args, **kwargs):
        log.append(("activate", kwargs["execution_owner"]))
        if failure_stage == "activate": raise ValueError("authorization rejected")
        return {"execution_owner": kwargs["execution_owner"]}

    monkeypatch.setattr(execution_processes, "register_execution_process", register)
    monkeypatch.setattr(lifecycle, "activate_allocation", activate)
    monkeypatch.setattr(entry, "get_trainer", lambda *a, **k: pytest.fail("training must not start"), raising=False)
    cleanup_calls = []
    monkeypatch.setattr(execution_processes, "cleanup_registered_execution",
                        lambda *a, **kw: cleanup_calls.append(kw) or {"complete": True})

    result = entry.main(["--run-id", "run", "--allocation-id", "allocation",
                         "--worker-instance-id", "instance"])

    assert result == 1 and "commit" not in log
    assert cleanup_calls and cleanup_calls[0]["exclude_supervisor"] is True
    prepared = log[0][1]
    assert cleanup_calls[0]["execution_owner"] == prepared
    if failure_stage == "register":
        assert not any(item[0] == "activate" for item in log if isinstance(item, tuple))
    else:
        assert [item[0] for item in log if isinstance(item, tuple)] == ["register", "activate"]
        assert "rollback" in log


def test_execution_record_is_published_only_after_activation_commit(tmp_path, monkeypatch):
    log = []
    run = SimpleNamespace(parameters=object(), standard_dataset=object(), architecture=object())
    allocation = SimpleNamespace(worker_id="worker", devices=[SimpleNamespace(
        gpu_uuid="GPU-a", ordinal=0, reserved_memory_mib=1)], request_snapshot={"sharing": "exclusive"})
    monkeypatch.setattr(entry, "assert_valid_license", lambda: None)
    monkeypatch.setattr(entry, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(entry, "SessionLocal", lambda: Db(run, allocation, log))
    monkeypatch.setattr(entry.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 1.0))
    monkeypatch.setattr(entry.process_scope, "get_process_scope", lambda: SCOPE)
    monkeypatch.setattr(execution_processes, "register_execution_process",
                        lambda *a, **kw: log.append(("register", kw["execution_owner"])))
    monkeypatch.setattr(lifecycle, "activate_allocation",
                        lambda *a, **kw: log.append(("activate", kw["execution_owner"])) or
                        {"execution_owner": kw["execution_owner"]})
    monkeypatch.setattr(entry, "_write_execution_record",
                        lambda *a, **kw: log.append(("record", a[2])) or (_ for _ in ()).throw(OSError("stop")))
    monkeypatch.setattr(execution_processes, "cleanup_registered_execution", lambda *a, **kw: {"complete": True})
    monkeypatch.setattr(lifecycle, "record_execution_result", lambda *a, **kw: None)

    assert entry.main(["--run-id", "run", "--allocation-id", "allocation",
                       "--worker-instance-id", "instance"]) == 1
    names = [item[0] if isinstance(item, tuple) else item for item in log]
    assert names[:4] == ["register", "activate", "commit", "record"]
    owners = [item[1] for item in log if isinstance(item, tuple)]
    assert owners[0] == owners[1] == owners[2]
