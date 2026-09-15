import json
import time
from types import SimpleNamespace

import pytest

from train_platform.platform.runtime import execution_processes as runtime


SCOPE = {"boot_id": "boot", "pid_namespace": {"device": 1, "inode": 2}}
OWNER = {"allocation_id": "allocation", "guard_pid": 101, "guard_create_time": 1.0, "process_scope": SCOPE}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.process_scope, "get_process_scope", lambda: SCOPE)
    root = runtime.execution_root(tmp_path, "allocation")
    processes = root / "processes"
    processes.mkdir(parents=True)
    identity = {"run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                "process_scope": SCOPE, "role": "supervisor", "pid": 101, "create_time": 1.0}
    (processes / "supervisor-101.json").write_text(json.dumps(identity))
    return tmp_path, root


def cleanup(root, **kwargs):
    return runtime.cleanup_registered_execution(root, run_id="run", allocation_id="allocation",
                                                execution_owner=OWNER, **kwargs)


def test_empty_registry_cannot_prove_cleanup(tmp_path):
    (runtime.execution_root(tmp_path, "allocation") / "processes").mkdir(parents=True)
    assert not cleanup(tmp_path)["complete"]


def test_exited_supervisor_produces_allocation_scoped_proof(registry, monkeypatch):
    root, _ = registry
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))
    proof = cleanup(root)
    assert proof["complete"]
    assert proof["execution_owner"] == OWNER
    assert proof["allocation_id"] == "allocation"
    assert not proof["supervisor_excluded"]


def test_unknown_identity_keeps_cleanup_pending(registry, monkeypatch):
    root, _ = registry
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("unknown", None))
    proof = cleanup(root)
    assert not proof["complete"]
    assert proof["unknown"] == [101]


def test_child_cleanup_proof_does_not_claim_supervisor_exit(registry, monkeypatch):
    root, _ = registry
    monkeypatch.setattr(runtime, "process_state", lambda identity: (_ for _ in ()).throw(AssertionError("supervisor must be excluded")))
    proof = cleanup(root, exclude_supervisor=True)
    assert proof["complete"]
    assert proof["supervisor_excluded"]


def test_corrupt_registration_fails_closed(registry):
    root, directory = registry
    (directory / "processes" / "rank.json").write_text("not-json")
    assert not cleanup(root)["complete"]


def write_error(root, name, **values):
    directory = root / "registration-errors"
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": name, "allocation_id": "allocation", "execution_owner": OWNER,
        "stage": "descendant_registration_write", "error": "write failed",
        "target": {"pid": 202, "create_time": 2.0, "process_scope": SCOPE},
        "resolved": False,
    }
    entry.update(values)
    (directory / f"{name}.json").write_text(json.dumps(entry), encoding="utf-8")


def test_independent_target_error_can_resolve_while_scan_gap_remains(registry, monkeypatch):
    root, execution = registry
    write_error(execution, "target")
    write_error(execution, "scan", stage="descendant_scan",
                target={"pid": 101, "create_time": 1.0, "process_scope": SCOPE})
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))

    proof = cleanup(root)

    assert not proof["complete"]
    assert [item["stage"] for item in proof["registration_errors"]] == ["descendant_scan"]
    target = json.loads((execution / "registration-errors" / "target.json").read_text())
    assert target["resolved"] is True and target["last_check"] == "dead"


def test_untrusted_error_target_is_never_inspected_or_terminated(registry, monkeypatch):
    root, execution = registry
    other_owner = {**OWNER, "allocation_id": "other"}
    write_error(execution, "foreign", execution_owner=other_owner,
                target={"pid": 999, "create_time": 9.0, "process_scope": SCOPE})
    inspected = []

    def state(identity):
        inspected.append(identity.get("pid"))
        assert identity.get("pid") != 999
        return "dead", None

    monkeypatch.setattr(runtime, "process_state", state)
    proof = cleanup(root)
    assert not proof["complete"]
    assert 999 not in inspected
    assert proof["registration_errors"][0]["stage"] == "registration_ownership_mismatch"


def test_watcher_never_scans_children_after_supervisor_identity_is_dead(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))

    class ReusedPid:
        def children(self, recursive=False):
            pytest.fail("reused supervisor PID must not be scanned")

    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: ReusedPid())
    stopped, thread = runtime.start_descendant_registration(
        tmp_path, run_id="run", allocation_id="allocation", execution_owner=OWNER,
        supervisor_pid=101, interval_seconds=0.01,
    )
    time.sleep(0.03)
    stopped.set()
    thread.join(timeout=1)
    assert not thread.is_alive()


def session_fakes(monkeypatch):
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="posix", getsid=lambda pid: 101))
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))
    monkeypatch.setattr(runtime.psutil, "Process",
                        lambda pid: (_ for _ in ()).throw(runtime.psutil.NoSuchProcess(pid)))
    monkeypatch.setattr(runtime.psutil, "process_iter", lambda: iter([]))
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda processes, timeout: ([], []))


def test_legacy_scan_error_resolves_with_owned_complete_session_twice(registry, monkeypatch):
    training_root, root = registry
    supervisor_path = root / "processes" / "supervisor-101.json"
    supervisor = json.loads(supervisor_path.read_text())
    supervisor["sid"] = 101
    supervisor_path.write_text(json.dumps(supervisor))
    (root / "registration-error.json").write_text(json.dumps(
        {"allocation_id": "allocation", "error": "failed scan"}))
    session_fakes(monkeypatch)

    assert cleanup(training_root)["complete"]
    assert cleanup(training_root)["complete"]
    assert json.loads((root / "registration-error.json").read_text())["resolved"] is True


def test_legacy_scan_error_without_owned_session_stays_pending(registry, monkeypatch):
    training_root, root = registry
    (root / "registration-error.json").write_text(json.dumps(
        {"allocation_id": "allocation", "error": "failed scan"}))
    session_fakes(monkeypatch)
    assert not cleanup(training_root)["complete"]


def test_dead_missing_base_recovers_from_durable_supervisor_error(registry, monkeypatch):
    training_root, root = registry
    (root / "processes" / "supervisor-101.json").unlink()
    target = {
        "pid": 101, "create_time": 1.0, "process_scope": SCOPE, "sid": 101,
        "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
    }
    runtime._record_registration_error(
        training_root, "allocation", OWNER, stage="supervisor_registration_write",
        error="write failed", target=target,
    )
    session_fakes(monkeypatch)
    proof = cleanup(training_root, assigned_gpu_uuids=["GPU-a", "GPU-b"])
    assert proof["complete"] and proof["recovered_registration"]
    recovered = json.loads((root / "processes" / "supervisor-101.json").read_text())
    assert recovered["assigned_gpu_uuids"] == ["GPU-a", "GPU-b"]


def test_corrupt_registration_stays_pending_despite_complete_owned_session(registry, monkeypatch):
    training_root, root = registry
    supervisor_path = root / "processes" / "supervisor-101.json"
    supervisor = json.loads(supervisor_path.read_text())
    supervisor["sid"] = 101
    supervisor_path.write_text(json.dumps(supervisor))
    (root / "processes" / "corrupt.json").write_text("not-json")
    session_fakes(monkeypatch)
    assert not cleanup(training_root)["complete"]


class FakeProcess:
    def __init__(self, pid):
        self.pid, self.alive, self.terminated = pid, True, False
    def is_running(self): return self.alive
    def status(self): return runtime.psutil.STATUS_RUNNING
    def terminate(self): self.terminated = True; self.alive = False
    def kill(self): self.terminated = True; self.alive = False


def install_fake_registration(monkeypatch, training_root, processes):
    captured = []
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runtime, "process_state", lambda identity:
                        (("live", processes[identity["pid"]]) if processes.get(identity["pid"], None) and
                         processes[identity["pid"]].alive else ("dead", None)))
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda values, timeout: (list(values), []))

    def register(root, **kwargs):
        expected = dict(kwargs["expected_identity"])
        captured.append(expected)
        identity = {**expected, "run_id": kwargs["run_id"], "allocation_id": kwargs["allocation_id"],
                    "execution_owner": dict(kwargs["execution_owner"]), "role": kwargs["role"],
                    "assigned_gpu_uuids": list(kwargs.get("assigned_gpu_uuids") or [])}
        path = runtime.execution_root(root, kwargs["allocation_id"]) / "processes" / f"{kwargs['role']}-{kwargs['pid']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity))
        return identity

    monkeypatch.setattr(runtime, "register_execution_process", register)
    return captured


def test_live_missing_supervisor_is_registered_and_terminated(tmp_path, monkeypatch):
    supervisor = FakeProcess(101)
    captured = install_fake_registration(monkeypatch, tmp_path, {101: supervisor})
    proof = cleanup(tmp_path)
    assert proof["complete"] and proof["recovered_registration"]
    assert supervisor.terminated
    assert captured == [{"pid": 101, "create_time": 1.0, "process_scope": SCOPE}]


def test_live_failed_target_retries_exact_identity_and_scan_gap_remains(registry, monkeypatch):
    training_root, root = registry
    target_process = FakeProcess(202)
    captured = install_fake_registration(monkeypatch, training_root, {202: target_process})
    write_error(root, "target")
    write_error(root, "gap", stage="descendant_scan",
                target={"pid": 101, "create_time": 1.0, "process_scope": SCOPE})
    proof = cleanup(training_root)
    assert not proof["complete"] and target_process.terminated
    assert {"pid": 202, "create_time": 2.0, "process_scope": SCOPE} in captured
    assert any(item["stage"] == "descendant_scan" for item in proof["registration_errors"])


def test_owned_registration_with_incomplete_identity_reports_read_error(registry, monkeypatch):
    training_root, root = registry
    (root / "processes" / "incomplete.json").write_text(json.dumps({
        "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
        "process_scope": SCOPE, "role": "descendant",
    }))
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert any(item["stage"] == "registration_read" and "incomplete" in item["error"]
               for item in proof["registration_errors"])
