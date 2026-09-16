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
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    root = runtime.execution_root(tmp_path, "allocation")
    processes = root / "processes"
    processes.mkdir(parents=True)
    identity = {"run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                "process_scope": SCOPE, "role": "supervisor", "pid": 101, "create_time": 1.0, "sid": 101}
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
        "target": {"pid": 202, "create_time": 2.0, "process_scope": SCOPE,
                   "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER},
        "resolved": False,
    }
    entry.update(values)
    entry["target"] = {"run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                       **entry["target"]}
    (directory / f"{name}.json").write_text(json.dumps(entry), encoding="utf-8")


def test_dead_target_does_not_resolve_unchecked_descendants(registry, monkeypatch):
    root, execution = registry
    write_error(execution, "target")
    write_error(execution, "scan", stage="descendant_scan",
                target={"pid": 101, "create_time": 1.0, "process_scope": SCOPE})
    monkeypatch.setattr(runtime, "process_state", lambda identity: ("dead", None))

    proof = cleanup(root)

    assert not proof["complete"]
    assert len(proof["registration_errors"]) == 2
    target = json.loads((execution / "registration-errors" / "target.json").read_text())
    assert target["resolved"] is False and target["last_check"] == "dead"


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
    path = root / "processes" / "supervisor-101.json"
    identity = json.loads(path.read_text())
    identity["sid"] = None
    path.write_text(json.dumps(identity))
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
    recovered = json.loads((root / "processes" / "recovered-202.json").read_text())
    assert recovered["pid"] == 202 and recovered["create_time"] == 2.0
    assert captured == []
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


def failed_leader(root, *, resolved=False):
    target = {"pid": 202, "create_time": 2.0, "sid": 202, "pgid": 202,
              "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
              "process_scope": SCOPE, "assigned_gpu_uuids": ["GPU-a"]}
    write_error(root, "leader", target=target, resolved=resolved)
    return target


@pytest.mark.parametrize("resolved", [False, True])
def test_dead_failed_leader_session_is_scanned_on_every_retry(registry, monkeypatch, resolved):
    training_root, root = registry
    target = failed_leader(root, resolved=resolved)
    child = FakeProcess(303)
    child.create_time = lambda: 3.0
    session_fakes(monkeypatch)
    monkeypatch.setattr(runtime.os, "getsid", lambda pid: 202)
    monkeypatch.setattr(runtime.psutil, "process_iter", lambda: iter([child] if child.alive else []))
    monkeypatch.setattr(runtime, "process_state", lambda identity:
                        ("live", child) if identity["pid"] == 303 and child.alive else ("dead", None))

    def register(path, **kwargs):
        assert kwargs["pid"] == 303, "dead leader must not be recaptured"
        assert kwargs["expected_identity"]["sid"] == 202
        return {**kwargs["expected_identity"], "sid": 202}

    monkeypatch.setattr(runtime, "register_execution_process", register)
    proof = cleanup(training_root, terminate=False)
    assert not proof["complete"] and proof["survivors"] == [303]
    proof = cleanup(training_root)
    assert proof["complete"] and child.terminated
    stored = json.loads((root / "registration-errors" / "leader.json").read_text())
    assert stored["target"] == target and stored["resolved"]
    # Reappearance must be scanned even after the error was resolved.
    child.alive = True
    assert not cleanup(training_root, terminate=False)["complete"]


def test_reused_session_leader_blocks_cleanup_without_termination(registry, monkeypatch):
    training_root, root = registry
    failed_leader(root, resolved=True)
    session_fakes(monkeypatch)
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 99.0))
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert proof["unconfirmed_sessions"][0]["reason"] == "session leader PID was reused"


def test_session_enumeration_failure_blocks_resolved_error(registry, monkeypatch):
    training_root, root = registry
    failed_leader(root, resolved=True)
    session_fakes(monkeypatch)
    monkeypatch.setattr(runtime.psutil, "process_iter", lambda: (_ for _ in ()).throw(OSError("scan denied")))
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert "enumeration failed" in proof["unconfirmed_sessions"][0]["reason"]


def test_registration_write_failure_preserves_full_identity(tmp_path, monkeypatch):
    identity = {"pid": 202, "create_time": 2.0, "sid": 202, "pgid": 202,
                "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                "process_scope": SCOPE, "assigned_gpu_uuids": ["GPU-a"]}
    monkeypatch.setattr(runtime.process_scope, "get_process_scope", lambda: SCOPE)
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runtime, "process_identity", lambda *args, **kwargs: dict(identity))
    monkeypatch.setattr(runtime, "process_state", lambda value: ("live", None))
    atomic = runtime._atomic_json

    def fail_registration(path, value):
        if path.parent.name == "processes":
            raise OSError("write failed")
        atomic(path, value)

    monkeypatch.setattr(runtime, "_atomic_json", fail_registration)
    with pytest.raises(OSError):
        runtime.register_execution_process(tmp_path, run_id="run", allocation_id="allocation",
                                           execution_owner=OWNER, pid=202, role="descendant")
    entries = list((runtime.execution_root(tmp_path, "allocation") / "registration-errors").glob("*.json"))
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["target"] == identity


def test_sid_capture_survives_pgid_failure(monkeypatch):
    from train_platform.platform.runtime import execution_identity
    monkeypatch.setattr(execution_identity.process_scope, "get_process_scope", lambda: SCOPE)
    monkeypatch.setattr(execution_identity.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 2.0))
    monkeypatch.setattr(execution_identity, "os", SimpleNamespace(
        name="posix", getsid=lambda pid: 202,
        getpgid=lambda pid: (_ for _ in ()).throw(OSError("unavailable"))))
    identity = execution_identity.process_identity(202)
    assert identity["sid"] == 202 and identity["pgid"] is None


@pytest.mark.skipif(__import__("sys").platform != "linux", reason="requires Linux sessions")
def test_linux_failed_session_leader_exit_cleans_reparented_child(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from train_platform.platform.runtime.execution_identity import process_identity

    scope = runtime.process_scope.get_process_scope()
    owner = {"allocation_id": "allocation", "guard_pid": os.getpid(),
             "guard_create_time": runtime.psutil.Process().create_time(), "process_scope": scope}
    runtime.register_execution_process(tmp_path, run_id="run", allocation_id="allocation",
                                       execution_owner=owner, pid=os.getpid(), role="supervisor")
    script = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"
              "'import time; time.sleep(120)']); print(p.pid,flush=True); sys.stdin.readline()")
    leader = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    child = None
    try:
        child = runtime.psutil.Process(int(leader.stdout.readline()))
        original = process_identity(leader.pid, run_id="run", allocation_id="allocation",
                                    execution_owner=owner, assigned_gpu_uuids=["GPU-a"])
        atomic = runtime._atomic_json

        def fail_registration(path, value):
            if path.parent.name == "processes" and value["pid"] == leader.pid:
                raise OSError("injected registration failure")
            atomic(path, value)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_atomic_json", fail_registration)
            with pytest.raises(OSError):
                runtime.register_execution_process(tmp_path, run_id="run", allocation_id="allocation",
                                                   execution_owner=owner, pid=leader.pid, role="descendant",
                                                   assigned_gpu_uuids=["GPU-a"], expected_identity=original)
        leader.stdin.write("exit\n")
        leader.stdin.flush()
        leader.wait(timeout=5)
        assert child.is_running() and child.ppid() != leader.pid
        proof = runtime.cleanup_registered_execution(
            tmp_path, run_id="run", allocation_id="allocation", execution_owner=owner,
            exclude_supervisor=True, grace_seconds=0.1)
        assert proof["complete"], proof
        assert not child.is_running() or child.status() == runtime.psutil.STATUS_ZOMBIE
        assert runtime.cleanup_registered_execution(
            tmp_path, run_id="run", allocation_id="allocation", execution_owner=owner,
            exclude_supervisor=True, grace_seconds=0.1)["complete"]
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)
        if child is not None:
            try:
                child.kill()
            except runtime.psutil.NoSuchProcess:
                pass
        leader.stdin.close()
        leader.stdout.close()


@pytest.mark.parametrize("field,value", [("run_id", "other"), ("allocation_id", "other"),
                                         ("execution_owner", {}), ("process_scope", {})])
def test_error_target_ownership_fields_are_required(registry, monkeypatch, field, value):
    training_root, root = registry
    target = failed_leader(root, resolved=True)
    target[field] = value
    write_error(root, "leader", target=target, resolved=True)
    session_fakes(monkeypatch)
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert proof["registration_errors"][0]["stage"] == "registration_ownership_mismatch"


def test_resolved_target_without_session_identity_is_unconfirmed(registry, monkeypatch):
    training_root, root = registry
    write_error(root, "target", resolved=True)
    session_fakes(monkeypatch)
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert proof["unconfirmed_sessions"][0]["reason"] == "original session identity is unavailable"
    assert proof["registration_errors"]


def test_error_journal_write_failure_keeps_cleanup_pending(registry, monkeypatch):
    training_root, root = registry
    target = failed_leader(root)
    session_fakes(monkeypatch)
    monkeypatch.setattr(runtime, "_atomic_json", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert any(item["stage"] == "journal_write" for item in proof["registration_errors"])
    assert json.loads((root / "registration-errors" / "leader.json").read_text())["target"] == target



def test_conflicting_original_session_identity_is_not_terminated(registry, monkeypatch):
    training_root, root = registry
    target = failed_leader(root)
    other = {**target, "sid": 404}
    (root / "processes" / "descendant-202.json").write_text(json.dumps(other))
    session_fakes(monkeypatch)
    proof = cleanup(training_root)
    assert not proof["complete"] and 202 in proof["unknown"]
    assert any(item["stage"] == "identity_conflict" for item in proof["registration_errors"])


def test_resolved_nonleader_without_confirmed_session_leader_stays_pending(registry, monkeypatch):
    training_root, root = registry
    target = failed_leader(root, resolved=True)
    target["sid"] = 404
    write_error(root, "leader", target=target, resolved=True)
    session_fakes(monkeypatch)
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert proof["unconfirmed_sessions"][0]["reason"] == "no confirmed original session leader"


def test_session_ownership_changed_during_cleanup_cannot_complete(registry, monkeypatch):
    training_root, root = registry
    failed_leader(root, resolved=True)
    session_fakes(monkeypatch)
    calls = 0

    def leader(pid):
        nonlocal calls
        if pid == 101:
            raise runtime.psutil.NoSuchProcess(pid)
        calls += 1
        if calls == 1:
            raise runtime.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: 99.0)

    monkeypatch.setattr(runtime.psutil, "Process", leader)
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert proof["unconfirmed_sessions"][0]["reason"] == "session ownership changed during cleanup"


@pytest.mark.parametrize("failure", ["exit", "reuse"])
def test_watcher_preserves_enumerated_identity_on_capture_or_registration_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    child = SimpleNamespace(pid=202, create_time=lambda: 2.0)
    parent = SimpleNamespace(children=lambda recursive: [child])
    monkeypatch.setattr(runtime, "process_state", lambda target: ("live", parent))
    original = {"pid": 202, "create_time": 2.0, "sid": 202, "pgid": 202,
                "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                "process_scope": SCOPE, "assigned_gpu_uuids": ["GPU-a"]}
    captures = 0

    def capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        if failure == "reuse":
            return {**original, "create_time": 99.0, "sid": 999}
        if captures == 1:
            return dict(original)
        raise runtime.psutil.NoSuchProcess(202)

    monkeypatch.setattr(runtime, "process_identity", capture)
    monkeypatch.setattr(runtime.process_scope, "get_process_scope", lambda: SCOPE)

    class OnePass:
        def __init__(self): self.calls = 0
        def wait(self, interval):
            self.calls += 1
            return self.calls > 1

    class ImmediateThread:
        def __init__(self, *, target, **kwargs): self.target = target
        def start(self): self.target()

    monkeypatch.setattr(runtime, "threading", SimpleNamespace(Event=OnePass, Thread=ImmediateThread))
    runtime.start_descendant_registration(tmp_path, run_id="run", allocation_id="allocation",
                                         execution_owner=OWNER, supervisor_pid=101,
                                         assigned_gpu_uuids=["GPU-a"])
    root = runtime.execution_root(tmp_path, "allocation")
    errors = [json.loads(path.read_text()) for path in (root / "registration-errors").glob("*.json")]
    assert len(errors) == 1
    target = errors[0]["target"]
    assert target["pid"] == 202 and target["create_time"] == 2.0
    assert target["assigned_gpu_uuids"] == ["GPU-a"]
    assert not list((root / "processes").glob("*.json"))
    if failure == "exit":
        assert target == original and captures == 2
    else:
        assert "sid" not in target and captures == 1


@pytest.mark.parametrize("supplement", [False, True])
def test_registered_process_missing_sid_needs_same_identity_evidence(registry, monkeypatch, supplement):
    training_root, root = registry
    target = {"pid": 202, "create_time": 2.0, "sid": None, "pgid": 202,
              "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
              "process_scope": SCOPE, "assigned_gpu_uuids": ["GPU-a"]}
    (root / "processes" / "descendant-202.json").write_text(json.dumps(target))
    if supplement:
        write_error(root, "leader", target={**target, "sid": 202})
    session_fakes(monkeypatch)
    proof = cleanup(training_root)
    assert proof["complete"] == supplement
    if not supplement:
        assert any(item["target"]["pid"] == 202 and item["reason"] == "original session identity is unavailable"
                   for item in proof["unconfirmed_sessions"])


@pytest.mark.parametrize("live", [False, True])
def test_error_missing_sid_supplemented_by_nonleader_still_requires_leader(registry, monkeypatch, live):
    training_root, root = registry
    original = {"pid": 202, "create_time": 2.0, "sid": None, "pgid": 202,
                "run_id": "run", "allocation_id": "allocation", "execution_owner": OWNER,
                "process_scope": SCOPE, "assigned_gpu_uuids": ["GPU-a"]}
    (root / "processes" / "descendant-202.json").write_text(json.dumps({**original, "sid": 404}))
    write_error(root, "target", target=original, resolved=not live)
    session_fakes(monkeypatch)
    if live:
        process = FakeProcess(202)
        monkeypatch.setattr(runtime, "process_state", lambda identity:
                            ("live", process) if identity["pid"] == 202 and process.alive else ("dead", None))
    proof = cleanup(training_root)
    assert not proof["complete"]
    assert not proof["survivors"]
    assert any(item["sid"] == 404 and item["reason"] == "no confirmed original session leader"
               for item in proof["unconfirmed_sessions"])
    stored = json.loads((root / "registration-errors" / "target.json").read_text())
    assert stored["target"] == original
