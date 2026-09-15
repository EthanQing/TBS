import json

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
