import json
from types import SimpleNamespace

import pytest

from train_platform.platform.runtime import process_scope
from train_platform.platform.runtime import ultralytics_ddp as ddp
from train_platform.workers import worker_impl as worker


LOCAL_SCOPE = {
    "boot_id": "11111111-1111-4111-8111-111111111111",
    "pid_namespace": {"device": 4, "inode": 4026531836},
}
OTHER_NAMESPACE = {**LOCAL_SCOPE, "pid_namespace": {"device": 4, "inode": 4026532440}}
OTHER_BOOT = {**LOCAL_SCOPE, "boot_id": "22222222-2222-4222-8222-222222222222"}


def owner(scope=LOCAL_SCOPE):
    return {"guard_pid": 123, "guard_create_time": 456.0, "worker_id": "origin-worker", "process_scope": scope}


def no_local_process(*args, **kwargs):
    pytest.fail("An unverified scope must not query a local PID")


@pytest.mark.parametrize("recorded,current", [
    (OTHER_NAMESPACE, LOCAL_SCOPE), (OTHER_BOOT, LOCAL_SCOPE),
    (None, LOCAL_SCOPE), ({}, LOCAL_SCOPE), (LOCAL_SCOPE, None),
])
def test_foreign_or_unknown_identity_never_queries_local_pid(monkeypatch, recorded, current):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: current)
    monkeypatch.setattr(ddp.psutil, "Process", no_local_process)
    identity = {"pid": 123, "create_time": 456.0, "execution_owner": owner(recorded)}
    assert worker._identity_is_live(identity) is None
    assert ddp._matching_process(identity) is None
    assert ddp._process_state(identity) == ("unknown", None)


@pytest.mark.parametrize("state", ["gone", "reused", "zombie", "alive"])
def test_same_scope_enables_identity_and_liveness_check(monkeypatch, state):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    identity = {"pid": 123, "create_time": 456.0, "execution_owner": owner()}
    calls = []

    def process(pid):
        calls.append(pid)
        if state == "gone":
            raise ddp.psutil.NoSuchProcess(pid)
        return SimpleNamespace(
            create_time=lambda: 789.0 if state == "reused" else 456.0,
            is_running=lambda: True,
            status=lambda: ddp.psutil.STATUS_ZOMBIE if state == "zombie" else ddp.psutil.STATUS_RUNNING,
        )

    monkeypatch.setattr(ddp.psutil, "Process", process)
    assert worker._identity_is_live(identity) is (state == "alive")
    assert ddp._process_state(identity)[0] == ("live" if state == "alive" else "dead")
    assert calls == [123, 123]


@pytest.mark.parametrize("scope", [OTHER_NAMESPACE, OTHER_BOOT, None])
def test_cleanup_rejects_unqueryable_execution_even_without_attempt(tmp_path, monkeypatch, scope):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(ddp.psutil, "Process", no_local_process)
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete) as caught:
        ddp.terminate_registered_processes(tmp_path, run_id="run", owner=owner(scope))
    assert caught.value.run_id == "run"
    assert caught.value.execution_owner == owner(scope)


@pytest.mark.parametrize("source", ["context", "registration", "pending"])
def test_cleanup_checks_every_persisted_scope_before_querying_pid(tmp_path, monkeypatch, source):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(ddp.psutil, "Process", no_local_process)
    attempt = tmp_path / "runtime/ddp/attempt"
    (attempt / "processes").mkdir(parents=True)
    local = {"run_id": "run", "attempt_id": "attempt", "execution_owner": owner()}
    foreign = {**local, "execution_owner": owner(OTHER_NAMESPACE)}
    (attempt / "context.json").write_text(json.dumps(foreign if source == "context" else local))
    if source == "registration":
        (attempt / "processes/rank-0.json").write_text(json.dumps({**foreign, "pid": 987, "create_time": 654.0}))
    if source == "pending":
        (attempt / "cleanup-pending.json").write_text(json.dumps({**foreign, "survivors": []}))
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete):
        ddp.terminate_registered_processes(tmp_path, run_id="run", owner=owner())


def write_execution(root, scope):
    runtime = root / "run" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "execution.json"
    path.write_text(json.dumps({"run_id": "run", "execution_owner": owner(scope)}))
    return path


@pytest.mark.parametrize("scope", [OTHER_NAMESPACE, OTHER_BOOT, None, "missing-record"])
@pytest.mark.parametrize("worker_id", ["origin-worker", "different-worker"])
def test_stale_preparation_without_context_remains_claimed(tmp_path, monkeypatch, scope, worker_id):
    record = None
    if scope != "missing-record":
        record = write_execution(tmp_path, scope)
        original_record = record.read_bytes()
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker.psutil, "Process", no_local_process)
    monkeypatch.setattr(worker, "terminate_registered_processes", no_local_process)
    instance = worker.DbQueueWorker(worker_id=worker_id)
    run = SimpleNamespace(run_id="run", pid=123, worker_id="origin-worker")
    assert instance._cleanup_stale_ddp(run) is False
    if record is not None:
        assert record.read_bytes() == original_record


def test_same_scope_restarted_worker_can_recover_preparation_exit(tmp_path, monkeypatch):
    write_execution(tmp_path, LOCAL_SCOPE)
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: (_ for _ in ()).throw(ddp.psutil.NoSuchProcess(pid)))
    instance = worker.DbQueueWorker(worker_id="replacement-worker")
    run = SimpleNamespace(run_id="run", pid=123, worker_id="origin-worker")
    assert instance._cleanup_stale_ddp(run) is True


def test_matching_context_with_conflicting_scope_cannot_be_skipped_as_clean(tmp_path, monkeypatch):
    write_execution(tmp_path, LOCAL_SCOPE)
    attempt = tmp_path / "run/runtime/ddp/attempt"
    attempt.mkdir(parents=True)
    (attempt / "context.json").write_text(json.dumps({
        "run_id": "run", "attempt_id": "attempt", "execution_owner": owner(OTHER_NAMESPACE),
        "supervisor": {"pid": 123, "create_time": 456.0, "process_scope": OTHER_NAMESPACE},
    }))
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: (_ for _ in ()).throw(ddp.psutil.NoSuchProcess(pid)))
    monkeypatch.setattr(worker, "terminate_registered_processes", no_local_process)
    run = SimpleNamespace(run_id="run", pid=123, worker_id="origin-worker")
    assert worker.DbQueueWorker(worker_id="replacement")._cleanup_stale_ddp(run) is False


def test_reconcile_keeps_foreign_scope_out_of_terminal_lifecycle(tmp_path, monkeypatch):
    write_execution(tmp_path, OTHER_NAMESPACE)
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker.psutil, "Process", no_local_process)
    monkeypatch.setattr(worker, "finalize_execution", no_local_process)
    run = SimpleNamespace(
        run_id="run", pid=123, worker_id="origin-worker",
        architecture=SimpleNamespace(engine="ultralytics-yolo"),
        parameters=SimpleNamespace(device="0,1"),
    )
    results = iter([[], [run]])
    query = SimpleNamespace(all=lambda: next(results))
    query.filter = lambda *args: query
    db = SimpleNamespace(query=lambda *args: query)
    worker.DbQueueWorker(worker_id="different-worker")._reconcile_stale_claims(db)
    assert run.pid == 123 and run.worker_id == "origin-worker"


def test_rank_rejects_foreign_scope_before_registering(tmp_path, monkeypatch):
    from train_platform.workers.training import ultralytics_ddp_entry_impl as rank_entry

    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(ddp, "process_identity", no_local_process)
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}.items():
        monkeypatch.setenv(key, value)
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "run_id": "run", "attempt_id": "attempt", "world_size": 2,
        "cuda_visible_devices": "0,1", "execution_owner": owner(OTHER_NAMESPACE),
    }))
    with pytest.raises(RuntimeError):
        rank_entry.main(["--context", str(context)])


def test_worker_records_scope_before_publishing_started_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(training_dir=tmp_path))
    monkeypatch.setattr(worker, "worker_can_run_device", lambda *args: True)
    monkeypatch.setattr(worker, "evaluate_training_alerts_best_effort", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 456.0))
    proc = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(worker, "_spawn_training_subprocess", lambda *args, **kwargs: proc)
    run = SimpleNamespace(
        run_id="run", architecture=SimpleNamespace(engine="ultralytics-yolo"),
        parameters=SimpleNamespace(device="0,1"),
    )
    query = SimpleNamespace(all=lambda: [run])
    for method in ("join", "filter", "order_by", "with_for_update", "limit"):
        setattr(query, method, lambda *args, **kwargs: query)
    db = SimpleNamespace(query=lambda *args: query, close=lambda: None)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    published = []

    def mark_started(*args, **kwargs):
        record = json.loads((tmp_path / "run/runtime/execution.json").read_text())
        assert record["run_id"] == "run"
        assert record["execution_owner"] == owner()
        published.append(True)
        return run

    monkeypatch.setattr(worker, "mark_started", mark_started)
    instance = worker.DbQueueWorker(worker_id="origin-worker")
    monkeypatch.setattr(instance, "_reconcile_stale_claims", lambda db: None)
    try:
        instance._try_start_next_run()
        assert published == [True]
        assert instance.has_running_jobs()
    finally:
        for job in list(instance._running_jobs.values()):
            instance._cleanup_running(job)
        instance._cleanup_executor.shutdown(wait=False)


@pytest.mark.parametrize("foreign_pid", [987, 123])
def test_cleanup_handoff_does_not_relabel_foreign_survivor_as_local(tmp_path, monkeypatch, foreign_pid):
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    fake_process = SimpleNamespace(
        pid=123, create_time=lambda: 456.0, is_running=lambda: False,
        status=lambda: ddp.psutil.STATUS_ZOMBIE, children=lambda **kwargs: [],
        terminate=lambda: None, kill=lambda: None,
    )

    def query(pid):
        if pid == 987:
            pytest.fail("Foreign survivor reached the local PID query")
        return fake_process

    monkeypatch.setattr(ddp.psutil, "Process", query)
    monkeypatch.setattr(ddp.psutil, "wait_procs", lambda processes, **kwargs: (list(processes), []))
    monkeypatch.setattr(ddp.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **kwargs: 0))
    cleanup = ddp.terminate_registered_processes

    def incomplete(*args, **kwargs):
        survivor = {
            "pid": foreign_pid, "create_time": 456.0,
            "run_id": "run", "attempt_id": kwargs["attempt_id"],
            "execution_owner": owner(OTHER_NAMESPACE),
        }
        raise ddp.UltralyticsDDPCleanupIncomplete(
            "foreign survivor", run_id="run", attempt_id=kwargs["attempt_id"],
            execution_owner=owner(), survivors=[survivor],
        )

    monkeypatch.setattr(ddp, "terminate_registered_processes", incomplete)
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete):
        ddp.run_ultralytics_ddp(
            {"run_id": "run", "run_root": str(tmp_path), "world_size": 2,
             "cuda_visible_devices": "0,1", "execution_owner": owner()},
            cancel_requested=lambda: False, upsert_epoch_metrics=lambda *args: None,
        )
    pending_path = next((tmp_path / "runtime/ddp").glob("*/cleanup-pending.json"))
    pending = json.loads(pending_path.read_text())
    survivor = next(item for item in pending["survivors"] if item["pid"] == foreign_pid)
    assert process_scope.identity_process_scope(survivor) == OTHER_NAMESPACE
    assert survivor["execution_owner"] == owner(OTHER_NAMESPACE)
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete):
        cleanup(tmp_path, run_id="run", owner=owner())


@pytest.mark.parametrize("recorded_owner", [owner(OTHER_NAMESPACE), owner(None), "invalid owner", [123]])
def test_entry_keeps_unverifiable_record_in_cleanup_handoff(tmp_path, monkeypatch, recorded_owner):
    from train_platform.workers.training import train_entry_impl as entry

    record = write_execution(tmp_path, LOCAL_SCOPE)
    record.write_text(json.dumps({"run_id": "run", "execution_owner": recorded_owner}))
    monkeypatch.setattr(process_scope, "get_process_scope", lambda: LOCAL_SCOPE)
    monkeypatch.setattr(entry.psutil, "Process", no_local_process)
    with pytest.raises(ddp.UltralyticsDDPCleanupIncomplete):
        entry._execution_owner_from_record(tmp_path / "run", run_id="run", guard_pid=123, worker_id="origin-worker")
