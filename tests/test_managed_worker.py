from concurrent.futures import Future
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from train_platform.workers import worker_impl as worker


@pytest.mark.parametrize("matched", [True, False])
def test_cleanup_report_delegates_and_keeps_transaction_boundary(monkeypatch, matched):
    from train_platform.domains.training.resources import lifecycle

    db = Mock()
    mark = Mock(return_value=matched)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(lifecycle, "mark_cleanup_pending", mark)
    instance = worker.DbQueueWorker.__new__(worker.DbQueueWorker)
    owner = {"guard_pid": 123}
    instance._record_cleanup_pending(dict(run_id="run", allocation_id="allocation", owner=owner),
                                     "unconfirmed", "unknown process")
    mark.assert_called_once_with(db, run_id="run", allocation_id="allocation",
        execution_owner=owner, reason="unconfirmed", error="unknown process")
    assert db.commit.call_count == int(matched)
    assert db.rollback.call_count == int(not matched)
    db.close.assert_called_once()
    db.query.assert_not_called()


@pytest.mark.parametrize("exists", [True, False])
def test_running_count_delegates_observed_jobs_by_instance(monkeypatch, exists):
    from train_platform.domains.training.resources import inventory

    db = Mock()
    update = Mock(return_value=exists)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(inventory, "update_worker_running_task_count", update)
    instance = worker.DbQueueWorker.__new__(worker.DbQueueWorker)
    instance._gpu_resource_reporter = SimpleNamespace(instance_id="new-instance")
    instance._running_jobs = {"one-ddp-job": object(), "other-job": object()}
    instance._publish_running_task_count()
    update.assert_called_once_with(db, "new-instance", 2)
    assert db.commit.call_count == int(exists)
    db.close.assert_called_once()
    db.get.assert_not_called()


def test_one_job_failure_does_not_skip_other_jobs(monkeypatch):
    instance = worker.DbQueueWorker(worker_id="worker")
    instance._running_jobs = {"a": "first", "b": "second"}
    visited = []
    monkeypatch.setattr(worker, "assert_valid_license", lambda: None)
    monkeypatch.setattr(instance, "_reconcile_managed_allocations", lambda: None)
    monkeypatch.setattr(instance, "_publish_running_task_count", lambda: None)
    monkeypatch.setattr(instance, "available_training_slots", lambda: 0)

    def maintain(job):
        visited.append(job)
        if job == "first":
            raise RuntimeError("one task failed")

    monkeypatch.setattr(instance, "_tick_running", maintain)
    instance.tick()
    assert visited == ["first", "second"]
    instance._cleanup_executor.shutdown(wait=False)


def test_unmanaged_worker_keeps_single_task_limit(monkeypatch):
    instance = worker.DbQueueWorker(worker_id="worker")
    monkeypatch.setattr(instance, "_managed_scheduling_required", lambda: False)
    assert instance.available_training_slots() == 1
    instance._running_jobs["legacy"] = object()
    assert instance.has_running_jobs()
    assert instance.running_job_count() == 1
    assert instance.available_training_slots() == 0
    instance._cleanup_executor.shutdown(wait=False)


def test_spawn_freezes_only_child_uuid_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "worker-mask")
    captured = {}

    def popen(args, **kwargs):
        captured.update(args=args, **kwargs)
        return object()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    mask = ["GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]
    worker._spawn_training_subprocess("run", stdout_f=StringIO(), stderr_f=StringIO(),
        allocation_id="allocation", worker_instance_id="instance", assigned_gpu_uuids=mask)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(mask)
    assert captured["env"]["TRAIN_PLATFORM_ALLOCATION_ID"] == "allocation"
    assert "--allocation-id" in captured["args"]
    assert worker.os.environ["CUDA_VISIBLE_DEVICES"] == "worker-mask"


def test_scheduler_does_not_fall_back_when_reporter_is_missing(monkeypatch):
    instance = worker.DbQueueWorker(worker_id="worker")
    monkeypatch.setattr(instance, "_managed_scheduling_required", lambda: True)
    monkeypatch.setattr(worker, "settings", SimpleNamespace(gpu_scheduler_enabled=True))
    monkeypatch.setattr(worker, "SessionLocal", lambda: (_ for _ in ()).throw(AssertionError("legacy path reached")))
    assert instance._try_start_next_run() is False
    instance._cleanup_executor.shutdown(wait=False)


def test_spawned_child_retains_job_when_identity_read_fails(monkeypatch, tmp_path):
    from train_platform.domains.training.resources import allocator, lifecycle

    instance = worker.DbQueueWorker(worker_id="worker")
    instance._gpu_resource_reporter = SimpleNamespace(instance_id="instance")
    allocation = SimpleNamespace(allocation_id="allocation", run_id="run", worker_instance_id="instance",
        devices=[SimpleNamespace(gpu_uuid="GPU-a", ordinal=0)], request_snapshot={"sharing": "exclusive"},
        launcher_identity={})
    db = SimpleNamespace(get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="sqlite")),
        get=lambda *a: SimpleNamespace(architecture=SimpleNamespace(engine="ultralytics-yolo")),
        commit=lambda: None, rollback=lambda: None, close=lambda: None)
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(instance, "_adopt_proven_legacy_executions", lambda *_: None)
    monkeypatch.setattr(allocator, "reserve_next", lambda *a, **kw: SimpleNamespace(allocation=allocation))
    monkeypatch.setattr(lifecycle, "issue_start_authorization", lambda *a: None)
    monkeypatch.setattr(lifecycle, "revoke_unactivated_allocation",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("premature revoke")))
    monkeypatch.setattr(worker, "settings", SimpleNamespace(gpu_scheduler_enabled=True,
        gpu_shared_execution_enabled=False, gpu_max_shared_tasks_per_device=2,
        gpu_memory_safety_mib=4096, gpu_inventory_stale_after_seconds=20,
        gpu_allocation_start_timeout_seconds=30, training_dir=tmp_path))
    proc = SimpleNamespace(pid=123)
    monkeypatch.setattr(worker, "_spawn_training_subprocess", lambda *a, **kw: proc)
    monkeypatch.setattr(worker.psutil, "Process", lambda *_: (_ for _ in ()).throw(worker.psutil.NoSuchProcess(123)))
    assert instance._try_start_managed()
    job = instance._running_jobs["allocation"]
    assert job.proc is proc and not job.stdout_f.closed
    instance._cleanup_running(job)
    instance._cleanup_executor.shutdown(wait=False)
