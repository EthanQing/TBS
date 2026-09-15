from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3
from train_platform.models.v3 import V3Base
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TaskType, TrainingRunStatus
from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuCudaBinding
from train_platform.models.v3.gpu_resource import GpuDevice, GpuWorkerObservation, TrainingRunResourceRequest
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEvent, TrainingRunParameters
from train_platform.domains.training.resources import allocator, inventory


GPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SCOPE = {"boot_id": "boot", "pid_namespace": {"device": 1, "inode": 2}}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'allocator.db'}")
    V3Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        now = datetime.now(timezone.utc)
        session.add(ModelArchitecture(architecture_id=1, family="YOLO", variant="n",
                                     task_type=TaskType.DETECTION, engine="ultralytics-yolo"))
        worker = inventory.register_worker_instance(
            session, instance_id="instance", worker_id="worker", node_id="node",
            hostname="host", process_scope=SCOPE, allowed_engines=["ultralytics-yolo"],
            nvidia_visible_devices="all", cuda_visible_devices=None,
            launcher_identity={"pid": 1, "create_time": 1.0, "process_scope": SCOPE},
        )
        worker.cuda_inventory_status = "success"
        worker.cuda_environment_fingerprint = "fingerprint"
        worker.last_successful_cuda_inventory_at = now
        session.add(GpuDevice(gpu_uuid=GPU, name="test", node_id="node"))
        session.add(GpuCudaBinding(instance_id="instance", gpu_uuid=GPU, ordinal=0,
                                   verified_at=now, environment_fingerprint="fingerprint", status="success"))
        session.add(GpuWorkerObservation(instance_id="instance", gpu_uuid=GPU,
            memory_total_mib=49152, memory_used_mib=2048, memory_free_mib=47104,
            compute_mode="default", mig_mode="disabled", probe_source="nvml",
            probe_status="success", present=True, sampled_at=now, received_at=now))
        session.commit()
        yield session
    engine.dispose()


def enqueue(db, run_id, *, sharing="shared", count=1, budget=18432):
    run = TrainingRun(run_id=run_id, project_id=1, standard_dataset_id=1, architecture_id=1,
                      name=run_id, status=TrainingRunStatus.QUEUED, queued_at=datetime.now(timezone.utc))
    run.parameters = TrainingRunParameters(device="auto", batch_size=16)
    run.resource_request = TrainingRunResourceRequest(selection="auto", gpu_count=count,
        gpu_uuids=[], memory_mib_per_gpu=budget, sharing=sharing)
    db.add(run)
    db.commit()
    return run


def reserve(db, run_id):
    return allocator.allocate_run(db, run_id, "instance", scheduler_enabled=True,
        shared_execution_enabled=True, stale_after_seconds=20,
        node_defaults={"max_shared_tasks_per_device": 2, "memory_safety_mib": 4096})


def test_two_shared_reservations_account_before_either_starts(db):
    enqueue(db, "a")
    enqueue(db, "b")
    assert reserve(db, "a").allocation
    db.commit()
    assert reserve(db, "b").allocation
    db.commit()
    assert db.query(GpuAllocation).count() == 2
    enqueue(db, "c")
    assert reserve(db, "c").reason_code == "worker_slot_limit"


def test_exclusive_blocks_shared_even_with_small_requested_budget(db):
    enqueue(db, "a", sharing="exclusive", budget=1)
    enqueue(db, "b")
    assert reserve(db, "a").allocation
    db.commit()
    assert reserve(db, "b").reason_code == "gpu_exclusive_busy"


def test_multicard_request_never_partially_reserves(db):
    enqueue(db, "a", sharing="exclusive", count=2)
    assert reserve(db, "a").allocation is None
    db.commit()
    assert db.query(GpuAllocation).count() == 0


def test_releasing_still_blocks_allocation(db):
    enqueue(db, "a", sharing="exclusive")
    enqueue(db, "b")
    allocation = reserve(db, "a").allocation
    allocation.state = "releasing"
    db.commit()
    assert reserve(db, "b").reason_code == "gpu_exclusive_busy"


def test_revoked_start_cannot_be_activated_late(db):
    from train_platform.domains.training.resources import lifecycle

    enqueue(db, "a")
    allocation = reserve(db, "a").allocation
    allocation_id = allocation.allocation_id
    db.commit()
    assert lifecycle.revoke_unactivated_allocation(db, allocation_id, run_id="a",
        worker_instance_id="instance", reason="start failed")
    db.commit()
    with pytest.raises(ValueError):
        lifecycle.activate_allocation(db, allocation_id, run_id="a", worker_instance_id="instance",
            process_scope=SCOPE, supervisor_pid=202, supervisor_create_time=123.0,
            assigned_gpu_uuids=[GPU])
    db.rollback()
    assert db.get(GpuAllocation, allocation_id).state == "released"


def test_activation_is_one_use_and_release_requires_cleanup(db):
    from train_platform.domains.training.resources import lifecycle

    enqueue(db, "a")
    allocation_id = reserve(db, "a").allocation.allocation_id
    db.commit()
    result = lifecycle.activate_allocation(db, allocation_id, run_id="a", worker_instance_id="instance",
        process_scope=SCOPE, supervisor_pid=202, supervisor_create_time=123.0, assigned_gpu_uuids=[GPU])
    db.commit()
    with pytest.raises(ValueError):
        lifecycle.activate_allocation(db, allocation_id, run_id="a", worker_instance_id="instance",
            process_scope=SCOPE, supervisor_pid=203, supervisor_create_time=124.0, assigned_gpu_uuids=[GPU])
    db.rollback()
    owner = result["execution_owner"]
    lifecycle.record_execution_result(db, allocation_id, owner, 0, None)
    lifecycle.request_releasing(db, allocation_id, owner)
    db.commit()
    with pytest.raises(ValueError):
        lifecycle.finish_allocation(db, allocation_id, owner, {"allocation_id": allocation_id, "complete": False})
    db.rollback()
    assert db.get(GpuAllocation, allocation_id).state == "releasing"
    assert db.get(TrainingRun, "a").status == TrainingRunStatus.RUNNING


def test_unchanged_queue_reason_does_not_repeat_events(db):
    from train_platform.models.v3.training_run import TrainingRunEvent

    enqueue(db, "a", sharing="exclusive")
    enqueue(db, "b")
    reserve(db, "a")
    db.commit()
    for _ in range(3):
        assert reserve(db, "b").reason_code == "gpu_exclusive_busy"
        db.commit()
    assert db.query(TrainingRunEvent).filter_by(run_id="b", event_type="resource_wait").count() == 1


def test_scan_cursor_advances_past_more_than_one_page_of_resource_blocked_runs(db):
    cursor = allocator.AllocationScanCursor()
    for index in range(51):
        run = enqueue(db, f"blocked-{index:02d}")
        run.resource_request.selection = "manual"
        run.resource_request.gpu_uuids = [f"GPU-missing-{index:02d}"]
        db.commit()
    enqueue(db, "runnable")

    decision = None
    for _ in range(52):
        decision = allocator.reserve_next(
            db, "instance", scan_cursor=cursor, scheduler_enabled=True,
            shared_execution_enabled=True, stale_after_seconds=20,
            node_defaults={"max_shared_tasks_per_device": 2, "memory_safety_mib": 4096},
        )
        db.commit()
        if decision.allocation:
            break

    assert decision is not None and decision.allocation is not None
    assert decision.allocation.run_id == "runnable"


def test_candidate_query_excludes_foreign_engines_and_nodes_before_attempt(db, monkeypatch):
    db.add(ModelArchitecture(architecture_id=2, family="Other", variant="x",
                             task_type=TaskType.DETECTION, engine="other-engine"))
    db.commit()
    for index in range(51):
        run = enqueue(db, f"engine-{index:02d}")
        run.architecture_id = 2
        db.commit()
    for index in range(51):
        run = enqueue(db, f"node-{index:02d}")
        run.resource_request.node_id = "other-node"
        db.commit()
    enqueue(db, "query-runnable")
    original = allocator.allocate_run
    attempted = []
    monkeypatch.setattr(allocator, "allocate_run",
                        lambda session, run_id, worker_id, **kw:
                        attempted.append(run_id) or original(session, run_id, worker_id, **kw))
    cursor = allocator.AllocationScanCursor()
    decision = allocator.reserve_next(
        db, "instance", scan_cursor=cursor, scheduler_enabled=True,
        shared_execution_enabled=True, stale_after_seconds=20,
        node_defaults={"max_shared_tasks_per_device": 2, "memory_safety_mib": 4096})
    assert decision.allocation and attempted == ["query-runnable"]
    assert all(db.get(TrainingRun, f"engine-{i:02d}").resource_wait_reason is None for i in range(51))
    assert all(db.get(TrainingRun, f"node-{i:02d}").resource_wait_reason is None for i in range(51))
    db.commit()
    assert allocator.reserve_next(
        db, "instance", scan_cursor=cursor, scheduler_enabled=True,
        shared_execution_enabled=True, stale_after_seconds=20,
        node_defaults={"max_shared_tasks_per_device": 2, "memory_safety_mib": 4096}).reason_code == "scan_exhausted"
    db.get(TrainingRun, "node-00").resource_request.node_id = "node"
    db.commit()
    wrapped = allocator.reserve_next(
        db, "instance", scan_cursor=cursor, scheduler_enabled=True,
        shared_execution_enabled=True, stale_after_seconds=20,
        node_defaults={"max_shared_tasks_per_device": 2, "memory_safety_mib": 4096})
    assert wrapped.allocation and wrapped.allocation.run_id == "node-00"


def test_cancel_unactivated_allocation_revokes_and_preserves_intent(db):
    from train_platform.domains.training.runs.lifecycle import request_cancel

    enqueue(db, "a")
    allocation_id = reserve(db, "a").allocation.allocation_id
    db.commit()
    db.autoflush = False
    run = request_cancel(db, "a")
    assert run.status == TrainingRunStatus.CANCELLED
    assert run.cancel_requested_at is not None
    assert run.current_allocation_id is None
    assert db.get(GpuAllocation, allocation_id).authorization_state == "revoked"


def test_outer_worker_retains_budget_until_cleanup_proof(db, monkeypatch, tmp_path):
    from io import StringIO
    from types import SimpleNamespace
    from train_platform.domains.training.resources import lifecycle
    from train_platform.platform.runtime import execution_processes
    from train_platform.workers import worker_impl
    from train_platform.domains.training.runs import artifacts

    enqueue(db, "a")
    allocation_id = reserve(db, "a").allocation.allocation_id
    db.commit()
    owner = lifecycle.activate_allocation(db, allocation_id, run_id="a", worker_instance_id="instance",
        process_scope=SCOPE, supervisor_pid=202, supervisor_create_time=123.0,
        assigned_gpu_uuids=[GPU])["execution_owner"]
    db.commit()
    monkeypatch.setattr(worker_impl, "SessionLocal", sessionmaker(bind=db.bind, autoflush=False))
    monkeypatch.setattr(worker_impl, "settings", SimpleNamespace(training_dir=tmp_path))
    complete = False
    cleanup_calls = 0

    def cleanup(*args, **kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        with sessionmaker(bind=db.bind)() as check:
            assert check.get(GpuAllocation, allocation_id).state == "releasing"
            assert check.get(TrainingRun, "a").status == TrainingRunStatus.RUNNING
        errors = [{"id": "new-a", "checked_at": "later", "stage": "descendant_scan",
                   "error": "scan failed", "target": {"pid": 9, "create_time": 2.0, "process_scope": SCOPE}},
                  {"id": "new-b", "stage": "descendant_scan", "error": "scan failed",
                   "target": {"pid": 9, "create_time": 2.0, "process_scope": SCOPE}}]
        return {"complete": complete, "allocation_id": allocation_id,
                "execution_owner": owner, "process_scope": SCOPE, "supervisor_excluded": False,
                "registration_errors": [] if complete else list(reversed(errors))}

    monkeypatch.setattr(execution_processes, "cleanup_registered_execution", cleanup)
    worker = worker_impl.DbQueueWorker.__new__(worker_impl.DbQueueWorker)
    ddp_attempts = 0

    def cleanup_ddp(job):
        nonlocal ddp_attempts
        ddp_attempts += 1
        if ddp_attempts <= 2:
            raise RuntimeError("DDP registration cleanup failed")

    monkeypatch.setattr(worker, "_cleanup_registered_ddp", cleanup_ddp)
    job = worker_impl.RunningJob(run_id="a", engine="ultralytics-yolo", proc=SimpleNamespace(pid=202),
        stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", stdout_f=StringIO(), stderr_f=StringIO(),
        allocation_id=allocation_id, worker_instance_id="instance", guard_create_time=123.0,
        ultralytics_ddp=True)
    assert worker._finish_managed_job(job, 1)[0] is False
    assert cleanup_calls == 1
    assert worker._finish_managed_job(job, 1)[0] is False
    db.expire_all()
    assert db.get(GpuAllocation, allocation_id).active_run_id == "a"
    assert db.query(TrainingRunEvent).filter_by(run_id="a", event_type="cleanup_pending").count() == 1
    complete = True
    result = worker._finish_managed_job(job, 1)
    assert result[0] is True
    assert result[1] == TrainingRunStatus.FAILED
    db.expire_all()
    assert db.get(GpuAllocation, allocation_id).active_run_id is None
    assert db.get(TrainingRun, "a").current_allocation_id is None
    assert db.get(TrainingRun, "a").resource_wait_reason is None
    assert db.get(TrainingRun, "a").resource_wait_details is None
