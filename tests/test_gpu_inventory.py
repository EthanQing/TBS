from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3  # noqa: F401
from train_platform.domains.training.resources import inventory, queries
from train_platform.models.v3 import V3Base
from train_platform.models.v3.gpu_resource import GpuDevice, GpuWorkerInstance, GpuWorkerObservation
from train_platform.platform.runtime.gpu_probe import GpuProbeDevice, GpuProbeResult


GPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(queries, "datetime", FrozenDateTime)
    monkeypatch.setattr(inventory, "utcnow", lambda: NOW)
    engine = create_engine(f"sqlite:///{tmp_path / 'inventory.db'}")
    V3Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session
    engine.dispose()


def register(db, instance="instance-a", node="node-a", **kwargs):
    return inventory.register_worker_instance(
        db, instance_id=instance, worker_id="same-worker", node_id=node,
        hostname="container-hostname", process_scope={"boot_id": instance},
        allowed_engines=["ultralytics-yolo"], nvidia_visible_devices="all",
        cuda_visible_devices=None, started_at=NOW, **kwargs,
    )


def test_managed_node_latches_on_registration_without_queued_tasks(db):
    from train_platform.models.v3.gpu_allocation import GpuNodeSchedulingState

    policy = dict(shared_execution_enabled=True, max_shared_tasks_per_device=2, memory_safety_mib=4096)
    register(db, scheduling_policy=policy)
    db.commit()
    register(db, "instance-b", scheduling_policy={**policy, "memory_safety_mib": 0})
    register(db, "instance-c", scheduling_policy=None, accepting_tasks=False)
    db.commit()
    node = db.get(GpuNodeSchedulingState, "node-a")
    assert node.managed and node.shared_execution_enabled
    assert node.memory_safety_mib == 4096


def sample(*, at=NOW, used=100, total=1000, free=900):
    return GpuProbeResult("success", "nvml", at, [GpuProbeDevice(
        gpu_uuid=GPU, name="GPU", observed_index=0, memory_total_mib=total,
        memory_used_mib=used, memory_free_mib=free, utilization_percent=10,
        compute_mode="Default")], complete=True)


def test_same_gpu_multiple_workers_keeps_one_device_and_separate_samples(db):
    register(db)
    register(db, "instance-b")
    inventory.save_inventory(db, "instance-a", sample(at=NOW - timedelta(seconds=1), used=100))
    inventory.save_inventory(db, "instance-b", sample(used=200, free=800))
    db.commit()
    assert db.query(GpuDevice).count() == 1
    assert db.query(GpuWorkerObservation).count() == 2
    summary, = queries.list_gpu_resources(db)
    assert summary["memory_total_mib"] == 1000
    assert summary["memory_used_mib"] == 200
    assert len(summary["workers"]) == 2
    inventory.save_inventory(db, "instance-a", GpuProbeResult("empty", "nvml", NOW, complete=True))
    db.commit()
    db.expire_all()
    observations = {o.instance_id: o for o in db.query(GpuWorkerObservation)}
    assert observations["instance-a"].present is False
    assert observations["instance-b"].present is True


def test_failed_probe_updates_heartbeat_without_refreshing_memory(db, monkeypatch):
    register(db)
    old = NOW - timedelta(minutes=1)
    inventory.save_inventory(db, "instance-a", sample(at=old))
    db.commit()
    monkeypatch.setattr(inventory, "utcnow", lambda: NOW)
    inventory.save_inventory(db, "instance-a", GpuProbeResult("failed", "nvidia-smi", NOW, error="driver error"))
    db.commit()
    obs = db.query(GpuWorkerObservation).one()
    worker = db.get(GpuWorkerInstance, "instance-a")
    assert obs.present and obs.memory_used_mib == 100
    assert obs.sampled_at.replace(tzinfo=timezone.utc) == old
    assert worker.last_successful_inventory_at.replace(tzinfo=timezone.utc) == old
    assert worker.heartbeat_at.replace(tzinfo=timezone.utc) == NOW
    assert worker.inventory_status == "failed" and worker.inventory_error == "driver error"
    summary, = queries.list_gpu_resources(db)
    assert summary["freshness_status"] == "stale"
    assert summary["memory_free_mib"] is None


def test_incomplete_inventory_does_not_mark_existing_device_absent(db):
    register(db)
    inventory.save_inventory(db, "instance-a", sample())
    db.commit()
    inventory.save_inventory(db, "instance-a", GpuProbeResult("success", "nvml", NOW,
        [GpuProbeDevice(None, "Unknown", status="failed", error="missing UUID")], complete=False))
    db.commit()
    assert db.query(GpuWorkerObservation).one().present
    assert db.query(GpuDevice).count() == 1
    assert "UUID" in db.get(GpuWorkerInstance, "instance-a").inventory_error


def test_node_conflict_does_not_override_owner(db):
    register(db)
    register(db, "instance-b", "node-b")
    inventory.save_inventory(db, "instance-a", sample())
    inventory.save_inventory(db, "instance-b", sample())
    db.commit()
    assert db.get(GpuDevice, GPU).node_id == "node-a"
    summary, = queries.list_gpu_resources(db)
    assert summary["registration_status"] == "node_id_conflict"
    assert queries.list_gpu_resources(db, gpu_uuid="absent") == []
    assert len(queries.list_gpu_resources(db, node_id="node-a", gpu_uuid=GPU)) == 1


def test_restart_identity_and_unconfigured_node(db):
    register(db, node=None)
    inventory.save_inventory(db, "instance-a", sample())
    inventory.mark_worker_stopped(db, "instance-a")
    register(db, "instance-b", None)
    db.commit()
    assert db.query(GpuWorkerInstance).count() == 2
    assert db.get(GpuDevice, GPU).node_id is None
    summary, = queries.list_gpu_resources(db)
    assert summary["registration_status"] == "node_id_unconfigured"
    assert summary["workers"][0]["status"] == "stopped"
    assert summary["memory_free_mib"] is None


def test_latest_incomplete_memory_does_not_shadow_valid_observation(db):
    register(db)
    register(db, "instance-b")
    inventory.save_inventory(db, "instance-a", sample(at=NOW - timedelta(seconds=1)))
    inventory.save_inventory(db, "instance-b", sample(total=None, used=200, free=None))
    db.commit()
    summary, = queries.list_gpu_resources(db)
    assert summary["memory_total_mib"] == 1000
    assert summary["memory_used_mib"] == 100


def test_resource_reads_do_not_write_or_probe(db, monkeypatch):
    from train_platform.platform.runtime import gpu_probe

    register(db)
    inventory.save_inventory(db, "instance-a", sample())
    db.commit()
    monkeypatch.setattr(gpu_probe, "probe_gpus", lambda: pytest.fail("query probed GPU"))
    monkeypatch.setattr(db, "commit", lambda: pytest.fail("query committed"))
    assert queries.get_gpu_resources_overview(db)["items"]
    workers = queries.get_gpu_workers_overview(db)["items"]
    assert workers[0]["observations"][0]["gpu_uuid"] == GPU
    assert not db.dirty and not db.new and not db.deleted


def test_overviews_preserve_managed_flags_and_filters(db):
    register(db, scheduling_policy=dict(shared_execution_enabled=True,
        max_shared_tasks_per_device=2, memory_safety_mib=4096))
    inventory.save_inventory(db, "instance-a", sample())
    db.commit()
    resources = queries.get_gpu_resources_overview(db, node_id="node-a", gpu_uuid=GPU)
    assert resources["scheduler_stage"] == "managed_allocation"
    assert resources["allocation_enabled"] and resources["shared_execution_enabled"]
    assert [item["gpu_uuid"] for item in resources["items"]] == [GPU]
    assert queries.get_gpu_resources_overview(db, node_id="other") == {
        "scheduler_stage": "inventory_only", "allocation_enabled": False,
        "shared_execution_enabled": False, "items": [],
    }
    assert queries.get_gpu_workers_overview(db)["allocation_enabled"]
    inventory.mark_worker_stopped(db, "instance-a")
    db.commit()
    assert not queries.get_gpu_workers_overview(db)["allocation_enabled"]


def test_running_count_is_instance_scoped_and_caller_owns_transaction(db, monkeypatch):
    first = register(db)
    second = register(db, "instance-b")
    db.commit()
    heartbeat = first.heartbeat_at
    monkeypatch.setattr(db, "commit", lambda: pytest.fail("domain committed"))
    assert inventory.update_worker_running_task_count(db, "instance-a", 3)
    assert first.running_task_count == 3 and second.running_task_count == 0
    assert first.heartbeat_at == heartbeat and first.stopped_at is None
    assert first.accepting_tasks and first.max_training_slots == 2
    assert not inventory.update_worker_running_task_count(db, "missing", 9)
    db.rollback()
    assert first.running_task_count == 0


def test_ownership_snapshot_filters_and_detaches_nested_json(db):
    from train_platform.models.v3.gpu_allocation import GpuAllocation

    owner = {"process_scope": {"pid_namespace": {"inode": 2}}, "guard_pid": 1}
    for allocation_id, node_id, state, execution_owner in [
        ("active", "node-a", "releasing", owner),
        ("unknown", "node-a", "unknown", owner),
        ("released", "node-a", "released", owner),
        ("other", "node-b", "running", owner),
        ("unowned", "node-a", "starting", None),
    ]:
        db.add(GpuAllocation(allocation_id=allocation_id, run_id="run",
            worker_instance_id="instance-a", worker_id="worker", node_id=node_id,
            state=state, execution_owner=execution_owner, request_snapshot={},
            reserved_at=NOW, launch_deadline_at=NOW))
    db.commit()
    allocation = db.get(GpuAllocation, "active")
    snapshot = queries.get_active_allocation_ownership_snapshot(db, node_id="node-a")
    assert {item["allocation_id"] for item in snapshot} == {"active", "unknown"}
    active = next(item for item in snapshot if item["allocation_id"] == "active")
    active["execution_owner"]["process_scope"]["pid_namespace"]["inode"] = 99
    assert allocation.execution_owner["process_scope"]["pid_namespace"]["inode"] == 2
    db.expunge_all()
    assert active["run_id"] == "run"


def test_worker_response_preserves_unavailable_process_scope(db):
    from train_platform.schemas.v3.gpu_resources import GpuWorkersResponse

    worker = register(db, node=None)
    worker.process_scope = None
    inventory.save_inventory(db, "instance-a", sample())
    db.commit()
    result = GpuWorkersResponse(items=queries.list_gpu_workers(db))
    assert result.items[0].process_scope is None
    assert result.items[0].node_id_unconfigured


def test_freshness_describes_observation_even_when_memory_unknown(db):
    register(db)
    inventory.save_inventory(db, "instance-a", sample(total=None, used=None, free=None))
    db.commit()
    summary, = queries.list_gpu_resources(db)
    assert summary["freshness_status"] == "fresh"
    assert summary["memory_total_mib"] is None


def test_actual_dual_probe_failure_preserves_inventory(db, monkeypatch):
    from train_platform.platform.runtime import gpu_probe

    register(db)
    old = NOW - timedelta(minutes=1)
    inventory.save_inventory(db, "instance-a", sample(at=old))
    db.commit()

    def fail_handle(index):
        raise RuntimeError("NVML handle unavailable")

    monkeypatch.setattr(gpu_probe, "pynvml", SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=fail_handle,
    ))
    monkeypatch.setattr(gpu_probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(gpu_probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="smi cannot read devices"))
    result = gpu_probe.probe_gpus()
    inventory.save_inventory(db, "instance-a", result)
    db.commit()
    db.expire_all()
    worker = db.get(GpuWorkerInstance, "instance-a")
    observation = db.query(GpuWorkerObservation).one()
    assert worker.inventory_status == "failed"
    assert "NVML handle unavailable" in worker.inventory_error
    assert "smi cannot read devices" in worker.inventory_error
    assert worker.heartbeat_at.replace(tzinfo=timezone.utc) == NOW
    assert worker.last_successful_inventory_at.replace(tzinfo=timezone.utc) == old
    assert observation.sampled_at.replace(tzinfo=timezone.utc) == old
    assert observation.present
    assert (observation.memory_total_mib, observation.memory_used_mib, observation.memory_free_mib) == (1000, 100, 900)


def test_partial_inventory_updates_good_card_without_hiding_old_card(db):
    register(db)
    old = NOW - timedelta(minutes=1)
    inventory.save_inventory(db, "instance-a", sample(at=old))
    db.commit()
    second_gpu = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    result = GpuProbeResult("success", "nvml", NOW, [
        GpuProbeDevice(None, "GPU 0", observed_index=0, status="failed", error="handle failed"),
        GpuProbeDevice(second_gpu, "Second GPU", observed_index=1, memory_total_mib=1000,
                       memory_used_mib=200, memory_free_mib=800),
        GpuProbeDevice(None, "Unidentified GPU", observed_index=2, memory_total_mib=1000,
                       status="partial", error="full GPU UUID unavailable"),
    ], error="GPU 0 handle failed", complete=False)
    inventory.save_inventory(db, "instance-a", result)
    db.commit()
    db.expire_all()
    observations = {item.gpu_uuid: item for item in db.query(GpuWorkerObservation)}
    assert set(observations) == {GPU, second_gpu}
    assert observations[GPU].present and observations[GPU].sampled_at.replace(tzinfo=timezone.utc) == old
    assert observations[second_gpu].memory_used_mib == 200
    assert observations[second_gpu].sampled_at.replace(tzinfo=timezone.utc) == NOW
    error = db.get(GpuWorkerInstance, "instance-a").inventory_error
    assert "GPU 0 handle failed" in error and "UUID" in error
