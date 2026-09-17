from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3  # noqa: F401
from train_platform.domains.training.runs import service
from train_platform.models.v3 import V3Base
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import DatasetType, TaskType, TrainingRunStatus
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import TrainingRun, TrainingRunParameters
from train_platform.schemas.v3.training_runs import TrainingRunCreate, TrainingRunOut


GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def resource_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'resources.db'}")
    V3Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        dataset = StandardDataset(name="resources", dataset_type=DatasetType.DETECTION,
                                  storage_path=str(tmp_path))
        db.add(dataset)
        db.flush()
        db.add(Project(project_id=1, name="resources", standard_dataset_id=dataset.standard_dataset_id,
                       task_type=TaskType.DETECTION))
        db.add(ModelArchitecture(architecture_id=1, family="YOLO", variant="test",
                                 task_type=TaskType.DETECTION, engine="ultralytics-yolo"))
        db.commit()
    monkeypatch.setattr(service, "assert_valid_license", lambda: None)
    monkeypatch.setattr(service, "resolve_legacy_dataset_path", lambda _: tmp_path)
    yield factory
    engine.dispose()


def payload(request=None, **params):
    data = {"project_id": 1, "architecture_id": 1, "parameters": {"device": "auto", **params}}
    if request is not None:
        data["resource_request"] = request
    return data


def test_request_persisted_and_read_without_allocation(resource_db):
    from train_platform.models.v3.gpu_resource import TrainingRunResourceRequest

    request = {"selection": "manual", "gpu_count": 2,
               "gpu_uuids": [GPU_A, GPU_A, GPU_B], "memory_mib_per_gpu": 18432}
    obj = TrainingRunCreate.model_validate(payload(request)).model_dump()
    with resource_db() as db:
        svc = service.TrainingRunService()
        run = svc.create_run(db, obj=obj)
        assert run.resource_request.gpu_uuids == [GPU_A, GPU_B]
        assert run.resource_request.sharing == "exclusive"
        assert run.parameters.device == "auto"
        assert db.query(TrainingRunResourceRequest).count() == 1
        run_id = run.run_id
    with resource_db() as db:
        svc = service.TrainingRunService()
        result = TrainingRunOut.model_validate(svc.get_run(db, run_id))
        assert result.resource_request.memory_mib_per_gpu == 18432
        items, total = svc.list_runs_page(db)
        assert total == 1 and items[0].resource_request.gpu_count == 2
        svc.queue_run(db, run_id)
        assert svc.get_run(db, run_id).status == TrainingRunStatus.QUEUED
        run = svc.get_run(db, run_id)
        run.status = TrainingRunStatus.FAILED
        db.commit()
        resumed = svc.resume_run(db, run_id)
        assert resumed.status == TrainingRunStatus.QUEUED
        assert resumed.resource_request.gpu_uuids == [GPU_A, GPU_B]
        assert resumed.parameters.device == "auto"


def test_legacy_create_has_no_request(resource_db):
    from train_platform.models.v3.gpu_resource import TrainingRunResourceRequest

    with resource_db() as db:
        run = service.TrainingRunService().create_run(
            db, obj=TrainingRunCreate.model_validate(payload(device="cpu")).model_dump())
        assert run.resource_request is None
        assert run.parameters.device == "cpu"
        assert db.query(TrainingRunResourceRequest).count() == 0


@pytest.mark.parametrize("resource_request,params", [
    ({"gpu_count": 2}, {"batch_size": 15}),
    ({"gpu_count": 2}, {"batch_size": -1}),
    ({"sharing": "shared", "memory_mib_per_gpu": 18432}, {"batch_size": -1}),
    ({"sharing": "shared"}, {}),
    ({"sharing": "shared", "gpu_count": 2, "memory_mib_per_gpu": 18432}, {}),
    ({"gpu_uuids": [GPU_A]}, {}),
    ({"selection": "manual", "gpu_count": 2, "gpu_uuids": [GPU_A, GPU_A]}, {}),
    ({"selection": "manual", "gpu_uuids": ["GPU-abcd"]}, {}),
    ({"selection": "manual", "gpu_uuids": ["0"]}, {}),
    ({"gpu_count": True}, {}),
    ({"memory_mib_per_gpu": 0}, {}),
    ({"memory_mib_per_gpu": 1.5}, {}),
    ({}, {"device": "cpu"}),
    ({}, {"device": "0,1"}),
    ({"sharing": "shared", "memory_mib_per_gpu": 1}, {"batch_size": True}),
])
def test_invalid_resource_request_rejected(resource_db, resource_request, params):
    with resource_db() as db:
        with pytest.raises((ValueError, service.ValidationError)):
            obj = TrainingRunCreate.model_validate(payload(resource_request, **params)).model_dump()
            service.TrainingRunService().create_run(db, obj=obj)
        assert db.query(TrainingRun).count() == 0


def test_paddle_multigpu_request_rejected(resource_db):
    with resource_db() as db:
        db.get(ModelArchitecture, 1).engine = "paddle-det"
        db.commit()
        with pytest.raises(service.ValidationError, match="[Mm]ulti-GPU"):
            service.TrainingRunService().create_run(db, obj=payload({"gpu_count": 2}, batch_size=16))


@pytest.mark.parametrize("resource_request", [[], "invalid", 1, True])
def test_malformed_resource_object_is_validation_error(resource_request):
    with pytest.raises(ValueError):
        TrainingRunCreate.model_validate(payload(resource_request))


def test_manual_uuid_case_deduplicates_and_blank_node_is_null(resource_db):
    with resource_db() as db:
        run = service.TrainingRunService().create_run(db, obj=payload({
            "selection": "manual", "gpu_count": 1,
            "gpu_uuids": [GPU_A, "GPU-" + GPU_A[4:].upper()], "node_id": "   ",
        }))
        assert run.resource_request.gpu_uuids == [GPU_A]
        assert run.resource_request.node_id is None


def test_worker_excludes_resource_requests_before_candidate_limit(resource_db, monkeypatch):
    from train_platform.models.v3.gpu_resource import TrainingRunResourceRequest
    from train_platform.workers import worker_impl as worker

    now = datetime.now(timezone.utc)
    with resource_db() as db:
        for i in range(51):
            run = TrainingRun(run_id=f"queued-{i}", project_id=1, standard_dataset_id=1,
                              architecture_id=1, name=f"queued-{i}", status=TrainingRunStatus.QUEUED,
                              queued_at=now + timedelta(seconds=i))
            run.parameters = TrainingRunParameters(device="auto")
            if i < 50:
                run.resource_request = TrainingRunResourceRequest(
                    selection="auto", gpu_count=1, gpu_uuids=[], sharing="exclusive")
            db.add(run)
        db.commit()
    instance = worker.DbQueueWorker(worker_id="test")
    monkeypatch.setattr(worker, "SessionLocal", resource_db)
    monkeypatch.setattr(instance, "_reconcile_stale_claims", lambda db: None)
    candidates = []

    def observe_candidate(device, visible):
        candidates.append(device)
        return False

    monkeypatch.setattr(worker, "worker_can_run_device", observe_candidate)
    instance._try_start_next_run()
    assert candidates == ["auto"]


def test_resource_get_endpoints_are_inventory_only(resource_db, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from train_platform.api.v3.gpu_resources import router
    from train_platform.db.session import get_db

    with resource_db() as db:
        svc = service.TrainingRunService()
        new_run = svc.create_run(db, obj=payload({"sharing": "shared", "memory_mib_per_gpu": 18432}, batch_size=16))
        legacy_run = svc.create_run(db, obj=payload(device="0,1"))
        new_id, legacy_id = new_run.run_id, legacy_run.run_id

    app = FastAPI()
    app.include_router(router, prefix="/api/v3")

    def override_db():
        with resource_db() as db:
            monkeypatch.setattr(db, "commit", lambda: pytest.fail("GET committed"))
            yield db
            assert not db.new and not db.dirty and not db.deleted

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        for run_id, legacy in [(new_id, False), (legacy_id, True)]:
            response = client.get(f"/api/v3/training-runs/{run_id}/resources")
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["scheduler_stage"] == "inventory_only"
            assert data["allocation_enabled"] is False
            assert data["allocation"] is None
            assert data["legacy_device_mode"] is legacy
            if legacy:
                assert data["resource_request"] is None
            else:
                assert data["reason_code"] == "scheduler_disabled"
                assert data["resource_request"]["memory_mib_per_gpu"] == 18432
                assert data["resource_request"]["run_id"] == run_id
                assert data["resource_request"]["created_at"]
        for path in ["gpu-resources", "gpu-workers"]:
            response = client.get(f"/api/v3/{path}")
            assert response.status_code == 200, response.text
            assert response.json()["scheduler_stage"] == "inventory_only"
            assert response.json()["allocation_enabled"] is False
            assert "reserved_mib" not in response.text


@pytest.mark.parametrize(('target', 'allocation_node', 'state', 'reason', 'enabled', 'expected_reason'), [
    (None, None, None, None, True, None),
    ('disabled', None, None, None, False, 'scheduler_disabled'),
    ('disabled', 'enabled', 'running', None, True, None),
    ('enabled', 'disabled', 'running', None, False, 'scheduler_disabled'),
    ('disabled', None, None, 'gpu_exclusive_busy', False, 'gpu_exclusive_busy'),
    ('disabled', 'disabled', 'releasing', 'gpu_exclusive_busy', False, 'cleanup_pending'),
])
def test_resource_status_node_and_reason_precedence(
    resource_db, target, allocation_node, state, reason, enabled, expected_reason,
):
    from train_platform.api.v3.gpu_resources import get_training_run_resources
    from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuNodeSchedulingState

    with resource_db() as db:
        run = service.TrainingRunService().create_run(db, obj=payload({'node_id': target}))
        run.hidden = True  # Resource status historically includes hidden runs.
        run.resource_wait_reason = reason
        run.resource_wait_details = {'message': 'retained'}
        db.add_all([
            GpuNodeSchedulingState(node_id='enabled', managed=True, accepting_allocations=True),
            GpuNodeSchedulingState(node_id='disabled', managed=True, accepting_allocations=False),
        ])
        if allocation_node:
            now = datetime.now(timezone.utc)
            db.add(GpuAllocation(allocation_id='current', run_id=run.run_id,
                worker_instance_id='instance', worker_id='worker', node_id=allocation_node,
                state=state, request_snapshot={'selection': 'auto'},
                reserved_at=now, launch_deadline_at=now))
            run.current_allocation_id = 'current'
        db.commit()
        response = get_training_run_resources(run.run_id, db).model_dump()
        assert response['allocation_enabled'] is enabled
        assert response['scheduler_stage'] == 'managed_allocation'
        assert response['reason_code'] == expected_reason
        assert response['reason_details'] == {'message': 'retained'}
        assert response['resource_request']['run_id'] == run.run_id
        assert len(response['allocation_history']) == bool(allocation_node)
        if allocation_node:
            assert response['allocation']['node_id'] == allocation_node
            assert response['effective_request'] == {'selection': 'auto'}
        assert not db.new and not db.dirty and not db.deleted


def test_resource_status_missing_run_preserves_error(resource_db):
    from train_platform.domains.training.resources.queries import get_training_run_resource_status
    from train_platform.utils.exceptions import NotFoundError

    with resource_db() as db, pytest.raises(NotFoundError, match='^Training run not found$'):
        get_training_run_resource_status(db, 'missing')
