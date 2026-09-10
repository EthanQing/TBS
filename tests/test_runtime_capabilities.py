from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.domains.deployment.runs.service import DeploymentRunService
from train_platform.domains.deployment.service import DeploymentService
from train_platform.domains.model_assets.candidates import ModelCandidateService
from train_platform.domains.model_assets.runtime import resolve_model_runtime
from train_platform.domains.training.runs.benchmarks import TrainingRunBenchmarkService
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.deployment import Deployment, DeploymentLog
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import ModelStage
from train_platform.models.v3.enums import (
    DeploymentPlatform,
    DeploymentStatus,
    TaskType,
    TrainingRunStatus,
)
from train_platform.models.v3.base import V3Base
import train_platform.models.v3  # noqa: F401 - register the complete V3 metadata
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.runtime.model_workers import ModelWorkerClient, ModelWorkerError
from train_platform.utils.exceptions import ConflictError, ValidationError


class RuntimeDb:
    def __init__(self, *, run=None, architecture=None):
        self.run = run
        self.architecture = architecture

    def query(self, entity):
        query = MagicMock()
        query.filter.return_value = query
        if entity is TrainingRun:
            query.first.return_value = self.run
        elif entity is ModelArchitecture:
            query.first.return_value = self.architecture
        else:
            query.first.return_value = None
        return query


def model_version(weights_path="weights.pt"):
    return SimpleNamespace(
        model_version_id=10,
        run_id="run-1",
        project_id=20,
        weights_path=weights_path,
        stage=ModelStage.DEVELOPMENT,
    )


def architecture(engine, *, config_path=None):
    return SimpleNamespace(
        architecture_id=30,
        engine=engine,
        family="family",
        variant="variant",
        default_params={"config_path": config_path} if config_path else {},
    )


def run(architecture_id=30):
    return SimpleNamespace(run_id="run-1", architecture_id=architecture_id)


@pytest.fixture
def sqlite_db():
    engine = create_engine("sqlite:///:memory:")
    V3Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("engine", ["custom-source", "future-engine", ""])
def test_resolve_model_runtime_rejects_unsupported_or_missing_engine(tmp_path, engine):
    weights = tmp_path / "weights.pt"
    weights.touch()
    db = RuntimeDb(run=run(), architecture=architecture(engine))
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ), pytest.raises(ConflictError):
        resolve_model_runtime(db, model_version=model_version())


def test_resolve_model_runtime_rejects_missing_architecture_without_yolo_fallback(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.touch()
    db = RuntimeDb(run=run(), architecture=None)
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ), pytest.raises(ConflictError, match="Architecture not found"):
        resolve_model_runtime(db, model_version=model_version())


def test_resolve_model_runtime_rejects_missing_source_run(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.touch()
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ), pytest.raises(ConflictError, match="Source training run not found"):
        resolve_model_runtime(RuntimeDb(), model_version=model_version())


def test_resolve_model_runtime_rejects_paddle_without_valid_config(tmp_path):
    weights = tmp_path / "weights.pdparams"
    weights.touch()
    db = RuntimeDb(run=run(), architecture=architecture("paddle-det"))
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ), pytest.raises(ValidationError):
        resolve_model_runtime(db, model_version=model_version("weights.pdparams"))


def test_resolve_model_runtime_supports_ultralytics(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.touch()
    db = RuntimeDb(run=run(), architecture=architecture(" ULTRALYTICS-YOLO "))
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ):
        spec = resolve_model_runtime(db, model_version=model_version())
    assert spec.engine == "ultralytics-yolo"
    assert spec.config_path is None


def test_resolve_model_runtime_supports_paddle_with_valid_config(tmp_path):
    weights = tmp_path / "weights.pdparams"
    config = tmp_path / "model.yml"
    weights.touch()
    config.touch()
    db = RuntimeDb(run=run(), architecture=architecture("paddle-det", config_path="model.yml"))
    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ), patch(
        "train_platform.domains.model_assets.runtime.resolve_paddledet_config_path",
        return_value=config,
    ):
        spec = resolve_model_runtime(db, model_version=model_version("weights.pdparams"))
    assert spec.engine == "paddle-det"
    assert spec.config_path == config


@pytest.mark.parametrize("suffix", [".pt", ".pth"])
@pytest.mark.parametrize("engine", ["custom-source", "future-engine", "", None])
def test_unsupported_weights_are_not_model_candidates(tmp_path, suffix, engine):
    weights = tmp_path / f"weights{suffix}"
    weights.touch()
    service = ModelCandidateService()
    candidate_run = SimpleNamespace(
        run_id="run-1",
        project_id=20,
        architecture_id=30,
        result=SimpleNamespace(best_weights_path=str(weights), last_weights_path=None),
        finished_at=None,
        created_at=None,
    )
    with patch(
        "train_platform.domains.model_assets.candidates.resolve_training_path",
        return_value=weights,
    ):
        candidate = service._build_candidate(
            source="training_run",
            model_version=None,
            run=candidate_run,
            arch=architecture(engine) if engine is not None else None,
        )
    assert candidate is None


@pytest.mark.parametrize("engine", ["custom-source", "future-engine", ""])
def test_worker_rejects_invalid_engine_before_http_request(engine):
    client = ModelWorkerClient()
    with patch("train_platform.platform.runtime.model_workers.requests.post") as post:
        with pytest.raises(ModelWorkerError):
            client.execute_model(
                engine=engine,
                weights_path="weights.pt",
                image_path="image.jpg",
                conf=0.25,
                iou=0.45,
            )
    post.assert_not_called()


def test_worker_url_rejects_invalid_engine():
    with pytest.raises(ModelWorkerError):
        ModelWorkerClient._worker_url("custom-source")


@pytest.mark.parametrize("method", ["infer_video_frames", "dispatch_inference_job"])
def test_other_worker_entrypoints_reject_invalid_engine(method):
    client = ModelWorkerClient()
    kwargs = {
        "engine": "custom-source",
        "weights_path": "weights.pt",
        "conf": 0.25,
        "iou": 0.45,
    }
    if method == "infer_video_frames":
        kwargs.update(video_token="video", frame_interval=1)
    else:
        kwargs.update(
            job_id="job",
            mode="image",
            input_tokens=["image"],
            video_token=None,
            show_labels=True,
            show_confidence=True,
        )
    with patch("train_platform.platform.runtime.model_workers.requests.post") as post:
        with pytest.raises(ModelWorkerError):
            getattr(client, method)(**kwargs)
    post.assert_not_called()


def _seed_runtime_deployment_rows(db, *, weights_path):
    safe_arch = ModelArchitecture(
        architecture_id=30,
        family="YOLO",
        variant="safe",
        task_type=TaskType.DETECTION,
        engine="ultralytics-yolo",
    )
    custom_arch = ModelArchitecture(
        architecture_id=31,
        family="Custom",
        variant="unsafe",
        task_type=TaskType.DETECTION,
        engine="custom-source",
    )
    safe_run = TrainingRun(
        run_id="safe-run",
        project_id=20,
        standard_dataset_id=1,
        architecture_id=30,
        name="safe",
        status=TrainingRunStatus.COMPLETED,
    )
    custom_run = TrainingRun(
        run_id="custom-run",
        project_id=20,
        standard_dataset_id=1,
        architecture_id=31,
        name="custom",
        status=TrainingRunStatus.COMPLETED,
    )
    safe_model = ModelVersion(
        model_version_id=10,
        project_id=20,
        run_id="safe-run",
        version="v1",
        stage=ModelStage.PRODUCTION,
        weights_path=str(weights_path),
    )
    custom_model = ModelVersion(
        model_version_id=11,
        project_id=20,
        run_id="custom-run",
        version="v2",
        stage=ModelStage.PRODUCTION,
        weights_path=str(weights_path),
    )
    current = Deployment(
        deployment_id=40,
        model_version_id=10,
        name="current",
        platform=DeploymentPlatform.LOCAL,
        status=DeploymentStatus.PENDING,
        is_active=False,
    )
    unsafe_history = Deployment(
        deployment_id=41,
        model_version_id=11,
        name="unsafe-history",
        platform=DeploymentPlatform.LOCAL,
        status=DeploymentStatus.INACTIVE,
        is_active=False,
    )
    db.add_all([safe_arch, custom_arch, safe_run, custom_run, safe_model, custom_model, current, unsafe_history])
    db.commit()
    return current, custom_model


@pytest.mark.parametrize("engine", ["custom-source", "future-engine", ""])
def test_real_db_deployment_admission_and_rollback_fail_closed(sqlite_db, tmp_path, engine):
    weights = tmp_path / "weights.pt"
    weights.touch()
    current, custom_model = _seed_runtime_deployment_rows(sqlite_db, weights_path=weights)
    sqlite_db.get(ModelArchitecture, 31).engine = engine
    unsafe = sqlite_db.get(Deployment, 41)
    unsafe.api_key_hash = "existing-hash"
    unsafe.api_key_hint = "existing-hint"
    sqlite_db.commit()
    deployment_service = DeploymentService()
    run_service = DeploymentRunService()

    with patch(
        "train_platform.domains.model_assets.runtime.resolve_training_path",
        return_value=weights,
    ):
        with patch("train_platform.domains.deployment.service.assert_valid_license"), pytest.raises(ConflictError):
            deployment_service.create_deployment(
                sqlite_db,
                obj={
                    "model_version_id": custom_model.model_version_id,
                    "name": "rejected",
                    "platform": DeploymentPlatform.LOCAL,
                },
            )
        assert sqlite_db.query(Deployment).filter(Deployment.name == "rejected").count() == 0

        with patch.object(run_service, "_start_pipeline_thread") as start_thread, patch(
            "train_platform.domains.deployment.runs.service.generate_api_key"
        ) as generate_key, pytest.raises(ConflictError):
            run_service.execute_deployment(sqlite_db, 41, payload={})
        start_thread.assert_not_called()
        generate_key.assert_not_called()
        assert sqlite_db.query(DeploymentRun).count() == 0
        sqlite_db.refresh(unsafe)
        assert unsafe.status == DeploymentStatus.INACTIVE
        assert unsafe.is_active is False
        assert unsafe.api_key_hash == "existing-hash"
        assert unsafe.api_key_hint == "existing-hint"
        sqlite_db.refresh(current)
        assert current.model_version_id == 10
        assert current.status == DeploymentStatus.PENDING
        assert current.api_key_hash is None

        candidates = deployment_service.get_rollback_candidates(sqlite_db, 40)
        assert candidates["candidates"] == []
        with pytest.raises(ConflictError):
            deployment_service.rollback_deployment(
                sqlite_db,
                40,
                target_model_version_id=11,
                reason="unsafe",
                operator="tester",
            )
    sqlite_db.refresh(current)
    assert current.model_version_id == 10
    assert sqlite_db.query(DeploymentLog).count() == 0
    assert custom_model.stage == ModelStage.PRODUCTION


def test_create_deployment_rejects_unsafe_model_before_row_creation():
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.first.return_value = model_version()
    service = DeploymentService()
    with patch("train_platform.domains.deployment.service.assert_valid_license"), patch(
        "train_platform.domains.deployment.service.resolve_model_runtime",
        side_effect=ConflictError("unsupported"),
    ), pytest.raises(ConflictError):
        service.create_deployment(
            db,
            obj={"model_version_id": 10, "name": "unsafe", "platform": "local"},
        )
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_create_deployment_keeps_supported_model_flow():
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.first.return_value = model_version()
    row = SimpleNamespace(deployment_id=40)
    with patch("train_platform.domains.deployment.service.assert_valid_license"), patch(
        "train_platform.domains.deployment.service.resolve_model_runtime"
    ) as resolve, patch(
        "train_platform.domains.deployment.service.Deployment", return_value=row
    ), patch("train_platform.domains.deployment.service.append_deployment_log"):
        result = DeploymentService().create_deployment(
            db,
            obj={"model_version_id": 10, "name": "safe", "platform": "local"},
        )
    assert result is row
    resolve.assert_called_once()
    db.add.assert_called_once_with(row)
    db.commit.assert_called_once()


def test_execute_deployment_rejects_unsafe_model_before_side_effects():
    unsafe_deployment = SimpleNamespace(deployment_id=40, model_version_id=10)
    unsafe_model = model_version()
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.first.side_effect = [unsafe_deployment, unsafe_model]
    service = DeploymentRunService()
    with patch(
        "train_platform.domains.deployment.runs.service.resolve_model_runtime",
        side_effect=ConflictError("unsupported"),
    ), patch(
        "train_platform.domains.deployment.runs.service.generate_api_key"
    ) as generate_key, patch.object(service, "_start_pipeline_thread") as start_thread, pytest.raises(ConflictError):
        service.execute_deployment(db, 40, payload={})
    assert db.query.call_count == 2
    generate_key.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()
    start_thread.assert_not_called()


def test_execute_deployment_keeps_supported_model_flow():
    deployment = SimpleNamespace(
        deployment_id=40,
        model_version_id=10,
        api_key_hint=None,
        api_key_hash="existing",
    )
    model = model_version()
    project = SimpleNamespace(project_id=20)
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.with_for_update.return_value = query
    query.order_by.return_value = query
    query.first.side_effect = [deployment, model, project, None]
    queued = SimpleNamespace(run_id="new-run")
    service = DeploymentRunService()
    with patch(
        "train_platform.domains.deployment.runs.service.resolve_model_runtime"
    ) as resolve, patch(
        "train_platform.domains.deployment.runs.service.lifecycle.new_queued_run",
        return_value=queued,
    ), patch(
        "train_platform.domains.deployment.runs.service.lifecycle.snapshot_steps",
        return_value=[],
    ), patch(
        "train_platform.domains.deployment.runs.service.activation.prepare_deployment_for_run"
    ), patch(
        "train_platform.domains.deployment.runs.service.append_run_log"
    ), patch.object(service, "_start_pipeline_thread") as start_thread:
        result = service.execute_deployment(db, 40, payload={"rotate_api_key": False})
    assert result["run"] is queued
    resolve.assert_called_once()
    db.add.assert_called_once_with(queued)
    db.commit.assert_called_once()
    start_thread.assert_called_once()
    assert isinstance(start_thread.call_args.args[0], str)


def test_rollback_candidates_filter_unsafe_runtime():
    deployment = SimpleNamespace(deployment_id=40, model_version_id=10)
    safe = SimpleNamespace(model_version_id=11)
    unsafe = SimpleNamespace(model_version_id=12)
    service = DeploymentService()
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.all.return_value = [safe, unsafe]
    with patch.object(service, "get_deployment", return_value=deployment), patch.object(
        service, "_project_id_of_deployment", return_value=20
    ), patch.object(
        service, "_candidate_model_version_ids", return_value={11, 12}
    ), patch(
        "train_platform.domains.deployment.service.resolve_model_runtime",
        side_effect=[SimpleNamespace(), ConflictError("unsupported")],
    ):
        result = service.get_rollback_candidates(db, 40)
    assert result["candidates"] == [safe]


def test_direct_rollback_validates_runtime_before_activation():
    deployment = SimpleNamespace(deployment_id=40, model_version_id=10)
    current = SimpleNamespace(model_version_id=10, project_id=20)
    target = SimpleNamespace(model_version_id=11, project_id=20)
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.first.side_effect = [deployment, current, target]
    service = DeploymentService()
    with patch.object(
        service, "_candidate_model_version_ids", return_value={11}
    ), patch(
        "train_platform.domains.deployment.service.resolve_model_runtime",
        side_effect=ConflictError("unsupported"),
    ), patch(
        "train_platform.domains.deployment.service.activation.activate_deployment"
    ) as activate, pytest.raises(ConflictError):
        service.rollback_deployment(
            db,
            40,
            target_model_version_id=11,
            reason="rollback",
            operator="tester",
        )
    activate.assert_not_called()
    db.commit.assert_not_called()


def test_direct_rollback_keeps_supported_target_flow():
    deployment = SimpleNamespace(deployment_id=40, model_version_id=10)
    current = SimpleNamespace(model_version_id=10, project_id=20, version="v1")
    target = SimpleNamespace(model_version_id=11, project_id=20, version="v2")
    activated = SimpleNamespace(deployment_id=40)
    activation_result = SimpleNamespace(
        deployment=activated,
        previous_model_version_id=10,
    )
    event = SimpleNamespace()
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.first.side_effect = [deployment, current, target, current, target]
    service = DeploymentService()
    with patch.object(
        service, "_candidate_model_version_ids", return_value={11}
    ), patch(
        "train_platform.domains.deployment.service.resolve_model_runtime"
    ) as resolve, patch(
        "train_platform.domains.deployment.service.activation.activate_deployment",
        return_value=activation_result,
    ) as activate, patch(
        "train_platform.domains.deployment.service.append_deployment_log",
        return_value=event,
    ), patch(
        "train_platform.domains.deployment.service.map_rollback_log",
        return_value={"action": "rollback"},
    ):
        result = service.rollback_deployment(
            db,
            40,
            target_model_version_id=11,
            reason="rollback",
            operator="tester",
        )
    assert result["deployment"] is activated
    resolve.assert_called_once_with(db, model_version=target)
    activate.assert_called_once()
    db.commit.assert_called_once()


def test_latency_benchmark_rejects_missing_engine_before_worker_request(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.touch()
    benchmark = tmp_path / "image.jpg"
    benchmark.touch()
    service = TrainingRunBenchmarkService()
    benchmark_run = SimpleNamespace(
        architecture=SimpleNamespace(engine=""),
        result=SimpleNamespace(best_weights_path="weights.pt", last_weights_path=None),
    )
    with patch(
        "train_platform.domains.training.runs.benchmarks.resolve_training_path",
        return_value=weights,
    ), patch.object(service._worker, "execute_model") as execute, pytest.raises(ConflictError):
        service.measure_inference_latency(
            MagicMock(),
            run=benchmark_run,
            benchmark_image=benchmark,
        )
    execute.assert_not_called()


def test_batch_benchmark_rejects_missing_engine_before_cached_result(tmp_path):
    service = TrainingRunBenchmarkService()
    benchmark_run = SimpleNamespace(
        run_id="run-1",
        architecture=SimpleNamespace(engine=""),
        status=SimpleNamespace(),
        result=SimpleNamespace(inference_time_ms=12.5),
    )
    db = MagicMock()
    with patch.object(service, "ensure_benchmark_image", return_value=tmp_path / "image.jpg"), patch(
        "train_platform.domains.training.runs.benchmarks.TrainingRunService.get_run",
        return_value=benchmark_run,
    ):
        result = service.benchmark_inference_times(db, run_ids=["run-1"])
    assert result["items"][0]["status"] == "failed"
    assert "Model runtime engine is missing" in result["items"][0]["message"]
    db.commit.assert_not_called()
