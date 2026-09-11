import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3  # noqa: F401

from train_platform.domains.training.execution_paths import (
    prepare_ultralytics_execution,
    read_ultralytics_paths,
    resolve_ultralytics_checkpoint,
    resolve_ultralytics_resume_checkpoint,
)
from train_platform.domains.training.runs import artifacts
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.base import V3Base
from train_platform.models.v3.enums import TaskType, TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunArtifact, TrainingRunResult


def test_prepare_records_new_layout_and_read_does_not_create_directories(tmp_path):
    untouched = tmp_path / "untouched"
    legacy = read_ultralytics_paths(untouched)
    assert legacy.output_dir == untouched.resolve()
    assert not untouched.exists()

    paths = prepare_ultralytics_execution(tmp_path / "run")
    assert paths.output_dir == (tmp_path / "run" / "output").resolve()
    assert paths.runtime_dir.is_dir()
    assert paths.logs_dir.is_dir()
    assert paths.output_dir.is_dir()
    assert json.loads((paths.runtime_dir / "layout.json").read_text(encoding="utf-8")) == {
        "version": 1,
        "engine": "ultralytics-yolo",
        "output_dir": "output",
    }


def test_recorded_layout_is_authoritative_and_resume_has_legacy_fallback(tmp_path):
    run_root = tmp_path / "run"
    paths = prepare_ultralytics_execution(run_root)
    legacy_last = run_root / "weights" / "last.pt"
    legacy_last.parent.mkdir(parents=True)
    legacy_last.write_bytes(b"legacy")

    assert resolve_ultralytics_checkpoint(run_root, "last") is None
    assert resolve_ultralytics_resume_checkpoint(run_root) == legacy_last

    active_last = paths.weights_dir / "last.pt"
    active_last.parent.mkdir(parents=True)
    active_last.write_bytes(b"active")
    assert resolve_ultralytics_checkpoint(run_root, "last") == active_last
    assert resolve_ultralytics_resume_checkpoint(run_root) == active_last
    assert legacy_last.read_bytes() == b"legacy"


def test_legacy_layout_and_invalid_manifest(tmp_path):
    run_root = tmp_path / "legacy"
    last = run_root / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"checkpoint")
    assert read_ultralytics_paths(run_root).output_dir == run_root.resolve()
    assert resolve_ultralytics_checkpoint(run_root, "last") == last

    manifest = run_root / "runtime" / "layout.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"version": 1, "engine": "ultralytics-yolo", "output_dir": "../outside"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="confined"):
        read_ultralytics_paths(run_root)


def test_platform_trainer_reapplies_paths_during_real_base_initialization(monkeypatch, tmp_path):
    import importlib

    import torch
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.engine import trainer as trainer_module

    from train_platform.domains.training.frameworks.ultralytics_trainers import PlatformDetectionTrainer

    checkpoint = tmp_path / "source" / "last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"source-checkpoint")
    old_data = tmp_path / "old-data.yaml"
    old_data.write_text("names: [old]\n", encoding="utf-8")
    checkpoint_args = dict(DEFAULT_CFG_DICT)
    checkpoint_args.update(
        {
            "data": str(old_data),
            "project": str(tmp_path / "old-project"),
            "name": "old-name",
            "save_dir": str(tmp_path / "old-output"),
            "model": "old.pt",
            "resume": str(checkpoint),
        }
    )
    checkpoint_model = SimpleNamespace(args=checkpoint_args)
    monkeypatch.setattr(trainer_module, "load_checkpoint", lambda path: (checkpoint_model, {}))
    monkeypatch.setattr(trainer_module, "select_device", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(trainer_module, "init_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "print_args", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "check_model_file_from_stem", lambda model: model)
    monkeypatch.setattr(trainer_module.callbacks, "add_integration_callbacks", lambda trainer: None)
    monkeypatch.setattr(PlatformDetectionTrainer, "get_dataset", lambda self: ({}, {}))

    output = tmp_path / "run" / "output"
    overrides = {
        "data": str(tmp_path / "run" / "runtime" / "data.runtime.yaml"),
        "project": str(tmp_path / "run"),
        "name": "output",
        "exist_ok": True,
        "save_dir": str(output),
        "model": str(checkpoint),
        "resume": str(checkpoint),
        "device": "cpu",
    }
    trainer = PlatformDetectionTrainer(overrides=dict(overrides), _callbacks={})

    for key, value in overrides.items():
        assert getattr(trainer.args, key) == value
    assert trainer.save_dir == output.resolve()
    assert trainer.wdir == output.resolve() / "weights"
    assert trainer.csv == output.resolve() / "results.csv"
    assert (output / "args.yaml").is_file()

    ddp_module = importlib.import_module(trainer.__class__.__module__)
    ddp_class = getattr(ddp_module, trainer.__class__.__name__)
    ddp_trainer = ddp_class(overrides=vars(trainer.args).copy(), _callbacks={})
    assert ddp_trainer.save_dir == output.resolve()
    assert checkpoint.read_bytes() == b"source-checkpoint"


def test_platform_trainer_keeps_base_resume_state_restoration():
    from train_platform.domains.training.frameworks.ultralytics_trainers import PlatformDetectionTrainer

    trainer = object.__new__(PlatformDetectionTrainer)
    trainer.resume = True
    trainer.args = SimpleNamespace(model="checkpoint.pt", close_mosaic=0)
    trainer.model = "model"
    trainer.epochs = 10
    trainer.start_epoch = 0
    restored = []
    trainer._load_checkpoint_state = restored.append
    checkpoint_state = {"epoch": 3, "optimizer": {"state": "preserved"}}

    trainer.resume_training(checkpoint_state)

    assert trainer.start_epoch == 4
    assert restored == [checkpoint_state]


@pytest.mark.parametrize("recorded", [False, True])
def test_ultralytics_artifact_index_uses_only_effective_layout(tmp_path, monkeypatch, recorded):
    engine = create_engine(f"sqlite:///{tmp_path / 'artifacts.db'}")
    V3Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    architecture = ModelArchitecture(
        architecture_id=1,
        family="YOLO",
        variant="yolo11n",
        task_type=TaskType.DETECTION,
        engine="ultralytics-yolo",
    )
    run = TrainingRun(
        run_id="run-1",
        project_id=1,
        standard_dataset_id=1,
        architecture_id=1,
        name="run",
        status=TrainingRunStatus.COMPLETED,
        progress=100,
        current_epoch=1,
        architecture=architecture,
    )
    db.add(run)
    db.commit()
    training_dir = tmp_path / "training"
    run_root = training_dir / "run-1"
    output = prepare_ultralytics_execution(run_root).output_dir if recorded else run_root
    (output / "weights").mkdir(parents=True, exist_ok=True)
    (output / "weights" / "best.pt").write_bytes(b"effective")
    (output / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    if recorded:
        (run_root / "weights").mkdir(parents=True, exist_ok=True)
        (run_root / "weights" / "last.pt").write_bytes(b"stale")
        (run_root / "results.csv").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(artifacts, "settings", SimpleNamespace(training_dir=training_dir))

    artifacts.index_completion_artifacts(db, "run-1")
    paths = {row.path for row in db.query(TrainingRunArtifact).all()}
    prefix = "run-1/output" if recorded else "run-1"
    assert f"{prefix}/weights/best.pt" in paths
    assert f"{prefix}/results.csv" in paths
    result = db.query(TrainingRunResult).one()
    assert result.best_weights_path == f"{prefix}/weights/best.pt"
    assert result.last_weights_path is None
    assert result.results_dir == "run-1"
    if recorded:
        assert "run-1/weights/last.pt" not in paths
        assert "run-1/results.csv" not in paths
    db.close()
    engine.dispose()


@pytest.mark.parametrize("training_engine", ["paddle-det", "custom-source"])
def test_other_engines_keep_native_and_reported_artifacts(tmp_path, monkeypatch, training_engine):
    database = create_engine("sqlite://")
    V3Base.metadata.create_all(database)
    training_dir = tmp_path / "training"
    run_root = training_dir / "run-1"
    monkeypatch.setattr(artifacts, "settings", SimpleNamespace(training_dir=training_dir))
    with sessionmaker(bind=database)() as db:
        architecture = ModelArchitecture(
            family="test", variant="test", task_type=TaskType.DETECTION, engine=training_engine,
        )
        run = TrainingRun(
            run_id="run-1", project_id=1, standard_dataset_id=1, architecture=architecture,
            name="run", status=TrainingRunStatus.COMPLETED,
        )
        db.add(run)
        db.flush()
        (run_root / "logs").mkdir(parents=True)
        (run_root / "logs" / "train.stdout.log").write_text("training log", encoding="utf-8")
        if training_engine == "paddle-det":
            best_path = "run-1/weights/best.pdparams"
            (run_root / "weights").mkdir()
            (training_dir / best_path).write_bytes(b"paddle weights")
            (run_root / "weights" / "best.pdopt").write_bytes(b"optimizer")
            (run_root / "results.png").write_bytes(b"plot")
        else:
            best_path = "run-1/custom_model/output/model.pth"
            (training_dir / best_path).parent.mkdir(parents=True)
            (training_dir / best_path).write_bytes(b"custom weights")
            db.add(TrainingRunArtifact(
                run_id="run-1", kind="weights", role="best_weights", name="model.pth",
                path=best_path, meta={"source": "reported"},
            ))
        db.flush()

        artifacts.index_completion_artifacts(db, "run-1")

        paths = {artifact.path for artifact in db.query(TrainingRunArtifact).all()}
        assert best_path in paths
        assert "run-1/logs/train.stdout.log" in paths
        if training_engine == "paddle-det":
            assert "run-1/weights/best.pdopt" in paths
            assert "run-1/results.png" in paths
        result = db.query(TrainingRunResult).one()
        assert result.best_weights_path == best_path
        assert result.results_dir == "run-1"
        assert not (run_root / "runtime").exists()
        assert not (run_root / "output").exists()
    database.dispose()
