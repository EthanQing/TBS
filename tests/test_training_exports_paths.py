from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3  # noqa: F401
from train_platform.domains.training.execution_paths import prepare_ultralytics_execution
from train_platform.domains.training.runs import exports
from train_platform.models.v3.base import V3Base
from train_platform.models.v3.training_run import TrainingRunArtifact


@pytest.fixture
def export_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'exports.db'}")
    V3Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    training_dir = tmp_path / "training"
    monkeypatch.setattr(exports, "settings", SimpleNamespace(training_dir=training_dir))
    monkeypatch.setattr(
        exports.TrainingRunService,
        "get_run",
        lambda self, session, run_id: SimpleNamespace(run_id=str(run_id)),
    )
    try:
        yield db, training_dir
    finally:
        db.close()
        engine.dispose()


@pytest.mark.parametrize("recorded", [False, True])
def test_export_training_run_uses_effective_weights_directory(export_db, monkeypatch, recorded):
    db, training_dir = export_db
    run_root = training_dir / "run-1"
    output_dir = prepare_ultralytics_execution(run_root).output_dir if recorded else run_root
    source = output_dir / "weights" / "best.pt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-weights")
    calls = []

    class FakeModelWorkerClient:
        def export_ultralytics_onnx(self, *, src_pt, out_onnx, dynamic, opset, imgsz):
            calls.append((src_pt, out_onnx, dynamic, opset, imgsz))
            out_onnx.write_bytes(b"onnx")

    monkeypatch.setattr(exports, "ModelWorkerClient", FakeModelWorkerClient)

    result = exports.export_training_run(
        db,
        "run-1",
        format="onnx",
        weights="best",
        dynamic=True,
        opset=17,
        imgsz=320,
    )

    destination = output_dir / "weights" / "best.onnx"
    assert calls == [(source.resolve(), destination.resolve(), True, 17, 320)]
    assert destination.read_bytes() == b"onnx"
    assert source.read_bytes() == b"source-weights"
    expected_path = "run-1/output/weights/best.onnx" if recorded else "run-1/weights/best.onnx"
    assert result.artifact is not None
    assert result.artifact.path == expected_path
    assert db.query(TrainingRunArtifact).one().path == expected_path


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("format", ["pt", "onnx"])
def test_download_export_falls_back_to_effective_layout(export_db, recorded, format):
    db, training_dir = export_db
    run_root = training_dir / "run-1"
    output_dir = prepare_ultralytics_execution(run_root).output_dir if recorded else run_root
    extension = "onnx" if format == "onnx" else "pt"
    expected = output_dir / "weights" / f"best.{extension}"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"effective")
    if recorded:
        stale = run_root / "weights" / f"best.{extension}"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale")

    result = exports.download_export(db, "run-1", format=format, weights="best")

    assert result.path == expected.resolve()
    assert result.path.read_bytes() == b"effective"


def test_download_export_prefers_indexed_onnx_artifact(export_db):
    db, training_dir = export_db
    run_root = training_dir / "run-1"
    output_dir = prepare_ultralytics_execution(run_root).output_dir
    fallback = output_dir / "weights" / "best.onnx"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"layout-fallback")
    stale = run_root / "weights" / "best.onnx"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale-root")
    indexed = output_dir / "exports" / "selected.onnx"
    indexed.parent.mkdir(parents=True, exist_ok=True)
    indexed.write_bytes(b"indexed")
    db.add(
        TrainingRunArtifact(
            run_id="run-1",
            kind="export",
            name="best.onnx",
            path="run-1/output/exports/selected.onnx",
            size_bytes=len(b"indexed"),
        )
    )
    db.commit()

    result = exports.download_export(db, "run-1", format="onnx", weights="best")

    assert result.path == indexed.resolve()
    assert result.path.read_bytes() == b"indexed"
