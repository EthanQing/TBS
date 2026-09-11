from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import train_platform.models.v3  # noqa: F401 - register complete metadata
from train_platform.domains.training.frameworks.contract import TrainingArtifactReport
from train_platform.domains.training.runs import artifacts, lifecycle, progress
from train_platform.models.v3.base import V3Base
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunArtifact,
    TrainingRunEpochMetric,
)
from train_platform.workers.training import train_entry_impl


def test_resolve_execution_guard_pid_accepts_direct_pid():
    run = SimpleNamespace(pid=101)

    assert train_entry_impl._resolve_execution_guard_pid(run, actual_pid=101) == 101


def test_resolve_execution_guard_pid_accepts_launcher_ancestor(monkeypatch):
    monkeypatch.setattr(
        train_entry_impl.psutil,
        "Process",
        lambda pid: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=202)]),
    )

    assert (
        train_entry_impl._resolve_execution_guard_pid(
            SimpleNamespace(pid=202), actual_pid=303
        )
        == 202
    )


def test_resolve_execution_guard_pid_rejects_unrelated_pid(monkeypatch):
    monkeypatch.setattr(
        train_entry_impl.psutil,
        "Process",
        lambda pid: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=404)]),
    )

    with pytest.raises(RuntimeError, match="ownership mismatch"):
        train_entry_impl._resolve_execution_guard_pid(
            SimpleNamespace(pid=202), actual_pid=303
        )


def test_resolve_execution_guard_pid_rejects_missing_claim_pid():
    with pytest.raises(RuntimeError, match="does not have a pid"):
        train_entry_impl._resolve_execution_guard_pid(
            SimpleNamespace(pid=None), actual_pid=303
        )


def test_resolve_execution_guard_pid_fails_closed_when_psutil_fails(monkeypatch):
    def fail_process(pid):
        raise train_entry_impl.psutil.Error("unavailable")

    monkeypatch.setattr(train_entry_impl.psutil, "Process", fail_process)

    with pytest.raises(RuntimeError, match="ownership mismatch"):
        train_entry_impl._resolve_execution_guard_pid(
            SimpleNamespace(pid=202), actual_pid=303
        )


class _ClaimSession:
    def __init__(self, run):
        self.run = run
        self.closed = False

    def query(self, model):
        return self

    def filter(self, condition):
        return self

    def first(self):
        return self.run

    def close(self):
        self.closed = True


def test_wait_for_execution_guard_pid_observes_delayed_worker_claim(monkeypatch):
    sessions = [
        _ClaimSession(SimpleNamespace(status=TrainingRunStatus.QUEUED, pid=None)),
        _ClaimSession(SimpleNamespace(status=TrainingRunStatus.RUNNING, pid=303)),
    ]
    monkeypatch.setattr(train_entry_impl, "SessionLocal", lambda: sessions.pop(0))
    monkeypatch.setattr(train_entry_impl.time, "sleep", lambda interval: None)

    assert (
        train_entry_impl._wait_for_execution_guard_pid(
            "run-1", actual_pid=303, timeout_sec=1, poll_interval_sec=0
        )
        == 303
    )


def test_wait_for_execution_guard_pid_times_out_without_claim(monkeypatch):
    session = _ClaimSession(
        SimpleNamespace(status=TrainingRunStatus.RUNNING, pid=None)
    )
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(train_entry_impl, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        train_entry_impl.time, "monotonic", lambda: next(monotonic_values)
    )

    with pytest.raises(RuntimeError, match="Timed out"):
        train_entry_impl._wait_for_execution_guard_pid(
            "run-1", actual_pid=303, timeout_sec=0.5, poll_interval_sec=0
        )


@pytest.fixture
def ownership_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ownership.db'}")
    V3Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    db.add(
        TrainingRun(
            run_id="run-ownership",
            project_id=1,
            standard_dataset_id=1,
            architecture_id=1,
            name="ownership",
            status=TrainingRunStatus.RUNNING,
            pid=202,
            worker_id="worker-1",
            total_epochs=2,
            current_epoch=0,
            progress=0,
        )
    )
    db.commit()
    db.close()
    try:
        yield factory
    finally:
        engine.dispose()


def test_launcher_guard_persists_metrics_progress_and_artifact(
    ownership_db, tmp_path, monkeypatch
):
    monkeypatch.setattr(progress, "SessionLocal", ownership_db)
    training_dir = tmp_path / "training"
    artifact_path = (
        training_dir / "run-ownership" / "custom_model" / "output" / "best.pt"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"weights")
    monkeypatch.setattr(artifacts, "settings", SimpleNamespace(training_dir=training_dir))

    progress.upsert_epoch_metrics(
        "run-ownership", 0, {"train/box_loss": 1.0}, expected_pid=202
    )
    db = ownership_db()
    first_epoch_run = db.get(TrainingRun, "run-ownership")
    assert first_epoch_run.current_epoch == 0
    assert first_epoch_run.progress == 50
    db.close()

    progress.upsert_epoch_metrics(
        "run-ownership", 1, {"metrics/mAP50(B)": 0.5}, expected_pid=202
    )
    db = ownership_db()
    artifacts.register_reported_artifact(
        db,
        "run-ownership",
        TrainingArtifactReport(role="best_weights", path="best.pt"),
        expected_pid=202,
    )
    db.close()

    db = ownership_db()
    run = db.get(TrainingRun, "run-ownership")
    rows = db.query(TrainingRunEpochMetric).order_by(TrainingRunEpochMetric.epoch).all()
    stored_artifact = db.query(TrainingRunArtifact).one()
    assert [(row.epoch, row.metrics) for row in rows] == [
        (0, {"train/box_loss": 1.0}),
        (1, {"metrics/mAP50(B)": 0.5}),
    ]
    assert run.current_epoch == 1
    assert run.progress == 100
    assert stored_artifact.role == "best_weights"
    db.close()


def test_replaced_claim_rejects_stale_metrics_artifact_heartbeat_and_finalize(
    ownership_db, tmp_path, monkeypatch
):
    monkeypatch.setattr(progress, "SessionLocal", ownership_db)
    training_dir = tmp_path / "training"
    artifact_path = (
        training_dir / "run-ownership" / "custom_model" / "output" / "last.pt"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"weights")
    monkeypatch.setattr(artifacts, "settings", SimpleNamespace(training_dir=training_dir))

    db = ownership_db()
    run = db.get(TrainingRun, "run-ownership")
    heartbeat = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run.pid = 999
    run.heartbeat_at = heartbeat
    db.commit()
    db.close()

    progress.upsert_epoch_metrics(
        "run-ownership", 0, {"train/box_loss": 1.0}, expected_pid=202
    )
    db = ownership_db()
    assert (
        artifacts.register_reported_artifact(
            db,
            "run-ownership",
            TrainingArtifactReport(role="last_weights", path="last.pt"),
            expected_pid=202,
        )
        is None
    )
    assert lifecycle.touch_heartbeat(db, "run-ownership", expected_pid=202) is False
    result = lifecycle.finalize_execution(
        db, "run-ownership", exit_code=0, expected_pid=202
    )
    db.expire_all()
    run = db.get(TrainingRun, "run-ownership")
    assert result.changed is False
    assert run.status == TrainingRunStatus.RUNNING
    assert run.pid == 999
    assert run.heartbeat_at.replace(tzinfo=timezone.utc) == heartbeat
    assert db.query(TrainingRunEpochMetric).count() == 0
    assert db.query(TrainingRunArtifact).count() == 0
    db.close()


def test_main_does_not_finalize_when_execution_claim_is_unresolved(monkeypatch):
    finalized = []
    monkeypatch.setattr(train_entry_impl, "assert_valid_license", lambda: None)
    monkeypatch.setattr(
        train_entry_impl,
        "_wait_for_execution_guard_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unresolved")),
    )
    monkeypatch.setattr(
        train_entry_impl, "finalize_execution", lambda *args, **kwargs: finalized.append(True)
    )

    assert train_entry_impl.main(["--run-id", "run-ownership"]) == 1
    assert finalized == []
