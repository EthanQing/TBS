from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import DatasetType, TaskType, TrainingRunStatus
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import TrainingRun, TrainingRunParameters, TrainingRunResult
from train_platform.models.v3.training_run_meta import TrainingRunMeta
from train_platform.services.v3.project_service import ProjectService
from train_platform.services.v3.training_run_service import TrainingRunService


def _session():
    engine = create_engine("sqlite:///:memory:")
    for table in (
        StandardDataset.__table__,
        Project.__table__,
        ModelArchitecture.__table__,
        TrainingRun.__table__,
        TrainingRunParameters.__table__,
        TrainingRunResult.__table__,
        TrainingRunMeta.__table__,
    ):
        table.create(engine)
    return sessionmaker(bind=engine)()


def _seed_base(db):
    dataset = StandardDataset(
        standard_dataset_id=1,
        name="dataset",
        dataset_type=DatasetType.DETECTION,
        format="yolo",
        storage_path="standard/1",
    )
    project = Project(
        project_id=10,
        name="project",
        standard_dataset_id=1,
        task_type=TaskType.DETECTION,
        is_active=True,
    )
    architecture = ModelArchitecture(
        architecture_id=20,
        family="YOLOv8",
        variant="yolov8n",
        task_type=TaskType.DETECTION,
        engine="ultralytics-yolo",
    )
    db.add_all([dataset, project, architecture])
    db.commit()


def _run(run_id: str, status: TrainingRunStatus, **overrides) -> TrainingRun:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    data = {
        "run_id": run_id,
        "project_id": 10,
        "standard_dataset_id": 1,
        "architecture_id": 20,
        "name": run_id,
        "status": status,
        "progress": 100 if status == TrainingRunStatus.COMPLETED else 35,
        "current_epoch": 4,
        "total_epochs": 10,
        "hidden": False,
        "created_at": now,
        "updated_at": now,
        "finished_at": now if status == TrainingRunStatus.COMPLETED else None,
    }
    data.update(overrides)
    return TrainingRun(**data)


def test_project_training_alerts_count_running_and_unreviewed_completed() -> None:
    db = _session()
    _seed_base(db)
    db.add_all(
        [
            _run("running", TrainingRunStatus.RUNNING),
            _run("completed-new", TrainingRunStatus.COMPLETED),
            _run("completed-reviewed", TrainingRunStatus.COMPLETED),
        ]
    )
    db.add(
        TrainingRunMeta(
            run_id="completed-reviewed",
            extra={"project_card_reviewed_at": "2024-01-02T00:00:00+00:00"},
        )
    )
    db.commit()

    alerts = ProjectService().list_training_alerts(db, [10])

    assert alerts == [
        {
            "project_id": 10,
            "running_count": 1,
            "latest_running_run": {
                "run_id": "running",
                "name": "running",
                "status": "running",
                "progress": 35,
                "current_epoch": 4,
                "total_epochs": 10,
                "updated_at": alerts[0]["latest_running_run"]["updated_at"],
                "finished_at": None,
            },
            "unreviewed_completed_count": 1,
            "latest_unreviewed_completed_run": {
                "run_id": "completed-new",
                "name": "completed-new",
                "status": "completed",
                "progress": 100,
                "current_epoch": 4,
                "total_epochs": 10,
                "updated_at": alerts[0]["latest_unreviewed_completed_run"]["updated_at"],
                "finished_at": alerts[0]["latest_unreviewed_completed_run"]["finished_at"],
            },
        }
    ]


def test_review_marks_completed_run_and_preserves_extra() -> None:
    db = _session()
    _seed_base(db)
    db.add(_run("completed-new", TrainingRunStatus.COMPLETED))
    db.add(TrainingRunMeta(run_id="completed-new", extra={"keep": "yes"}))
    db.commit()

    before = ProjectService().list_training_alerts(db, [10])[0]
    assert before["unreviewed_completed_count"] == 1

    result = TrainingRunService().mark_project_card_reviewed(db, "completed-new", source="training-report")

    assert result["run_id"] == "completed-new"
    assert result["reviewed"] is True
    meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == "completed-new").first()
    assert meta.extra["keep"] == "yes"
    assert meta.extra["project_card_reviewed_at"]
    assert meta.extra["project_card_review_source"] == "training-report"

    after = ProjectService().list_training_alerts(db, [10])[0]
    assert after["unreviewed_completed_count"] == 0
    assert after["latest_unreviewed_completed_run"] is None
