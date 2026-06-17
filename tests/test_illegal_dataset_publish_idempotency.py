from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.models.v3.enums import DatasetType, DatasetVersionStatus
from train_platform.models.v3.illegal_dataset import (
    IllegalDataset,
    IllegalDatasetLabelMapping,
    IllegalDatasetPublishJob,
    IllegalDatasetVersion,
)
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent, StandardDatasetImage
from train_platform.schemas.v3.illegal_datasets import IllegalDatasetPublishRequest
from train_platform.services.v3.illegal_dataset_publish_job_service import IllegalDatasetPublishJobService
from train_platform.services.v3.standard_dataset_service import StandardDatasetService


def _make_db():
    engine = create_engine("sqlite:///:memory:")
    for table in (
        StandardDataset.__table__,
        StandardDatasetEvent.__table__,
        StandardDatasetImage.__table__,
        IllegalDataset.__table__,
        IllegalDatasetVersion.__table__,
        IllegalDatasetLabelMapping.__table__,
        IllegalDatasetPublishJob.__table__,
    ):
        table.create(engine, checkfirst=True)
    factory = sessionmaker(bind=engine)
    return factory()


def _make_db_factory():
    engine = create_engine("sqlite:///:memory:")
    for table in (
        StandardDataset.__table__,
        StandardDatasetEvent.__table__,
        StandardDatasetImage.__table__,
        IllegalDataset.__table__,
        IllegalDatasetVersion.__table__,
        IllegalDatasetLabelMapping.__table__,
        IllegalDatasetPublishJob.__table__,
    ):
        table.create(engine, checkfirst=True)
    return sessionmaker(bind=engine)


def _seed_illegal_dataset(db):
    dataset = IllegalDataset(
        illegal_dataset_id=1000001,
        name="illegal source",
        dataset_type=DatasetType.DETECTION,
        format="yolo",
        storage_path="illegal/1000001",
    )
    db.add(dataset)
    db.flush()
    version = IllegalDatasetVersion(
        illegal_dataset_id=int(dataset.illegal_dataset_id),
        version=1,
        status=DatasetVersionStatus.FINALIZED,
        manifest_path="illegal/.versions/1000001/v1/.manifest.json",
    )
    db.add(version)
    db.flush()
    dataset.active_version_id = int(version.version_id)
    db.commit()
    return dataset, version


def _payload(**overrides) -> IllegalDatasetPublishRequest:
    data = {
        "name": "first-name",
        "description": None,
        "version_id": None,
        "label_filters": [],
        "label_mapping_overrides": {"raw": {"mapped_label": "mapped", "status": "keep"}},
        "split": {},
        "publish_config": {"conversion": {"slice": {"enabled": False}}},
    }
    data.update(overrides)
    return IllegalDatasetPublishRequest(**data)


def test_standard_dataset_create_uses_database_generated_id(tmp_path: Path, monkeypatch) -> None:
    db = _make_db()
    storage_root = tmp_path / "datasets"
    monkeypatch.setattr(
        "train_platform.services.v3.dataset_common.resolve_storage_token",
        lambda token: storage_root / str(token),
    )
    monkeypatch.setattr(
        "train_platform.services.v3.standard_dataset_service.resolve_storage_token",
        lambda token: storage_root / str(token),
    )
    try:
        row = StandardDatasetService().create_dataset(
            db,
            obj={"name": "std", "dataset_type": DatasetType.DETECTION, "format": "yolo"},
        )
        assert int(row.standard_dataset_id) == 1
        assert row.storage_path == "standard/1"
    finally:
        db.close()


def test_publish_job_reuses_same_request_even_when_name_changes(monkeypatch, tmp_path: Path) -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    first = svc.create_job(db, 1000001, _payload(name="name-with-timestamp-1"))
    second = svc.create_job(db, 1000001, _payload(name="name-with-timestamp-2"))

    rows = db.query(IllegalDatasetPublishJob).all()
    assert len(rows) == 1
    assert second.job_id == first.job_id
    assert second.reused is True


def test_publish_job_different_conversion_config_gets_new_job(monkeypatch, tmp_path: Path) -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    first = svc.create_job(db, 1000001, _payload())
    second = svc.create_job(
        db,
        1000001,
        _payload(publish_config={"conversion": {"slice": {"enabled": True, "slice_size": 1024}}}),
    )

    assert first.job_id != second.job_id
    assert db.query(IllegalDatasetPublishJob).count() == 2


def test_publish_job_mapping_delete_change_gets_new_job(monkeypatch, tmp_path: Path) -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    first = svc.create_job(
        db,
        1000001,
        _payload(label_mapping_overrides={"raw": {"mapped_label": "mapped", "status": "keep"}}),
    )
    second = svc.create_job(
        db,
        1000001,
        _payload(label_mapping_overrides={"raw": {"mapped_label": "", "status": "delete"}}),
    )

    assert first.job_id != second.job_id
    assert db.query(IllegalDatasetPublishJob).count() == 2


def test_publish_job_snapshot_keeps_saved_parent_delete_mapping() -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    db.add(
        IllegalDatasetLabelMapping(
            illegal_dataset_id=1000001,
            raw_label="车辆",
            mapped_label="",
            status="delete",
        )
    )
    db.commit()

    snapshot = IllegalDatasetPublishJobService()._effective_mapping_snapshot(
        db,
        1000001,
        {"label_mapping_overrides": {}},
    )

    assert snapshot == {"车辆": "__DISCARD__"}


def test_completed_publish_job_returns_existing_result(monkeypatch, tmp_path: Path) -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    first = svc.create_job(db, 1000001, _payload())
    row = db.query(IllegalDatasetPublishJob).filter_by(job_id=first.job_id).one()
    row.status = "completed"
    row.phase = "done"
    row.progress = 100
    row.standard_dataset_id = 2000001
    row.result = {
        "standard_dataset_id": 2000001,
        "name": "published",
        "source_illegal_dataset_id": 1000001,
        "source_illegal_version_id": 1,
        "publish_config": {},
    }
    row.finished_at = datetime.utcnow()
    db.commit()

    second = svc.create_job(db, 1000001, _payload(name="new-timestamp"))

    assert second.job_id == first.job_id
    assert second.status == "completed"
    assert second.result is not None
    assert second.result.standard_dataset_id == 2000001


def test_failed_publish_job_is_reset_for_retry(monkeypatch, tmp_path: Path) -> None:
    db = _make_db()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    first = svc.create_job(db, 1000001, _payload())
    row = db.query(IllegalDatasetPublishJob).filter_by(job_id=first.job_id).one()
    row.status = "failed"
    row.phase = "failed"
    row.progress = 100
    row.error_message = "boom"
    row.logs = ["boom"]
    db.commit()

    retry = svc.create_job(db, 1000001, _payload())

    assert retry.job_id == first.job_id
    assert retry.status == "queued"
    assert retry.error_message is None
    assert retry.progress == 0


def test_get_active_publish_job_returns_latest_running_job(monkeypatch, tmp_path: Path) -> None:
    factory = _make_db_factory()
    db = factory()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_job_service.SessionLocal",
        factory,
    )
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    older = svc.create_job(db, 1000001, _payload(publish_config={"conversion": {"slice": {"enabled": False}}}))
    newer = svc.create_job(
        db,
        1000001,
        _payload(publish_config={"conversion": {"slice": {"enabled": True, "slice_size": 1024}}}),
    )
    base = datetime.utcnow()
    older_row = db.query(IllegalDatasetPublishJob).filter_by(job_id=older.job_id).one()
    newer_row = db.query(IllegalDatasetPublishJob).filter_by(job_id=newer.job_id).one()
    older_row.status = "running"
    older_row.phase = "converting"
    older_row.progress = 30
    older_row.updated_at = base
    newer_row.status = "running"
    newer_row.phase = "publishing"
    newer_row.progress = 80
    newer_row.updated_at = base + timedelta(seconds=10)
    db.commit()

    active = svc.get_active_job(1000001)

    assert active is not None
    assert active.job_id == newer.job_id
    assert active.progress == 80
    db.close()


def test_get_active_publish_job_returns_latest_queued_job(monkeypatch, tmp_path: Path) -> None:
    factory = _make_db_factory()
    db = factory()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_job_service.SessionLocal",
        factory,
    )
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    job = svc.create_job(db, 1000001, _payload())
    row = db.query(IllegalDatasetPublishJob).filter_by(job_id=job.job_id).one()
    row.status = "queued"
    row.phase = "queued"
    row.progress = 0
    db.commit()

    active = svc.get_active_job(1000001)

    assert active is not None
    assert active.job_id == job.job_id
    assert active.status == "queued"
    db.close()


def test_get_active_publish_job_ignores_terminal_jobs(monkeypatch, tmp_path: Path) -> None:
    factory = _make_db_factory()
    db = factory()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_job_service.SessionLocal",
        factory,
    )
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    job = svc.create_job(db, 1000001, _payload())
    row = db.query(IllegalDatasetPublishJob).filter_by(job_id=job.job_id).one()
    row.status = "completed"
    row.phase = "done"
    row.progress = 100
    db.commit()

    assert svc.get_active_job(1000001) is None
    db.close()


def test_cancel_running_publish_job_marks_terminal_and_syncs_status(monkeypatch, tmp_path: Path) -> None:
    factory = _make_db_factory()
    db = factory()
    _seed_illegal_dataset(db)
    svc = IllegalDatasetPublishJobService()
    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_job_service.SessionLocal",
        factory,
    )
    monkeypatch.setattr(svc, "jobs_root", lambda dataset_id: tmp_path / "jobs" / str(dataset_id))

    job = svc.create_job(db, 1000001, _payload())
    row = db.query(IllegalDatasetPublishJob).filter_by(job_id=job.job_id).one()
    row.status = "running"
    row.phase = "converting"
    row.progress = 43
    row.processed = 5638
    row.total = 17401
    db.commit()

    cancelled = svc.cancel_job(1000001, job.job_id)

    assert cancelled.status == "cancelled"
    assert cancelled.phase == "cancelled"
    assert cancelled.progress == 100
    assert cancelled.cancel_requested is True
    assert svc.get_active_job(1000001) is None

    status_payload = svc._read_json_retry(
        svc.status_path(1000001, job.job_id),
        missing_message="missing",
    )
    assert status_payload["status"] == "cancelled"
    assert status_payload["cancel_requested"] is True
    db.close()
