from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    return sessionmaker(bind=engine)()


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
