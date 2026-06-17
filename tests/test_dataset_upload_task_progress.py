from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.models.v3.dataset_upload import DatasetUploadTask
from train_platform.services.v3 import dataset_upload_service as service_module


def test_import_progress_callback_updates_detail_fields_at_same_percent(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    DatasetUploadTask.__table__.create(engine, checkfirst=True)
    TestSessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(service_module, "SessionLocal", TestSessionLocal)

    with TestSessionLocal() as db:
        task = DatasetUploadTask(
            task_id="task-progress",
            dataset_kind="illegal",
            dataset_id=1,
            mode="upload",
            source_path=str(tmp_path),
            source_type="dir_link",
            status="running",
            stage="scanning",
            progress=10,
        )
        db.add(task)
        db.commit()

    callback = service_module.DatasetUploadService()._make_import_progress_callback("task-progress")
    callback(60, "parsing", {"processed_count": 1, "total_count": 2, "current_item": "a.json", "detail_message": "Parsed 1/2 JSON files"})
    callback(60, "parsing", {"processed_count": 2, "total_count": 2, "current_item": "b.json", "detail_message": "Parsed 2/2 JSON files"})

    with TestSessionLocal() as db:
        task = db.query(DatasetUploadTask).filter(DatasetUploadTask.task_id == "task-progress").one()
        assert task.progress == 60
        assert task.stage == "parsing"
        assert task.processed_count == 2
        assert task.total_count == 2
        assert task.current_item == "b.json"
        assert task.detail_message == "Parsed 2/2 JSON files"
