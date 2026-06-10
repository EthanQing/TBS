from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from train_platform.services.v3.standard_dataset_service import StandardDatasetService


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def offset(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def all(self):
        return self.rows


class FakeDb:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *_args, **_kwargs):
        return FakeQuery(self.rows)


def _dataset(**overrides):
    now = datetime(2024, 1, 1, 12, 0, 0)
    data = {
        "standard_dataset_id": 1001,
        "name": "traffic-signs",
        "dataset_type": "detection",
        "format": "yolo",
        "storage_path": "standard/1001",
        "description": None,
        "source_type": None,
        "publish_config": None,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_list_datasets_lightweight_skips_statistics_and_preview(monkeypatch) -> None:
    svc = StandardDatasetService()
    db = FakeDb([_dataset()])

    def fail_build_statistics(*_args, **_kwargs):
        raise AssertionError("statistics should not be built for lightweight lists")

    def fail_preview(*_args, **_kwargs):
        raise AssertionError("preview should not be loaded for lightweight lists")

    monkeypatch.setattr(svc, "_build_dataset_statistics", fail_build_statistics)
    monkeypatch.setattr(svc, "_first_image_preview_url", fail_preview)

    items = svc.list_datasets(db, include_statistics=False)

    assert len(items) == 1
    assert items[0]["standard_dataset_id"] == 1001
    assert items[0]["name"] == "traffic-signs"
    assert items[0]["statistics"] is None
    assert items[0]["preview_image_url"] is None


def test_list_datasets_default_includes_statistics_and_preview(monkeypatch) -> None:
    svc = StandardDatasetService()
    db = FakeDb([_dataset()])

    monkeypatch.setattr(svc, "_build_dataset_statistics", lambda *_args: {"num_images": 2})
    monkeypatch.setattr(svc, "_first_image_preview_url", lambda *_args, **_kwargs: "/thumb.jpg")

    items = svc.list_datasets(db)

    assert len(items) == 1
    assert items[0]["statistics"] == {"num_images": 2}
    assert items[0]["preview_image_url"] == "/thumb.jpg"
