from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService


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
        "illegal_dataset_id": 2001,
        "name": "raw-traffic-signs",
        "dataset_type": "detection",
        "format": "json",
        "storage_path": "illegal/2001",
        "description": None,
        "active_version_id": 3001,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_list_datasets_lightweight_skips_statistics(monkeypatch) -> None:
    svc = IllegalDatasetService()
    db = FakeDb([_dataset()])

    def fail_build_statistics(*_args, **_kwargs):
        raise AssertionError("statistics should not be built for lightweight lists")

    monkeypatch.setattr(svc, "_build_dataset_statistics", fail_build_statistics)

    items = svc.list_datasets(db, include_statistics=False)

    assert len(items) == 1
    assert items[0]["illegal_dataset_id"] == 2001
    assert items[0]["name"] == "raw-traffic-signs"
    assert items[0]["statistics"] is None
    assert items[0]["preview_image_url"] is None


def test_list_datasets_default_includes_statistics(monkeypatch) -> None:
    svc = IllegalDatasetService()
    db = FakeDb([_dataset()])

    monkeypatch.setattr(svc, "_build_dataset_statistics", lambda *_args: {"num_images": 3})

    items = svc.list_datasets(db)

    assert len(items) == 1
    assert items[0]["statistics"] == {"num_images": 3}
    assert items[0]["preview_image_url"] is None
