from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.models.v3.enums import DatasetType, DatasetVersionStatus
from train_platform.models.v3.illegal_dataset import (
    IllegalDataset,
    IllegalDatasetEvent,
    IllegalDatasetImage,
    IllegalDatasetVersion,
)
from train_platform.services.v3 import illegal_dataset_service as service_module
from train_platform.services.v3.dataset_import_service import DatasetImportService
from train_platform.services.v3.illegal_dataset_publish_service import IllegalDatasetPublishService


def test_collect_pairs_ignores_mounted_manifest(tmp_path: Path) -> None:
    (tmp_path / ".mounted_manifest.json").write_text(
        '{"source_type":"mounted_dir_link","format":"json"}',
        encoding="utf-8",
    )

    pairs, warnings, unmatched_files = IllegalDatasetPublishService()._collect_pairs(tmp_path)

    assert pairs == []
    assert warnings == []
    assert unmatched_files == []


def test_publish_uses_original_source_for_mounted_json_versions(tmp_path: Path, monkeypatch) -> None:
    mounted_source = tmp_path / "imports" / "illegal-json"
    mounted_source.mkdir(parents=True)
    version = SimpleNamespace(
        version_id=20,
        version=1,
        meta={
            "source_type": "mounted_dir_link",
            "format": "json",
            "source_root": str(mounted_source),
        },
    )
    dataset = SimpleNamespace(
        illegal_dataset_id=10,
        name="mounted illegal",
        dataset_type="detection",
    )
    backend_temp = tmp_path / "backend-temp"
    backend_temp.mkdir()
    captured: dict[str, Path] = {}

    class FakePublishService:
        def convert_dataset(self, source_root, output_root, **_kwargs):
            captured["source_root"] = Path(source_root).resolve(strict=False)
            Path(output_root).mkdir(parents=True, exist_ok=True)
            return {
                "pairs_total": 1,
                "pairs_processed": 1,
                "pairs_skipped": 0,
                "skipped_details": [],
                "warnings": [],
                "class_names": ["mapped"],
                "stats": {"images": 1, "slices": 1, "labels": 1, "empty_slices": 0},
                "split_summary": None,
                "normalized_slice_config": {"enabled": False},
            }

    class FakeStandardDatasetService:
        def materialize_from_source_tree(self, *_args, name: str, **_kwargs):
            return SimpleNamespace(standard_dataset_id=30, name=name)

    fake_db = SimpleNamespace(commit=lambda: None)
    svc = service_module.IllegalDatasetService()
    monkeypatch.setattr(svc, "get_dataset", lambda _db, _dataset_id: dataset)
    monkeypatch.setattr(svc, "_selected_version", lambda _db, _row, version_id=None: version)
    monkeypatch.setattr(svc, "get_label_mappings", lambda _db, _dataset_id: [])
    monkeypatch.setattr(svc, "_add_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service_module, "illegal_dataset_temp_root", lambda: backend_temp)
    monkeypatch.setattr(DatasetImportService, "allowed_roots", lambda _self: (tmp_path / "imports",))
    monkeypatch.setattr(service_module, "IllegalDatasetPublishService", FakePublishService)
    monkeypatch.setattr(
        "train_platform.services.v3.standard_dataset_service.StandardDatasetService",
        FakeStandardDatasetService,
    )

    result = svc.publish_standard_dataset(
        fake_db,
        10,
        obj={"name": "published mounted", "publish_config": {}, "split": {}},
    )

    assert captured["source_root"] == mounted_source.resolve(strict=False)
    assert result["standard_dataset_id"] == 30


def test_mounted_append_uses_next_dataset_version(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    for table in (
        IllegalDataset.__table__,
        IllegalDatasetVersion.__table__,
        IllegalDatasetEvent.__table__,
        IllegalDatasetImage.__table__,
    ):
        table.create(engine, checkfirst=True)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        dataset = IllegalDataset(
            illegal_dataset_id=1000004,
            name="mounted append",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="illegal/1000004",
        )
        db.add(dataset)
        db.flush()
        first_version = IllegalDatasetVersion(
            illegal_dataset_id=1000004,
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            snapshot_path="illegal/.versions/1000004/v1",
        )
        db.add(first_version)
        db.flush()
        dataset.active_version_id = int(first_version.version_id)
        db.commit()

        storage_root = tmp_path / "datasets"
        source_root = tmp_path / "imports" / "train"
        source_root.mkdir(parents=True)

        def fake_link_source_tree(target_root: Path, source_root: Path, *, prefer_yolo: bool = True):
            target_root.mkdir(parents=True, exist_ok=True)
            (target_root / "images").mkdir(parents=True, exist_ok=True)
            (target_root / "images" / "sample.jpg").write_bytes(b"fake")
            manifest = {
                "source_type": "mounted_dir_link",
                "format": "yolo",
                "source_root": str(source_root),
                "link_type": "copy",
                "image_count": 1,
                "image_paths": ["images/sample.jpg"],
            }
            import json

            (target_root / ".mounted_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        svc = service_module.IllegalDatasetService()
        monkeypatch.setattr(
            svc,
            "_version_root",
            lambda dataset_id, version: storage_root / "illegal" / ".versions" / str(int(dataset_id)) / f"v{int(version)}",
        )
        monkeypatch.setattr(svc, "_root_path", lambda dataset: storage_root / str(dataset.storage_path))
        monkeypatch.setattr(
            service_module,
            "to_storage_token",
            lambda path: Path(path).resolve(strict=False).relative_to(storage_root.resolve(strict=False)).as_posix(),
        )
        monkeypatch.setattr(service_module, "resolve_storage_token", lambda token: storage_root / str(token))
        monkeypatch.setattr(service_module, "link_source_tree", fake_link_source_tree)
        monkeypatch.setattr(svc, "_refresh_version_raw_labels_cache", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(svc, "_refresh_version_statistics_cache", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(svc, "_refresh_version_view_index_cache", lambda *_args, **_kwargs: {})

        result = svc.import_mounted_source_tree(db, 1000004, source_root, append=True, filename="train")

        versions = (
            db.query(IllegalDatasetVersion)
            .filter(IllegalDatasetVersion.illegal_dataset_id == 1000004)
            .order_by(IllegalDatasetVersion.version)
            .all()
        )
        assert [int(item.version) for item in versions] == [1, 2]
        assert int(result.active_version_id) == int(versions[-1].version_id)
        assert versions[-1].parent_version_id == int(first_version.version_id)
    finally:
        db.close()
