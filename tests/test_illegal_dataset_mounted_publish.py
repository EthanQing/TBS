from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image
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


def test_convert_dataset_skips_truncated_image_and_keeps_valid_pairs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()

    good_image = source_root / "good.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(good_image)
    good_json = {
        "shapes": [
            {
                "label": "car",
                "shape_type": "rectangle",
                "points": [[5, 5], [40, 40]],
            }
        ]
    }
    (source_root / "good.json").write_text(json.dumps(good_json), encoding="utf-8")

    bad_image = source_root / "bad.jpg"
    bad_image.write_bytes(good_image.read_bytes()[:-13])
    (source_root / "bad.json").write_text(json.dumps(good_json), encoding="utf-8")

    events: list[tuple[str, dict]] = []
    result = IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={"car": "car"},
        publish_config={"conversion": {"slice": {"enabled": False, "output_format": "jpg"}}},
        progress_callback=lambda phase, info: events.append((phase, info)),
    )

    assert result["pairs_total"] == 2
    assert result["pairs_processed"] == 1
    assert result["pairs_skipped"] == 1
    assert any("bad.jpg / bad.json" in item for item in result["warnings"])
    assert any("跳过 bad.jpg / bad.json" in str(info.get("message", "")) for _phase, info in events)
    assert len(list((output_root / "images").glob("*.jpg"))) == 1


def test_convert_dataset_excludes_deleted_mappings_from_yolo_outputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()

    Image.new("RGB", (100, 100), (255, 255, 255)).save(source_root / "sample.jpg")
    annotation = {
        "shapes": [
            {"label": "keep", "shape_type": "rectangle", "points": [[5, 5], [30, 30]]},
            {"label": "drop_status", "shape_type": "rectangle", "points": [[35, 35], [60, 60]]},
            {"label": "drop_sentinel", "shape_type": "rectangle", "points": [[65, 65], [90, 90]]},
        ]
    }
    (source_root / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    result = IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={
            "keep": "vehicle",
            "drop_status": "",
            "drop_sentinel": "__DISCARD__",
        },
        publish_config={"slice": {"enabled": False, "output_format": "jpg"}},
    )

    assert result["class_names"] == ["vehicle"]
    assert (output_root / "classes.txt").read_text(encoding="utf-8").splitlines() == ["vehicle"]
    data_yaml = yaml.safe_load((output_root / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == 1
    assert data_yaml["names"] == ["vehicle"]
    label_lines = list((output_root / "labels").glob("*.txt"))[0].read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")


def test_convert_dataset_parent_delete_excludes_descendant_labels(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()

    Image.new("RGB", (100, 100), (255, 255, 255)).save(source_root / "sample.jpg")
    annotation = {
        "shapes": [
            {"label": "keep", "shape_type": "rectangle", "points": [[5, 5], [30, 30]]},
            {"label": "车辆%轿车", "shape_type": "rectangle", "points": [[35, 35], [60, 60]]},
        ]
    }
    (source_root / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    result = IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={
            "keep": "vehicle",
            "车辆": "__DISCARD__",
        },
        publish_config={"conversion": {"slice": {"enabled": False, "output_format": "jpg", "label_separator": "%"}}},
    )

    assert result["class_names"] == ["vehicle"]
    assert (output_root / "classes.txt").read_text(encoding="utf-8").splitlines() == ["vehicle"]
    data_yaml = yaml.safe_load((output_root / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == ["vehicle"]
    label_lines = list((output_root / "labels").glob("*.txt"))[0].read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")


def _convert_single_annotation(tmp_path: Path, annotation: dict, dirname: str) -> list[float]:
    source_root = tmp_path / dirname / "source"
    output_root = tmp_path / dirname / "output"
    source_root.mkdir(parents=True)
    Image.new("RGB", (100, 100), (255, 255, 255)).save(source_root / "sample.jpg")
    (source_root / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")

    IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={"car": "car"},
        publish_config={"conversion": {"slice": {"enabled": False, "output_format": "jpg"}}},
    )

    label_line = list((output_root / "labels").glob("*.txt"))[0].read_text(encoding="utf-8").strip()
    return [float(item) for item in label_line.split()]


def test_convert_dataset_version_1_converts_bottom_left_points(tmp_path: Path) -> None:
    annotation = {
        "version": 1,
        "shapes": [
            {"label": "car", "shape_type": "rectangle", "points": [[10, 10], [30, 30]]},
        ],
    }

    class_id, cx, cy, width, height = _convert_single_annotation(tmp_path, annotation, "v1")

    assert class_id == 0
    assert cx == 0.2
    assert cy == 0.8
    assert width == 0.2
    assert height == 0.2

    annotation["version"] = "1"
    _class_id, _cx, string_version_cy, _width, _height = _convert_single_annotation(
        tmp_path,
        annotation,
        "v1_string",
    )
    assert string_version_cy == 0.8


def test_convert_dataset_version_1_float_converts_bottom_left_points(tmp_path: Path) -> None:
    annotation = {
        "version": 1.0,
        "shapes": [
            {"label": "car", "shape_type": "rectangle", "points": [[10, 10], [30, 30]]},
        ],
    }

    class_id, cx, cy, width, height = _convert_single_annotation(tmp_path, annotation, "v1_float")

    assert class_id == 0
    assert cx == 0.2
    assert cy == 0.8
    assert width == 0.2
    assert height == 0.2


def test_convert_dataset_newer_version_keeps_top_left_points(tmp_path: Path) -> None:
    annotation = {
        "version": "5.0.1",
        "shapes": [
            {"label": "car", "shape_type": "rectangle", "points": [[10, 10], [30, 30]]},
        ],
    }

    class_id, cx, cy, width, height = _convert_single_annotation(tmp_path, annotation, "v5")

    assert class_id == 0
    assert cx == 0.2
    assert cy == 0.2
    assert width == 0.2
    assert height == 0.2


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

    fake_db = SimpleNamespace(commit=lambda: None, refresh=lambda _row: None)
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
        (source_root / "sample.jpg").write_bytes(b"fake-source-image")

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
            "illegal_manifest_path",
            lambda dataset_id, version: storage_root / "illegal" / ".versions" / str(int(dataset_id)) / f"v{int(version)}.manifest.json",
        )
        monkeypatch.setattr(
            service_module,
            "to_storage_token",
            lambda path: Path(path).resolve(strict=False).relative_to(storage_root.resolve(strict=False)).as_posix(),
        )
        monkeypatch.setattr(service_module, "resolve_storage_token", lambda token: storage_root / str(token))
        monkeypatch.setattr(service_module, "link_source_tree", fake_link_source_tree)
        monkeypatch.setattr(svc, "_refresh_version_raw_labels_cache", lambda *_args, **_kwargs: [])

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
