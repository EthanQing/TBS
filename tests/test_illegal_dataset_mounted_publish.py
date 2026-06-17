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
    IllegalDatasetLabelMapping,
    IllegalDatasetVersion,
)
from train_platform.services.v3 import illegal_dataset_service as service_module
from train_platform.services.v3 import illegal_dataset_cas as cas_module
from train_platform.services.v3.dataset_import_service import DatasetImportService
from train_platform.services.v3.illegal_dataset_cas import load_manifest_token
from train_platform.services.v3.illegal_dataset_publish_service import IllegalDatasetPublishService
from train_platform.services.v3.mounted_dataset_service import build_illegal_mounted_manifest


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


def test_convert_dataset_parallel_remaps_class_ids_after_skipped_pair(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()

    Image.new("RGB", (80, 80), (255, 255, 255)).save(source_root / "bad-first.jpg")
    (source_root / "bad-first.json").write_text(
        json.dumps({"shapes": [{"label": "dropped", "shape_type": "rectangle", "points": [[5, 5], [30, 30]]}]}),
        encoding="utf-8",
    )
    (source_root / "bad-first.jpg").write_bytes((source_root / "bad-first.jpg").read_bytes()[:-11])

    for name, label in (("good-a", "keep-a"), ("good-b", "keep-b"), ("good-c", "keep-a")):
        Image.new("RGB", (80, 80), (255, 255, 255)).save(source_root / f"{name}.jpg")
        (source_root / f"{name}.json").write_text(
            json.dumps({"shapes": [{"label": label, "shape_type": "rectangle", "points": [[10, 10], [40, 40]]}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_service.settings",
        SimpleNamespace(illegal_dataset_publish_max_workers=3),
    )

    result = IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={"dropped": "dropped", "keep-a": "keep-a", "keep-b": "keep-b"},
        publish_config={"conversion": {"slice": {"enabled": False, "output_format": "jpg"}}},
    )

    assert result["pairs_total"] == 4
    assert result["pairs_processed"] == 3
    assert result["pairs_skipped"] == 1
    assert result["class_names"] == ["keep-a", "keep-b"]
    assert yaml.safe_load((output_root / "data.yaml").read_text(encoding="utf-8"))["names"] == ["keep-a", "keep-b"]
    class_ids = {
        int(line.split()[0])
        for label_path in (output_root / "labels").glob("*.txt")
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert class_ids == {0, 1}


def test_convert_dataset_publish_max_workers_one_keeps_serial_outputs(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()

    for idx in range(2):
        Image.new("RGB", (64, 64), (255, 255, 255)).save(source_root / f"sample-{idx}.jpg")
        (source_root / f"sample-{idx}.json").write_text(
            json.dumps({"shapes": [{"label": "car", "shape_type": "rectangle", "points": [[5, 5], [30, 30]]}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "train_platform.services.v3.illegal_dataset_publish_service.settings",
        SimpleNamespace(illegal_dataset_publish_max_workers=1),
    )

    result = IllegalDatasetPublishService().convert_dataset(
        source_root,
        output_root,
        label_mapping={"car": "vehicle"},
        publish_config={"conversion": {"slice": {"enabled": False, "output_format": "jpg"}}},
    )

    assert result["pairs_processed"] == 2
    assert result["class_names"] == ["vehicle"]
    assert len(list((output_root / "images").glob("*.jpg"))) == 2
    assert all(
        line.startswith("0 ")
        for label_path in (output_root / "labels").glob("*.txt")
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


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


def test_build_illegal_mounted_manifest_is_lightweight_for_json(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "imports" / "illegal-json"
    source_root.mkdir(parents=True)
    (source_root / "sample.jpg").write_bytes(b"not-a-real-image")
    (source_root / "sample.json").write_text(
        json.dumps({"shapes": [{"label": "person"}, {"label": "helmet"}]}),
        encoding="utf-8",
    )

    def fail_image_size(_path):
        raise AssertionError("mounted illegal import must not read image dimensions")

    monkeypatch.setattr("train_platform.services.v3.mounted_dataset_service.image_size", fail_image_size)

    progress: list[tuple[int, str, dict]] = []

    manifest = build_illegal_mounted_manifest(
        source_root,
        progress_callback=lambda value, stage, detail: progress.append((value, stage, detail)),
        max_workers=1,
    )

    assert manifest["format"] == "json"
    assert manifest["image_count"] == 1
    assert manifest["json_count"] == 1
    assert manifest["raw_labels"] == ["helmet", "person"]
    assert manifest["object_count"] == 2
    assert "images/source/sample.jpg" in manifest["files"]
    assert "sample.json" in manifest["files"]
    assert not (source_root / "labels").exists()
    assert not (source_root / "data.yaml").exists()
    assert any(stage == "parsing" and detail.get("processed_count") == 1 for _value, stage, detail in progress)


def test_build_illegal_mounted_manifest_parallel_matches_serial(tmp_path: Path) -> None:
    source_root = tmp_path / "imports" / "parallel-json"
    source_root.mkdir(parents=True)
    for idx, label in enumerate(("person", "helmet", "vest", "person"), start=1):
        (source_root / f"sample-{idx}.jpg").write_bytes(b"not-a-real-image")
        (source_root / f"sample-{idx}.json").write_text(
            json.dumps({"shapes": [{"label": label}]}),
            encoding="utf-8",
        )

    serial = build_illegal_mounted_manifest(source_root, max_workers=1)
    parallel_progress: list[tuple[int, str, dict]] = []
    parallel = build_illegal_mounted_manifest(
        source_root,
        max_workers=4,
        progress_callback=lambda value, stage, detail: parallel_progress.append((value, stage, detail)),
    )

    assert parallel["image_paths"] == serial["image_paths"]
    assert parallel["raw_labels"] == serial["raw_labels"]
    assert parallel["object_count"] == serial["object_count"]
    assert parallel["files"] == serial["files"]
    assert any(stage == "parsing" and detail.get("total_count") == 4 for _value, stage, detail in parallel_progress)


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

        def fake_build_illegal_mounted_manifest(source_root: Path, *, prefer_yolo: bool = True, **_kwargs):
            manifest = {
                "source_type": "mounted_dir_link",
                "format": "yolo",
                "source_root": str(source_root),
                "image_count": 1,
                "image_paths": ["images/sample.jpg"],
                "files": {
                    "images/sample.jpg": {
                        "storage": "mounted",
                        "source_path": str(source_root / "sample.jpg"),
                        "size": 1,
                        "mtime": 1.0,
                    }
                },
            }
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
        monkeypatch.setattr(cas_module, "resolve_storage_token", lambda token: storage_root / str(token))
        monkeypatch.setattr(service_module, "build_illegal_mounted_manifest", fake_build_illegal_mounted_manifest)
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


def test_mounted_json_import_records_manifest_labels_and_images(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    for table in (
        IllegalDataset.__table__,
        IllegalDatasetVersion.__table__,
        IllegalDatasetEvent.__table__,
        IllegalDatasetImage.__table__,
        IllegalDatasetLabelMapping.__table__,
    ):
        table.create(engine, checkfirst=True)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        dataset = IllegalDataset(
            illegal_dataset_id=1000005,
            name="mounted json",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="illegal/1000005",
        )
        db.add(dataset)
        db.commit()

        storage_root = tmp_path / "datasets"
        source_root = tmp_path / "imports" / "train"
        source_root.mkdir(parents=True)
        (source_root / "sample.jpg").write_bytes(b"fake-image")
        (source_root / "sample.json").write_text(
            json.dumps({"shapes": [{"label": "person"}, {"label": "helmet"}]}),
            encoding="utf-8",
        )
        progress: list[tuple[int, str, dict]] = []

        svc = service_module.IllegalDatasetService()
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
        monkeypatch.setattr(cas_module, "resolve_storage_token", lambda token: storage_root / str(token))

        result = svc.import_mounted_source_tree(
            db,
            1000005,
            source_root,
            progress_callback=lambda value, stage, detail=None: progress.append((value, stage, detail or {})),
        )

        version = db.query(IllegalDatasetVersion).filter(IllegalDatasetVersion.version_id == result.active_version_id).one()
        manifest = load_manifest_token(str(version.manifest_path))
        assert manifest["format"] == "json"
        assert manifest["raw_labels"] == ["helmet", "person"]
        assert manifest["stats"]["image_count"] == 1
        assert manifest["stats"]["object_count"] == 2
        assert db.query(IllegalDatasetImage).filter(IllegalDatasetImage.version_id == version.version_id).count() == 1
        assert not (storage_root / "illegal" / "1000005" / "labels").exists()
        assert not (storage_root / "illegal" / "1000005" / "data.yaml").exists()
        assert "indexing" in [stage for _value, stage, _detail in progress]
        assert any(detail.get("processed_count") == 1 and detail.get("total_count") == 1 for _value, _stage, detail in progress)
        assert svc.get_raw_labels(db, 1000005) == ["helmet", "person"]
    finally:
        db.close()


def test_activate_mounted_lightweight_version_does_not_materialize_files(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    for table in (
        IllegalDataset.__table__,
        IllegalDatasetVersion.__table__,
        IllegalDatasetEvent.__table__,
        IllegalDatasetImage.__table__,
        IllegalDatasetLabelMapping.__table__,
    ):
        table.create(engine, checkfirst=True)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        dataset = IllegalDataset(
            illegal_dataset_id=1000006,
            name="activate mounted",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="illegal/1000006",
        )
        db.add(dataset)
        db.flush()

        storage_root = tmp_path / "datasets"
        source_root = tmp_path / "imports" / "train"
        source_root.mkdir(parents=True)
        image_path = source_root / "sample.jpg"
        image_path.write_bytes(b"fake-image")
        manifest_path = storage_root / "illegal" / ".versions" / "1000006" / "v1.manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": {
                        "images/source/sample.jpg": {
                            "storage": "mounted",
                            "source_path": str(image_path),
                            "size": 10,
                            "mtime": 1.0,
                        }
                    },
                    "stats": {"image_count": 1},
                }
            ),
            encoding="utf-8",
        )
        version = IllegalDatasetVersion(
            illegal_dataset_id=1000006,
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="illegal/.versions/1000006/v1.manifest.json",
            meta={"source_type": "mounted_dir_link", "lightweight_import": True},
        )
        db.add(version)
        db.commit()

        svc = service_module.IllegalDatasetService()
        monkeypatch.setattr(svc, "_root_path", lambda dataset: storage_root / str(dataset.storage_path))
        monkeypatch.setattr(service_module, "resolve_storage_token", lambda token: storage_root / str(token))
        monkeypatch.setattr(cas_module, "resolve_storage_token", lambda token: storage_root / str(token))

        svc.activate_version(db, 1000006, int(version.version_id))

        active_root = storage_root / "illegal" / "1000006"
        assert active_root.exists()
        assert not (active_root / "images" / "source" / "sample.jpg").exists()
    finally:
        db.close()
