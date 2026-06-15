from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from train_platform.models.v3.enums import DatasetSplit
from train_platform.models.v3.standard_dataset import StandardDatasetImage
from train_platform.services.v3.model_evaluation_metrics import box_iou, compute_detection_metrics
from train_platform.services.v3.model_evaluation_service import ModelEvaluationService
from train_platform.workers.inference_worker import _extract_ultralytics_val_metrics


def test_box_iou_identical_and_disjoint() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_detection_metrics_precision_recall_f1_and_map() -> None:
    gts = {
        "a.jpg": [{"class_id": 0, "class_name": "person", "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
        "b.jpg": [{"class_id": 0, "class_name": "person", "x1": 20, "y1": 20, "x2": 40, "y2": 40}],
    }
    preds = {
        "a.jpg": [
            {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [0, 0, 10, 10]},
            {"class_id": 0, "class_name": "person", "confidence": 0.4, "xyxy": [50, 50, 60, 60]},
        ],
        "b.jpg": [],
    }

    metrics = compute_detection_metrics(gts, preds, iou_threshold=0.5, class_names=["person"])

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["map50"] > 0
    assert metrics["map50_95"] > 0
    assert metrics["class_metrics"][0]["class_name"] == "person"


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def filter(self, *args, **_kwargs):
        self.filters.extend(args)
        for arg in args:
            text = str(arg)
            if "standard_dataset_id" in text and " = " in text:
                continue
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def all(self):
        out = self.rows
        # SQLAlchemy expressions are not evaluated in this fake query. Use the
        # split filters that the service applies by inspecting the requested
        # enum value in the string form, which is stable enough for this unit.
        filter_text = " ".join(str(x) for x in self.filters)
        if "test" in filter_text:
            out = [r for r in out if r.split == DatasetSplit.TEST]
        elif "val" in filter_text:
            out = [r for r in out if r.split == DatasetSplit.VAL]
        elif "train" in filter_text:
            out = [r for r in out if r.split == DatasetSplit.TRAIN]
        return out


class FakeDb:
    def __init__(self, rows):
        self.rows = rows

    def query(self, model):
        if model is StandardDatasetImage:
            return FakeQuery(self.rows)
        return FakeQuery([])


def _image(path: str, split):
    return SimpleNamespace(path=path, split=split, standard_dataset_id=1001)


def test_select_image_rows_all_keeps_unsplit_uploaded_test_dataset() -> None:
    rows = [
        _image("images/a.jpg", None),
        _image("images/b.jpg", DatasetSplit.TEST),
        _image("images/c.jpg", DatasetSplit.VAL),
    ]
    svc = ModelEvaluationService()
    dataset = SimpleNamespace(standard_dataset_id=1001)

    selected = svc._select_image_rows(FakeDb(rows), dataset, scope="all")

    assert [r.path for r in selected] == ["images/a.jpg", "images/b.jpg", "images/c.jpg"]


def test_select_labeled_image_rows_skips_missing_and_empty_labels(tmp_path) -> None:
    root = tmp_path
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "images" / "ok.jpg").write_bytes(b"x")
    (root / "labels" / "ok.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "images" / "empty.jpg").write_bytes(b"x")
    (root / "labels" / "empty.txt").write_text("\n", encoding="utf-8")
    (root / "images" / "missing.jpg").write_bytes(b"x")
    svc = ModelEvaluationService()

    labeled, skipped = svc._select_labeled_image_rows(
        root,
        [
            _image("images/ok.jpg", None),
            _image("images/empty.jpg", None),
            _image("images/missing.jpg", None),
        ],
    )

    assert [r.path for r in labeled] == ["images/ok.jpg"]
    assert skipped == 2


def test_write_ultralytics_eval_data_yaml_uses_labeled_images(tmp_path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"x")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    svc = ModelEvaluationService()
    monkeypatch.setattr(svc, "jobs_root", lambda: tmp_path / "jobs")

    data_yaml = svc._write_ultralytics_eval_data_yaml("job-yolo", root, [_image("images/a.jpg", None)], ["person"])

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    eval_list = data_yaml.parent / "eval_images.txt"
    assert payload["val"] == eval_list.resolve(strict=False).as_posix()
    assert payload["names"] == ["person"]
    assert eval_list.read_text(encoding="utf-8").strip().endswith("images/a.jpg")


def test_get_active_job_returns_none_without_running_status(tmp_path, monkeypatch) -> None:
    svc = ModelEvaluationService()
    monkeypatch.setattr(svc, "jobs_root", lambda: tmp_path)

    assert svc.get_active_job() is None


def test_cancelled_evaluation_job_cannot_be_reactivated(tmp_path, monkeypatch) -> None:
    svc = ModelEvaluationService()
    monkeypatch.setattr(svc, "jobs_root", lambda: tmp_path)
    job_id = "job-cancel"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "job_id": job_id,
        "status": "running",
        "phase": "inferring",
        "progress": 20,
        "processed": 1,
        "total": 5,
        "seq": 1,
        "last_result_id": 0,
        "standard_dataset_id": 1001,
        "scope": "all",
        "conf": 0.25,
        "iou": 0.5,
        "cancel_requested": False,
        "result": {"metrics": None},
    }
    (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

    cancelled = svc.cancel_job(job_id)
    svc._update_status_if_not_terminal(
        job_id,
        {"status": "completed", "phase": "done", "progress": 100},
        bump_seq=True,
    )

    assert cancelled.status == "cancelled"
    final = svc.get_job(job_id, include_items=False)
    assert final.status == "cancelled"
    assert final.phase == "cancelled"
    assert final.cancel_requested is True


class _FakeBoxMetrics:
    mp = 0.75
    mr = 0.6
    map50 = 0.7
    map = 0.55
    ap_class_index = [0]
    p = [0.75]
    r = [0.6]
    maps = [0.55]
    all_ap = [[0.7, 0.6]]


class _FakeValResults:
    box = _FakeBoxMetrics()
    names = {0: "person"}
    nt_per_class = [10]


def test_extract_ultralytics_val_metrics() -> None:
    metrics = _extract_ultralytics_val_metrics(_FakeValResults(), elapsed_ms=1234.5)

    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.6)
    assert metrics["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert metrics["map50"] == pytest.approx(0.7)
    assert metrics["map50_95"] == pytest.approx(0.55)
    assert metrics["total_targets"] == 10
    assert metrics["class_metrics"][0]["class_name"] == "person"
