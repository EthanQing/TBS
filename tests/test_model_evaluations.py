from __future__ import annotations

from types import SimpleNamespace

import pytest

from train_platform.models.v3.enums import DatasetSplit
from train_platform.models.v3.standard_dataset import StandardDatasetImage
from train_platform.services.v3.model_evaluation_metrics import box_iou, compute_detection_metrics
from train_platform.services.v3.model_evaluation_service import ModelEvaluationService


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


def test_get_active_job_returns_none_without_running_status(tmp_path, monkeypatch) -> None:
    svc = ModelEvaluationService()
    monkeypatch.setattr(svc, "jobs_root", lambda: tmp_path)

    assert svc.get_active_job() is None
