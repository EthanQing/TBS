from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train_platform.services.dataset_service import DatasetService
from train_platform.models.enums import DatasetType


class DatasetAnnotationHelpersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = DatasetService()

    def test_guess_label_rel_path_from_image(self) -> None:
        self.assertEqual(
            self.svc._guess_label_rel_path_from_image("images/train/example.jpg"),
            "labels/train/example.txt",
        )
        self.assertEqual(
            self.svc._guess_label_rel_path_from_image("train/example.jpg"),
            "labels/train/example.txt",
        )

    def test_parse_yolo_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            label_path = Path(td) / "sample.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            boxes = self.svc._parse_yolo_boxes(
                label_path,
                width=100,
                height=50,
                class_names=["car"],
            )

        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertEqual(box["class_id"], 0)
        self.assertEqual(box["class_name"], "car")
        self.assertEqual(box["x1"], 40.0)
        self.assertEqual(box["y1"], 15.0)
        self.assertEqual(box["x2"], 60.0)
        self.assertEqual(box["y2"], 35.0)

    def test_build_view_index_payload_detection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (root / "images" / "train" / "a.jpg").write_bytes(b"fake")
            (root / "images" / "train" / "b.jpg").write_bytes(b"fake")
            (root / "labels" / "train" / "a.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
                encoding="utf-8",
            )

            payload = self.svc._build_view_index_payload(
                dataset_path=root,
                image_rels=["images/train/a.jpg", "images/train/b.jpg"],
                dataset_type=DatasetType.DETECTION,
            )

        self.assertEqual(len(payload["images"]), 2)
        first = payload["images"][0]
        second = payload["images"][1]
        self.assertEqual(first["rel"], "images/train/a.jpg")
        self.assertEqual(first["object_count"], 2)
        self.assertEqual(first["classes"], [0, 1])
        self.assertEqual(second["object_count"], 0)
        self.assertEqual(second["classes"], [])
        self.assertEqual(payload["class_image_count"], {"0": 1, "1": 1})

    def test_thumbnail_static_url_uses_version_prefix_for_snapshot(self) -> None:
        version = SimpleNamespace(version_id=7, snapshot_path=".versions/demo/v7/snapshot")
        url = self.svc._thumbnail_static_url(dataset_id=3, version=version, rel="images/train/a.jpg")
        self.assertEqual(url, "/static/thumbnails/3/v7/images/train/a.webp")


if __name__ == "__main__":
    unittest.main()
