from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.db.init_db import _seed_architectures
from train_platform.db.seed_data import DEFAULT_ARCHITECTURES
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.enums import TaskType


class SeedArchitecturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_default_architectures_cover_requested_ultralytics_detection_variants(self) -> None:
        expected_variants = {
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolov9t",
            "yolov9s",
            "yolov9m",
            "yolov9c",
            "yolov9e",
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolo12n",
            "yolo12s",
            "yolo12m",
            "yolo12l",
            "yolo12x",
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
            "rtdetr-l",
            "rtdetr-x",
        }
        ultralytics_rows = [
            row
            for row in DEFAULT_ARCHITECTURES
            if row.get("engine") == "ultralytics-yolo" and row.get("task_type") == TaskType.DETECTION
        ]
        actual_variants = {str(row.get("variant")) for row in ultralytics_rows}
        self.assertTrue(expected_variants.issubset(actual_variants))

        dedupe_keys = [(str(r.get("family")), str(r.get("variant")), str(r.get("task_type"))) for r in DEFAULT_ARCHITECTURES]
        self.assertEqual(len(dedupe_keys), len(set(dedupe_keys)))

    def test_seed_architectures_is_idempotent(self) -> None:
        _seed_architectures(self.db)
        count_after_first = self.db.query(ModelArchitecture).count()
        _seed_architectures(self.db)
        count_after_second = self.db.query(ModelArchitecture).count()

        self.assertGreater(count_after_first, 0)
        self.assertEqual(count_after_first, count_after_second)

        variants = {row[0] for row in self.db.query(ModelArchitecture.variant).all()}
        self.assertIn("yolo26n", variants)
        self.assertIn("rtdetr-l", variants)


if __name__ == "__main__":
    unittest.main()

