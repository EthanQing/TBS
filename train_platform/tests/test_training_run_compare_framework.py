from __future__ import annotations

import unittest
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus, TaskType, TrainingRunStatus
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun, TrainingRunResult
from train_platform.services.training_run_service import FrameworkCompareConflict, TrainingRunService


class TrainingRunCompareFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.service = TrainingRunService()
        self.project, self.dataset_version = self._seed_project()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _seed_project(self) -> tuple[Project, DatasetVersion]:
        ds = Dataset(
            name="compare-dataset",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="datasets/compare",
        )
        self.db.add(ds)
        self.db.flush()

        dsv = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="datasets/compare/manifest.json",
        )
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = int(dsv.version_id)

        project = Project(
            name="compare-project",
            dataset_id=int(ds.dataset_id),
            task_type=TaskType.DETECTION,
            is_active=True,
        )
        self.db.add(project)
        self.db.commit()
        return project, dsv

    def _create_run(self, *, engine: str, family: str, variant: str, inference_time_ms: float | None = None) -> TrainingRun:
        arch = ModelArchitecture(
            family=family,
            variant=variant,
            task_type=TaskType.DETECTION,
            engine=engine,
        )
        self.db.add(arch)
        self.db.flush()

        run = TrainingRun(
            run_id=str(uuid4()),
            project_id=int(self.project.project_id),
            dataset_version_id=int(self.dataset_version.version_id),
            architecture_id=int(arch.architecture_id),
            name=f"run-{variant}",
            status=TrainingRunStatus.COMPLETED,
            progress=100,
            current_epoch=10,
            total_epochs=10,
        )
        self.db.add(run)
        self.db.flush()

        if inference_time_ms is not None:
            result = TrainingRunResult(
                run_id=str(run.run_id),
                inference_time_ms=float(inference_time_ms),
            )
            self.db.add(result)

        self.db.commit()
        return run

    def test_compare_same_paddle_framework_success(self) -> None:
        r1 = self._create_run(engine="paddle-det", family="PP-YOLOE", variant="ppyoloe_s")
        r2 = self._create_run(engine="paddle-det", family="PicoDet", variant="picodet_s")

        out = self.service.compare_runs(self.db, [r1.run_id, r2.run_id])
        self.assertEqual(len(out["runs"]), 2)
        for item in out["runs"]:
            self.assertEqual(item["framework_key"], "paddle")
            self.assertEqual(item["framework_label"], "Paddle")
            self.assertEqual(item["engine"], "paddle-det")
            self.assertTrue(bool(item["family"]))
            self.assertTrue(bool(item["variant"]))

    def test_compare_same_pytorch_framework_success(self) -> None:
        r1 = self._create_run(engine="ultralytics-yolo", family="YOLOv8", variant="yolov8n", inference_time_ms=11.5)
        r2 = self._create_run(engine="ultralytics-yolo", family="YOLOv8", variant="yolov8s", inference_time_ms=14.2)

        out = self.service.compare_runs(self.db, [r1.run_id, r2.run_id])
        self.assertEqual(len(out["runs"]), 2)
        for item in out["runs"]:
            self.assertEqual(item["framework_key"], "pytorch")
            self.assertEqual(item["framework_label"], "PyTorch")
            self.assertEqual(item["engine"], "ultralytics-yolo")
        by_id = {it["run_id"]: it for it in out["runs"]}
        self.assertAlmostEqual(float(by_id[r1.run_id]["inference_time_ms"]), 11.5, places=6)
        self.assertAlmostEqual(float(by_id[r2.run_id]["inference_time_ms"]), 14.2, places=6)

    def test_compare_mixed_pytorch_and_paddle_conflict(self) -> None:
        r1 = self._create_run(engine="ultralytics-yolo", family="YOLOv8", variant="yolov8n")
        r2 = self._create_run(engine="paddle-det", family="PP-YOLOE", variant="ppyoloe_s")

        with self.assertRaises(FrameworkCompareConflict) as ctx:
            self.service.compare_runs(self.db, [r1.run_id, r2.run_id])

        err = ctx.exception
        self.assertIn("pytorch", err.framework_groups)
        self.assertIn("paddle", err.framework_groups)
        self.assertIn(r1.run_id, err.framework_groups["pytorch"])
        self.assertIn(r2.run_id, err.framework_groups["paddle"])

    def test_compare_unknown_same_engine_success(self) -> None:
        r1 = self._create_run(engine="my-engine", family="Custom", variant="c1")
        r2 = self._create_run(engine="my-engine", family="Custom", variant="c2")

        out = self.service.compare_runs(self.db, [r1.run_id, r2.run_id])
        self.assertEqual(len(out["runs"]), 2)
        for item in out["runs"]:
            self.assertEqual(item["framework_key"], "engine:my-engine")
            self.assertEqual(item["framework_label"], "Engine: my-engine")
            self.assertEqual(item["engine"], "my-engine")

    def test_compare_unknown_different_engine_conflict(self) -> None:
        r1 = self._create_run(engine="my-engine-a", family="Custom", variant="a")
        r2 = self._create_run(engine="my-engine-b", family="Custom", variant="b")

        with self.assertRaises(FrameworkCompareConflict) as ctx:
            self.service.compare_runs(self.db, [r1.run_id, r2.run_id])

        err = ctx.exception
        self.assertIn("engine:my-engine-a", err.framework_groups)
        self.assertIn("engine:my-engine-b", err.framework_groups)


if __name__ == "__main__":
    unittest.main()
