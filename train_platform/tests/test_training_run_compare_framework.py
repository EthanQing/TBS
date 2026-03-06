from __future__ import annotations

import unittest
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus, TaskType, TrainingRunStatus
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun, TrainingRunEpochMetric, TrainingRunResult
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

    def _create_run(
        self,
        *,
        engine: str,
        family: str,
        variant: str,
        inference_time_ms: float | None = None,
        status: TrainingRunStatus = TrainingRunStatus.COMPLETED,
        best_metrics: dict | None = None,
        final_metrics: dict | None = None,
        epoch_metrics: list[dict] | None = None,
    ) -> TrainingRun:
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
            status=status,
            progress=100,
            current_epoch=10,
            total_epochs=10,
        )
        self.db.add(run)
        self.db.flush()

        if inference_time_ms is not None or best_metrics is not None or final_metrics is not None:
            result = TrainingRunResult(
                run_id=str(run.run_id),
                inference_time_ms=float(inference_time_ms) if inference_time_ms is not None else None,
                best_metrics=best_metrics,
                final_metrics=final_metrics,
            )
            self.db.add(result)

        for item in epoch_metrics or []:
            self.db.add(
                TrainingRunEpochMetric(
                    run_id=str(run.run_id),
                    epoch=int(item["epoch"]),
                    metrics=item.get("metrics") or {},
                )
            )

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

    def test_compare_metric_summary_fallback_from_epoch(self) -> None:
        r1 = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="fallback",
            epoch_metrics=[
                {"epoch": 0, "metrics": {"metrics/mAP50(B)": 0.41, "metrics/precision(B)": 0.50}},
                {"epoch": 1, "metrics": {"metrics/mAP50(B)": 0.56, "metrics/precision(B)": 0.61}},
                {"epoch": 2, "metrics": {"metrics/mAP50(B)": 0.52, "metrics/precision(B)": 0.59}},
            ],
        )
        r2 = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="result",
            best_metrics={"metrics/mAP50(B)": 0.77, "metrics/precision(B)": 0.71},
            final_metrics={"metrics/mAP50(B)": 0.74, "metrics/precision(B)": 0.69},
        )

        out = self.service.compare_runs(self.db, [r1.run_id, r2.run_id])
        by_id = {it["run_id"]: it for it in out["runs"]}
        s1 = by_id[r1.run_id]["metric_summary"]
        s2 = by_id[r2.run_id]["metric_summary"]

        self.assertEqual(s1["source"], "epoch_fallback")
        self.assertAlmostEqual(float(s1["best"]["metrics/mAP50(B)"]), 0.56, places=6)
        self.assertAlmostEqual(float(s1["final"]["metrics/mAP50(B)"]), 0.52, places=6)
        self.assertEqual(s2["source"], "result")
        self.assertAlmostEqual(float(s2["best"]["metrics/mAP50(B)"]), 0.77, places=6)

    def test_benchmark_inference_time_cached_and_measured(self) -> None:
        cached = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="cached",
            inference_time_ms=11.5,
        )
        measured = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="measured",
        )

        with patch.object(TrainingRunService, "_ensure_benchmark_image", return_value=Path("dummy.jpg")), patch.object(
            TrainingRunService, "_measure_run_inference_latency", return_value=9.87
        ):
            out = self.service.benchmark_inference_times(self.db, run_ids=[cached.run_id, measured.run_id], force=False)

        by_id = {it["run_id"]: it for it in out["items"]}
        self.assertEqual(by_id[cached.run_id]["status"], "cached")
        self.assertEqual(by_id[measured.run_id]["status"], "measured")
        self.assertAlmostEqual(float(by_id[measured.run_id]["inference_time_ms"]), 9.87, places=6)

        refreshed = self.service.get_run(self.db, measured.run_id)
        self.assertIsNotNone(refreshed.result)
        self.assertAlmostEqual(float(refreshed.result.inference_time_ms), 9.87, places=6)

    def test_benchmark_inference_time_force_and_skipped(self) -> None:
        cached = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="cached-force",
            inference_time_ms=8.2,
        )
        running = self._create_run(
            engine="ultralytics-yolo",
            family="YOLOv8",
            variant="running",
            status=TrainingRunStatus.RUNNING,
        )

        with patch.object(TrainingRunService, "_ensure_benchmark_image", return_value=Path("dummy.jpg")), patch.object(
            TrainingRunService, "_measure_run_inference_latency", return_value=12.34
        ):
            out = self.service.benchmark_inference_times(self.db, run_ids=[cached.run_id, running.run_id], force=True)

        by_id = {it["run_id"]: it for it in out["items"]}
        self.assertEqual(by_id[cached.run_id]["status"], "measured")
        self.assertEqual(by_id[running.run_id]["status"], "skipped")
        refreshed = self.service.get_run(self.db, cached.run_id)
        self.assertAlmostEqual(float(refreshed.result.inference_time_ms), 12.34, places=6)


if __name__ == "__main__":
    unittest.main()
