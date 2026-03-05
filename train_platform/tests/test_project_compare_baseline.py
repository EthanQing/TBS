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
from train_platform.models.training_run import TrainingRun
from train_platform.services.project_service import ProjectService
from train_platform.utils.exceptions import ConflictError


class ProjectCompareBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.service = ProjectService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _seed_project(self, suffix: str) -> tuple[Project, DatasetVersion]:
        ds = Dataset(
            name=f"dataset-{suffix}",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path=f"datasets/{suffix}",
        )
        self.db.add(ds)
        self.db.flush()

        dsv = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path=f"datasets/{suffix}/manifest.json",
        )
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = int(dsv.version_id)

        project = Project(
            name=f"project-{suffix}",
            dataset_id=int(ds.dataset_id),
            task_type=TaskType.DETECTION,
            is_active=True,
        )
        self.db.add(project)
        self.db.commit()
        return project, dsv

    def _create_run(self, project: Project, dsv: DatasetVersion, *, engine: str, variant: str) -> TrainingRun:
        arch = ModelArchitecture(
            family="TestFamily",
            variant=str(variant),
            task_type=TaskType.DETECTION,
            engine=str(engine),
        )
        self.db.add(arch)
        self.db.flush()

        run = TrainingRun(
            run_id=str(uuid4()),
            project_id=int(project.project_id),
            dataset_version_id=int(dsv.version_id),
            architecture_id=int(arch.architecture_id),
            name=f"run-{variant}",
            status=TrainingRunStatus.COMPLETED,
            progress=100,
            current_epoch=10,
            total_epochs=10,
        )
        self.db.add(run)
        self.db.commit()
        return run

    def test_set_get_clear_baseline_success(self) -> None:
        project, dsv = self._seed_project("a")
        run = self._create_run(project, dsv, engine="ultralytics-yolo", variant="v1")

        out = self.service.set_compare_baseline(
            self.db,
            int(project.project_id),
            framework_key="pytorch",
            baseline_run_id=str(run.run_id),
        )
        self.assertEqual(out["project_id"], int(project.project_id))
        self.assertEqual(out["framework_key"], "pytorch")
        self.assertEqual(out["baseline_run_id"], str(run.run_id))
        self.assertIsNotNone(out["baseline_run"])
        self.assertEqual(out["baseline_run"]["run_id"], str(run.run_id))
        self.assertEqual(out["baseline_run"]["engine"], "ultralytics-yolo")

        out2 = self.service.get_compare_baseline(self.db, int(project.project_id), "pytorch")
        self.assertEqual(out2["baseline_run_id"], str(run.run_id))

        cleared = self.service.clear_compare_baseline(self.db, int(project.project_id), "pytorch")
        self.assertEqual(cleared["baseline_run_id"], None)

        out3 = self.service.get_compare_baseline(self.db, int(project.project_id), "pytorch")
        self.assertEqual(out3["baseline_run_id"], None)

    def test_set_baseline_rejects_cross_project_run(self) -> None:
        p1, dsv1 = self._seed_project("b1")
        p2, dsv2 = self._seed_project("b2")
        run = self._create_run(p2, dsv2, engine="paddle-det", variant="v2")

        with self.assertRaises(ConflictError):
            self.service.set_compare_baseline(
                self.db,
                int(p1.project_id),
                framework_key="paddle",
                baseline_run_id=str(run.run_id),
            )

    def test_set_baseline_rejects_framework_mismatch(self) -> None:
        project, dsv = self._seed_project("c")
        run = self._create_run(project, dsv, engine="paddle-det", variant="v3")

        with self.assertRaises(ConflictError):
            self.service.set_compare_baseline(
                self.db,
                int(project.project_id),
                framework_key="pytorch",
                baseline_run_id=str(run.run_id),
            )

    def test_unknown_engine_can_use_engine_key(self) -> None:
        project, dsv = self._seed_project("d")
        run = self._create_run(project, dsv, engine="my-engine", variant="v4")

        out = self.service.set_compare_baseline(
            self.db,
            int(project.project_id),
            framework_key="engine:my-engine",
            baseline_run_id=str(run.run_id),
        )
        self.assertEqual(out["framework_key"], "engine:my-engine")
        self.assertEqual(out["baseline_run_id"], str(run.run_id))


if __name__ == "__main__":
    unittest.main()
