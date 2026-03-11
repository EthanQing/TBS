from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus, TaskType
from train_platform.models.project import Project
from train_platform.services.architecture_service import ArchitectureService
from train_platform.services.training_run_service import TrainingRunService
from train_platform.utils.exceptions import ValidationError


class ArchitectureShieldingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_list_architectures_hides_paddle_engine(self) -> None:
        self.db.add_all(
            [
                ModelArchitecture(
                    family="YOLOv8",
                    variant="yolov8n",
                    task_type=TaskType.DETECTION,
                    engine="ultralytics-yolo",
                ),
                ModelArchitecture(
                    family="PP-YOLOE",
                    variant="ppyoloe_s",
                    task_type=TaskType.DETECTION,
                    engine="paddle-det",
                ),
            ]
        )
        self.db.commit()

        items = ArchitectureService().list_architectures(self.db, task_type=TaskType.DETECTION)
        variants = {str(item.variant) for item in items}

        self.assertIn("yolov8n", variants)
        self.assertNotIn("ppyoloe_s", variants)

    def test_create_run_rejects_disabled_paddle_engine(self) -> None:
        dataset = Dataset(
            name="shielding-dataset",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="datasets/shielding",
        )
        self.db.add(dataset)
        self.db.flush()

        version = DatasetVersion(
            dataset_id=int(dataset.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="datasets/shielding/manifest.json",
        )
        self.db.add(version)
        self.db.flush()
        dataset.active_version_id = int(version.version_id)

        project = Project(
            name="shielding-project",
            dataset_id=int(dataset.dataset_id),
            task_type=TaskType.DETECTION,
            is_active=True,
        )
        self.db.add(project)
        self.db.flush()

        paddle_arch = ModelArchitecture(
            family="PP-YOLOE",
            variant="ppyoloe_plus_crn_s_80e_coco",
            task_type=TaskType.DETECTION,
            engine="paddle-det",
        )
        self.db.add(paddle_arch)
        self.db.commit()

        with self.assertRaises(ValidationError) as ctx:
            TrainingRunService().create_run(
                self.db,
                obj={
                    "project_id": int(project.project_id),
                    "architecture_id": int(paddle_arch.architecture_id),
                    "dataset_version_id": int(version.version_id),
                    "parameters": {"epochs": 1},
                },
            )

        self.assertIn("disabled", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
