from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from train_platform.core.config import settings
from train_platform.db.base import Base
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus
from train_platform.schemas.v2.dataset_augmentations import DatasetAugmentationCreate, DatasetAugmentationPublishIn
from train_platform.services.dataset_augmentation_service import DatasetAugmentationService
from train_platform.utils.exceptions import ConflictError
from train_platform.utils.path_utils import resolve_dataset_path


class DatasetAugmentationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.svc = DatasetAugmentationService()
        self._tmp_root = Path(tempfile.mkdtemp(prefix="aug-tests-"))
        self._orig_datasets_dir = settings.datasets_dir
        self._orig_temp_dir = settings.temp_dir
        object.__setattr__(settings, "datasets_dir", (self._tmp_root / "datasets").resolve())
        object.__setattr__(settings, "temp_dir", (self._tmp_root / "temp").resolve())
        self._dataset_tokens: list[str] = []
        self._dataset_names: list[str] = []
        self._dataset_ids: list[int] = []
        settings.ensure_dirs()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        for token in self._dataset_tokens:
            shutil.rmtree(resolve_dataset_path(token), ignore_errors=True)
        for name in self._dataset_names:
            shutil.rmtree((settings.datasets_dir / ".versions" / name), ignore_errors=True)
        for ds_id in self._dataset_ids:
            shutil.rmtree((settings.temp_dir / "dataset_augmentations" / str(ds_id)), ignore_errors=True)
        object.__setattr__(settings, "datasets_dir", self._orig_datasets_dir)
        object.__setattr__(settings, "temp_dir", self._orig_temp_dir)
        shutil.rmtree(self._tmp_root, ignore_errors=True)

    def _seed_dataset(self, *, image_count: int = 2) -> Dataset:
        token = f"aug_test_{uuid4().hex[:10]}"
        name = f"dataset_{uuid4().hex[:8]}"
        root = resolve_dataset_path(token)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(image_count):
            img = Image.new("RGB", (128, 128), color=(255, 255, 255))
            img.save(root / "images" / f"img_{i}.jpg")
            (root / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (root / "data.yaml").write_text("train: images\nval: images\nnc: 1\nnames: ['obj']\n", encoding="utf-8")

        ds = Dataset(
            name=name,
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path=token,
        )
        self.db.add(ds)
        self.db.flush()
        ver = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            snapshot_path=token,
            manifest_path=f"{token}/manifest.ndjson",
        )
        self.db.add(ver)
        self.db.flush()
        ds.active_version_id = int(ver.version_id)
        self.db.commit()
        self._dataset_tokens.append(token)
        self._dataset_names.append(name)
        self._dataset_ids.append(int(ds.dataset_id))
        return ds

    def _wait_terminal(self, dataset_id: int, job_id: str, timeout_s: float = 20.0):
        t0 = time.time()
        while True:
            row = self.svc.get_job(dataset_id, job_id)
            if str(row.status) in {"completed", "failed", "cancelled"}:
                return row
            if (time.time() - t0) > timeout_s:
                self.fail(f"job timeout: {job_id}")
            time.sleep(0.1)

    def test_preview_reports_transform_counts(self) -> None:
        ds = self._seed_dataset(image_count=1)
        payload = DatasetAugmentationCreate(
            include_original=True,
            max_outputs_per_image=20,
            slice={"enabled": True, "scales": [64], "overlap": 0.2},
            rotate={"enabled": True, "angles": [90]},
            translate={"enabled": True, "offsets": [{"dx": 0.1, "dy": 0.0}]},
        )
        out = self.svc.preview(self.db, int(ds.dataset_id), payload)
        self.assertGreaterEqual(int(out.total_images), 1)
        self.assertGreaterEqual(int(out.estimated_generated_outputs), 1)
        self.assertIn("slice", out.per_transform)
        self.assertIn("rotate", out.per_transform)
        self.assertIn("translate", out.per_transform)

    def test_create_publish_default_not_activate(self) -> None:
        ds = self._seed_dataset(image_count=1)
        old_active = int(ds.active_version_id)
        payload = DatasetAugmentationCreate(
            include_original=True,
            max_outputs_per_image=5,
            slice={"enabled": False},
            rotate={"enabled": False},
            translate={"enabled": False},
        )
        with patch("train_platform.services.dataset_augmentation_service.SessionLocal", self.SessionLocal):
            job = self.svc.create_job(self.db, int(ds.dataset_id), payload)
            fin = self._wait_terminal(int(ds.dataset_id), str(job.job_id))
        self.assertEqual(str(fin.status), "completed")
        self.assertGreaterEqual(int(fin.result.generated_images), 1)

        pub = self.svc.publish_job(self.db, int(ds.dataset_id), str(job.job_id), DatasetAugmentationPublishIn())
        self.assertGreater(int(pub.version_id), 0)

        ds_db = self.db.query(Dataset).filter(Dataset.dataset_id == int(ds.dataset_id)).first()
        self.assertEqual(int(ds_db.active_version_id), old_active)

    def test_concurrency_guard_rejects_second_active_job(self) -> None:
        ds = self._seed_dataset(image_count=10)
        payload = DatasetAugmentationCreate(
            include_original=True,
            max_outputs_per_image=10,
            slice={"enabled": True, "scales": [64]},
            rotate={"enabled": False},
            translate={"enabled": False},
        )
        with patch("train_platform.services.dataset_augmentation_service.SessionLocal", self.SessionLocal):
            first = self.svc.create_job(self.db, int(ds.dataset_id), payload)
            with self.assertRaises(ConflictError):
                self.svc.create_job(self.db, int(ds.dataset_id), payload)
            self._wait_terminal(int(ds.dataset_id), str(first.job_id))

    def test_cancel_job_transitions_to_cancelled(self) -> None:
        ds = self._seed_dataset(image_count=12)
        payload = DatasetAugmentationCreate(
            include_original=True,
            max_outputs_per_image=30,
            slice={"enabled": True, "scales": [64, 80]},
            rotate={"enabled": True, "angles": [10, -10]},
            translate={"enabled": True, "offsets": [{"dx": 0.1, "dy": 0.0}, {"dx": -0.1, "dy": 0.05}]},
        )
        with patch("train_platform.services.dataset_augmentation_service.SessionLocal", self.SessionLocal):
            job = self.svc.create_job(self.db, int(ds.dataset_id), payload)
            time.sleep(0.2)
            self.svc.cancel_job(int(ds.dataset_id), str(job.job_id))
            fin = self._wait_terminal(int(ds.dataset_id), str(job.job_id))
        self.assertEqual(str(fin.status), "cancelled")


if __name__ == "__main__":
    unittest.main()
