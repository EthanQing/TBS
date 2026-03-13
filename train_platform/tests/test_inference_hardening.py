from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.core.config import settings
from train_platform.db.base import Base
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus, TaskType, TrainingRunStatus
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun, TrainingRunParameters
from train_platform.services.dataset_service import DatasetService
from train_platform.services.inference_job_service import InferenceJobService
from train_platform.services.inference_service import InferenceService
from train_platform.services.training_run_service import TrainingRunService
from train_platform.utils.exceptions import ConflictError, ValidationError
from train_platform.utils.path_utils import resolve_temp_path, resolve_training_path
from train_platform.workers.inference_worker import app as inference_worker_app


class SettingsOverrideMixin:
    def setUp(self) -> None:
        super().setUp()  # type: ignore[misc]
        self._tmp_root = (Path.cwd() / ".tmp-tests" / uuid4().hex).resolve()
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self._orig = {
            "datasets_dir": settings.datasets_dir,
            "training_dir": settings.training_dir,
            "temp_dir": settings.temp_dir,
            "internal_api_token": settings.internal_api_token,
            "inference_max_download_bytes": settings.inference_max_download_bytes,
            "inference_download_timeout_sec": settings.inference_download_timeout_sec,
            "inference_allowed_schemes": settings.inference_allowed_schemes,
            "inference_allowed_hosts": settings.inference_allowed_hosts,
        }
        object.__setattr__(settings, "datasets_dir", (self._tmp_root / "datasets").resolve())
        object.__setattr__(settings, "training_dir", (self._tmp_root / "training").resolve())
        object.__setattr__(settings, "temp_dir", (self._tmp_root / "temp").resolve())
        object.__setattr__(settings, "internal_api_token", "")
        object.__setattr__(settings, "inference_max_download_bytes", 1024 * 1024)
        object.__setattr__(settings, "inference_download_timeout_sec", 5.0)
        object.__setattr__(settings, "inference_allowed_schemes", ("http", "https"))
        object.__setattr__(settings, "inference_allowed_hosts", tuple())
        settings.ensure_dirs()

    def tearDown(self) -> None:
        for k, v in self._orig.items():
            object.__setattr__(settings, k, v)
        shutil.rmtree(self._tmp_root, ignore_errors=True)
        super().tearDown()  # type: ignore[misc]


class PathSafetyTests(SettingsOverrideMixin, unittest.TestCase):
    def test_resolve_temp_path_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValidationError):
            resolve_temp_path("../etc/passwd")

    def test_resolve_training_path_rejects_absolute(self) -> None:
        with self.assertRaises(ValidationError):
            resolve_training_path(str((self._tmp_root / "outside.pt").resolve()))

    def test_resolve_temp_path_valid_relative(self) -> None:
        p = resolve_temp_path("inputs/a.jpg")
        self.assertTrue(str(p).startswith(str(settings.temp_dir.resolve())))


class InferenceInputHardeningTests(SettingsOverrideMixin, unittest.TestCase):
    def test_materialize_input_rejects_absolute_local_path(self) -> None:
        svc = InferenceService()
        local = (settings.temp_dir / "inference" / "a.jpg").resolve()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"abc")

        with self.assertRaises(ValidationError):
            svc._materialize_input(input_path=str(local), image_url=None)

    def test_download_to_temp_respects_size_limit(self) -> None:
        svc = InferenceService()
        object.__setattr__(settings, "inference_max_download_bytes", 10)
        object.__setattr__(settings, "inference_allowed_hosts", ("example.com",))

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 65536):
                yield b"1234567890"
                yield b"11"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("train_platform.services.inference_service.requests.get", return_value=_Resp()):
            with self.assertRaises(ValidationError):
                svc._download_to_temp("https://example.com/sample.jpg")


class WorkerHardeningTests(SettingsOverrideMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        object.__setattr__(settings, "internal_api_token", "secret-token")
        self.client = TestClient(inference_worker_app)

    def test_internal_endpoint_requires_token(self) -> None:
        resp = self.client.post(
            "/internal/inference/yolo",
            json={"weights_path": "x.pt", "image_path": "x.jpg", "conf": 0.5, "iou": 0.45},
        )
        self.assertEqual(resp.status_code, 401)

    def test_export_onnx_rejects_output_outside_training_dir(self) -> None:
        src_pt = (settings.training_dir / "run-1" / "weights" / "best.pt").resolve()
        src_pt.parent.mkdir(parents=True, exist_ok=True)
        src_pt.write_bytes(b"pt")
        out_onnx = (self._tmp_root / "outside" / "x.onnx").resolve()
        resp = self.client.post(
            "/internal/training-runs/export-onnx",
            json={
                "src_pt": str(src_pt),
                "out_onnx": str(out_onnx),
                "dynamic": True,
                "opset": 12,
                "imgsz": 640,
            },
            headers={"X-Internal-Token": "secret-token"},
        )
        self.assertEqual(resp.status_code, 400)


class ResumeStateMachineTests(SettingsOverrideMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.svc = TrainingRunService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        super().tearDown()

    def test_resume_completed_run_is_rejected(self) -> None:
        ds = Dataset(name="d1", dataset_type=DatasetType.DETECTION, format="yolo", storage_path="datasets/d1")
        self.db.add(ds)
        self.db.flush()
        dsv = DatasetVersion(dataset_id=ds.dataset_id, version=1, status=DatasetVersionStatus.FINALIZED)
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = dsv.version_id
        project = Project(name="p1", dataset_id=ds.dataset_id, task_type=TaskType.DETECTION, is_active=True)
        self.db.add(project)
        self.db.flush()
        from train_platform.models.architecture import ModelArchitecture

        arch = ModelArchitecture(family="YOLOv8", variant="n", task_type=TaskType.DETECTION, engine="ultralytics-yolo")
        self.db.add(arch)
        self.db.flush()
        run = TrainingRun(
            run_id="run-completed",
            project_id=project.project_id,
            dataset_version_id=dsv.version_id,
            architecture_id=arch.architecture_id,
            name="r1",
            status=TrainingRunStatus.COMPLETED,
            progress=100,
            current_epoch=10,
            total_epochs=10,
        )
        self.db.add(run)
        self.db.add(TrainingRunParameters(run_id=run.run_id))
        self.db.commit()

        with self.assertRaises(ConflictError):
            self.svc.resume_run(self.db, run.run_id)


class DatasetViewCollisionTests(SettingsOverrideMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.svc = DatasetService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        super().tearDown()

    def test_get_view_uses_relative_key_without_stem_collision(self) -> None:
        ds_root = settings.datasets_dir / "datasets" / "collision"
        (ds_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (ds_root / "labels" / "val" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        manifest = settings.datasets_dir / "datasets" / "collision" / "manifest.ndjson"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"path": "images/train/a.jpg", "size_bytes": 1, "mtime": 1}),
                    json.dumps({"path": "images/val/a.jpg", "size_bytes": 1, "mtime": 1}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        ds = Dataset(name="collision-ds", dataset_type=DatasetType.DETECTION, format="yolo", storage_path="datasets/collision")
        self.db.add(ds)
        self.db.flush()
        ver = DatasetVersion(
            dataset_id=ds.dataset_id,
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="datasets/collision/manifest.ndjson",
            meta={"yolo": {"names": ["c0", "c1"]}},
        )
        self.db.add(ver)
        self.db.flush()
        ds.active_version_id = ver.version_id
        self.db.commit()

        out_all = self.svc.get_view(self.db, ds.dataset_id, version_id=ver.version_id, page=1, page_size=10)
        by_name = {x["name"]: x["classes"] for x in out_all["items"]}
        self.assertEqual(by_name["images/train/a.jpg"], [0])
        self.assertEqual(by_name["images/val/a.jpg"], [1])

        out_filtered = self.svc.get_view(
            self.db, ds.dataset_id, version_id=ver.version_id, class_id=1, page=1, page_size=10
        )
        self.assertEqual(len(out_filtered["items"]), 1)
        self.assertEqual(out_filtered["items"][0]["name"], "images/val/a.jpg")


class InferenceJobLockTests(SettingsOverrideMixin, unittest.TestCase):
    def test_create_job_lock_rejects_parallel_holder(self) -> None:
        svc = InferenceJobService()
        svc._acquire_create_job_lock(timeout_sec=0.2)
        try:
            with self.assertRaises(ConflictError):
                svc._acquire_create_job_lock(timeout_sec=0.1)
        finally:
            svc._release_create_job_lock()


if __name__ == "__main__":
    unittest.main()
