from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.deployment import Deployment, DeploymentLog
from train_platform.models.deployment_run import DeploymentRun
from train_platform.models.enums import (
    DatasetType,
    DatasetVersionStatus,
    DeploymentPlatform,
    DeploymentRunPhase,
    DeploymentRunStatus,
    DeploymentStatus,
    DeploymentTriggerType,
    ModelStage,
    TaskType,
    TrainingRunStatus,
)
from train_platform.models.model_registry import ModelVersion
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.services.deployment_runtime_service import DeploymentRuntimeService, generate_api_key, verify_api_key
from train_platform.utils.exceptions import ConflictError


class DeploymentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.svc = DeploymentRuntimeService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _seed(self):
        ds = Dataset(name="ds-runtime", dataset_type=DatasetType.DETECTION, format="yolo", storage_path="datasets/runtime")
        self.db.add(ds)
        self.db.flush()

        dsv = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="datasets/runtime/manifest.json",
        )
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = int(dsv.version_id)

        project = Project(name="project-runtime", dataset_id=int(ds.dataset_id), task_type=TaskType.DETECTION, is_active=True)
        self.db.add(project)
        self.db.flush()

        arch = ModelArchitecture(
            family="YOLO",
            variant="yolov8n-runtime",
            task_type=TaskType.DETECTION,
            engine="ultralytics-yolo",
        )
        self.db.add(arch)
        self.db.flush()

        tr = TrainingRun(
            run_id=str(uuid4()),
            project_id=int(project.project_id),
            dataset_version_id=int(dsv.version_id),
            architecture_id=int(arch.architecture_id),
            name="runtime-run",
            status=TrainingRunStatus.COMPLETED,
            progress=100,
            current_epoch=1,
            total_epochs=1,
        )
        self.db.add(tr)
        self.db.flush()

        mv = ModelVersion(
            project_id=int(project.project_id),
            run_id=str(tr.run_id),
            version="v1",
            stage=ModelStage.TESTING,
            # Path does not need to exist for queue/cancel tests.
            weights_path="runtime/missing.pt",
        )
        self.db.add(mv)
        self.db.flush()

        dep = Deployment(
            model_version_id=int(mv.model_version_id),
            name="runtime-deployment",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.PENDING,
            is_active=False,
        )
        self.db.add(dep)
        self.db.commit()
        return project, mv, dep

    def test_generate_and_verify_api_key(self) -> None:
        key, key_hash, _hint = generate_api_key()
        self.assertTrue(verify_api_key(key, key_hash))
        self.assertFalse(verify_api_key(f"{key}-bad", key_hash))

    def test_execute_creates_queued_run_and_pending_key_hash(self) -> None:
        _project, _mv, dep = self._seed()
        with patch.object(self.svc, "_start_pipeline_thread", lambda _rid: None):
            out = self.svc.execute_deployment(
                self.db,
                int(dep.deployment_id),
                payload={"operator": "tester", "reason": "deploy", "rotate_api_key": True, "conf": 0.3, "iou": 0.5},
            )

        run = out["run"]
        self.assertEqual(run.status, DeploymentRunStatus.QUEUED)
        self.assertIsNotNone(out["issued_api_key"])
        self.assertTrue(str(out["api_key_hint"] or "").strip())

        dep_db = self.db.query(Deployment).filter(Deployment.deployment_id == int(dep.deployment_id)).first()
        self.assertEqual(dep_db.status, DeploymentStatus.DEPLOYING)
        # Key hash is only applied in materialize step.
        self.assertIsNone(dep_db.api_key_hash)

        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        self.assertTrue(str(snapshot.get("pending_api_key_hash") or "").strip())
        self.assertEqual(len(snapshot.get("steps") or []), 4)

    def test_execute_rejects_when_project_has_active_run(self) -> None:
        project, mv, dep = self._seed()
        active = DeploymentRun(
            run_id=str(uuid4()),
            deployment_id=int(dep.deployment_id),
            project_id=int(project.project_id),
            model_version_id=int(mv.model_version_id),
            trigger_type=DeploymentTriggerType.MANUAL,
            status=DeploymentRunStatus.RUNNING,
            phase=DeploymentRunPhase.PREPARING,
            progress=10,
            cancel_requested=False,
            snapshot={"steps": []},
        )
        self.db.add(active)
        self.db.commit()

        with self.assertRaises(ConflictError):
            self.svc.execute_deployment(self.db, int(dep.deployment_id), payload={"operator": "tester"})

    def test_cancel_queued_run_marks_cancelled(self) -> None:
        _project, _mv, dep = self._seed()
        with patch.object(self.svc, "_start_pipeline_thread", lambda _rid: None):
            out = self.svc.execute_deployment(self.db, int(dep.deployment_id), payload={"operator": "tester"})
        run_id = out["run"].run_id

        cancelled = self.svc.cancel_run(self.db, run_id)
        self.assertEqual(cancelled.status, DeploymentRunStatus.CANCELLED)
        self.assertTrue(cancelled.cancel_requested)

        logs = (
            self.db.query(DeploymentLog)
            .filter(DeploymentLog.deployment_id == int(dep.deployment_id))
            .order_by(DeploymentLog.log_id.asc())
            .all()
        )
        hit = False
        for row in logs:
            data = row.data if isinstance(row.data, dict) else {}
            if str(data.get("run_id") or "") == str(run_id) and str(data.get("action") or "") == "cancel_requested":
                hit = True
                break
        self.assertTrue(hit)


if __name__ == "__main__":
    unittest.main()
