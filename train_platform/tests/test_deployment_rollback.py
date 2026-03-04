from __future__ import annotations

import unittest
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.deployment import Deployment, DeploymentLog
from train_platform.models.enums import (
    DatasetType,
    DatasetVersionStatus,
    DeploymentPlatform,
    DeploymentStatus,
    LogLevel,
    ModelStage,
    TaskType,
    TrainingRunStatus,
)
from train_platform.models.model_registry import ModelVersion
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.services.deployment_service import DeploymentService
from train_platform.utils.exceptions import ConflictError


class DeploymentRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.service = DeploymentService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _seed_project(self, name_prefix: str = "p"):
        ds = Dataset(
            name=f"dataset-{name_prefix}",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path=f"datasets/{name_prefix}",
        )
        self.db.add(ds)
        self.db.flush()

        dsv = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path=f"datasets/{name_prefix}/manifest.json",
        )
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = int(dsv.version_id)

        project = Project(
            name=f"project-{name_prefix}",
            dataset_id=int(ds.dataset_id),
            task_type=TaskType.DETECTION,
            is_active=True,
        )
        self.db.add(project)
        self.db.flush()

        arch = ModelArchitecture(
            family="YOLO",
            variant=f"yolov8n-{name_prefix}",
            task_type=TaskType.DETECTION,
            engine="ultralytics-yolo",
        )
        self.db.add(arch)
        self.db.flush()

        self.db.commit()
        return project, dsv, arch

    def _create_model_version(self, project_id: int, dataset_version_id: int, architecture_id: int, *, version: str, stage: ModelStage):
        run = TrainingRun(
            run_id=str(uuid4()),
            project_id=int(project_id),
            dataset_version_id=int(dataset_version_id),
            architecture_id=int(architecture_id),
            name=f"run-{version}",
            status=TrainingRunStatus.COMPLETED,
            progress=100,
            current_epoch=10,
            total_epochs=10,
        )
        self.db.add(run)
        self.db.flush()

        mv = ModelVersion(
            project_id=int(project_id),
            run_id=str(run.run_id),
            version=version,
            stage=stage,
            weights_path=f"weights/{version}.pt",
        )
        self.db.add(mv)
        self.db.flush()
        return mv

    def test_rollback_success_updates_deployment_and_stages_and_logs(self) -> None:
        project, dsv, arch = self._seed_project("a")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.TESTING)
        mv2 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v2", stage=ModelStage.TESTING)
        mv3 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v3", stage=ModelStage.PRODUCTION)

        dep_keep = Deployment(
            model_version_id=int(mv3.model_version_id),
            name="active-deployment",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        dep_other = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="other-active",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        dep_success = Deployment(
            model_version_id=int(mv2.model_version_id),
            name="history-success",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.INACTIVE,
            is_active=False,
        )
        self.db.add_all([dep_keep, dep_other, dep_success])
        self.db.commit()

        out = self.service.rollback_deployment(
            self.db,
            int(dep_keep.deployment_id),
            target_model_version_id=int(mv2.model_version_id),
            reason="hotfix regression",
            operator="tester",
        )
        self.assertEqual(int(out["deployment"].model_version_id), int(mv2.model_version_id))
        self.assertEqual(out["deployment"].status, DeploymentStatus.ACTIVE)
        self.assertTrue(bool(out["deployment"].is_active))

        dep_other_db = self.service.get_deployment(self.db, int(dep_other.deployment_id))
        self.assertFalse(bool(dep_other_db.is_active))
        self.assertEqual(dep_other_db.status, DeploymentStatus.INACTIVE)

        mv2_db = self.db.query(ModelVersion).filter(ModelVersion.model_version_id == int(mv2.model_version_id)).first()
        mv3_db = self.db.query(ModelVersion).filter(ModelVersion.model_version_id == int(mv3.model_version_id)).first()
        self.assertEqual(mv2_db.stage, ModelStage.PRODUCTION)
        self.assertEqual(mv3_db.stage, ModelStage.TESTING)

        logs = (
            self.db.query(DeploymentLog)
            .filter(DeploymentLog.deployment_id == int(dep_keep.deployment_id))
            .order_by(DeploymentLog.log_id.desc())
            .all()
        )
        rollback_log = next((x for x in logs if isinstance(x.data, dict) and x.data.get("action") == "rollback"), None)
        self.assertIsNotNone(rollback_log)
        self.assertEqual(rollback_log.data.get("from_model_version_id"), int(mv3.model_version_id))
        self.assertEqual(rollback_log.data.get("to_model_version_id"), int(mv2.model_version_id))
        self.assertEqual(rollback_log.data.get("reason"), "hotfix regression")
        self.assertEqual(rollback_log.data.get("operator"), "tester")

    def test_rollback_rejects_same_target(self) -> None:
        project, dsv, arch = self._seed_project("b")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.PRODUCTION)
        dep = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="dep",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        self.db.add(dep)
        self.db.commit()

        with self.assertRaises(ConflictError):
            self.service.rollback_deployment(
                self.db,
                int(dep.deployment_id),
                target_model_version_id=int(mv1.model_version_id),
                reason="same target",
                operator="tester",
            )

    def test_rollback_rejects_cross_project_target(self) -> None:
        p1, dsv1, arch1 = self._seed_project("c1")
        p2, dsv2, arch2 = self._seed_project("c2")
        mv1 = self._create_model_version(p1.project_id, dsv1.version_id, arch1.architecture_id, version="v1", stage=ModelStage.PRODUCTION)
        mv2 = self._create_model_version(p2.project_id, dsv2.version_id, arch2.architecture_id, version="v2", stage=ModelStage.TESTING)

        dep = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="dep",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        self.db.add(dep)
        self.db.commit()

        with self.assertRaises(ConflictError):
            self.service.rollback_deployment(
                self.db,
                int(dep.deployment_id),
                target_model_version_id=int(mv2.model_version_id),
                reason="cross project",
                operator="tester",
            )

    def test_candidates_merge_success_and_rollback_logs(self) -> None:
        project, dsv, arch = self._seed_project("d")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.TESTING)
        mv2 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v2", stage=ModelStage.TESTING)
        mv3 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v3", stage=ModelStage.PRODUCTION)
        mv4 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v4", stage=ModelStage.TESTING)

        dep = Deployment(
            model_version_id=int(mv3.model_version_id),
            name="dep",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        dep_hist1 = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="hist1",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.INACTIVE,
            is_active=False,
        )
        dep_hist2 = Deployment(
            model_version_id=int(mv2.model_version_id),
            name="hist2",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.INACTIVE,
            is_active=False,
        )
        self.db.add_all([dep, dep_hist1, dep_hist2])
        self.db.flush()

        self.db.add(
            DeploymentLog(
                deployment_id=int(dep.deployment_id),
                level=LogLevel.INFO,
                message="manual rollback",
                data={
                    "action": "rollback",
                    "from_model_version_id": int(mv4.model_version_id),
                    "to_model_version_id": int(mv3.model_version_id),
                    "reason": "previous rollback",
                    "operator": "tester",
                },
            )
        )
        self.db.commit()

        payload = self.service.get_rollback_candidates(self.db, int(dep.deployment_id))
        got = {int(x.model_version_id) for x in payload["candidates"]}
        self.assertSetEqual(got, {int(mv1.model_version_id), int(mv2.model_version_id), int(mv4.model_version_id)})

    def test_list_deployments_supports_project_filter(self) -> None:
        p1, dsv1, arch1 = self._seed_project("e1")
        p2, dsv2, arch2 = self._seed_project("e2")

        mv1 = self._create_model_version(p1.project_id, dsv1.version_id, arch1.architecture_id, version="v1", stage=ModelStage.TESTING)
        mv2 = self._create_model_version(p2.project_id, dsv2.version_id, arch2.architecture_id, version="v2", stage=ModelStage.TESTING)

        self.db.add_all(
            [
                Deployment(
                    model_version_id=int(mv1.model_version_id),
                    name="dep-p1",
                    platform=DeploymentPlatform.LOCAL,
                    status=DeploymentStatus.ACTIVE,
                    is_active=True,
                ),
                Deployment(
                    model_version_id=int(mv2.model_version_id),
                    name="dep-p2",
                    platform=DeploymentPlatform.LOCAL,
                    status=DeploymentStatus.ACTIVE,
                    is_active=True,
                ),
            ]
        )
        self.db.commit()

        p1_items = self.service.list_deployments(self.db, project_id=int(p1.project_id), skip=0, limit=50)
        p2_items = self.service.list_deployments(self.db, project_id=int(p2.project_id), skip=0, limit=50)
        self.assertEqual(len(p1_items), 1)
        self.assertEqual(len(p2_items), 1)

    def test_create_deployment_keeps_existing_active_and_new_pending(self) -> None:
        project, dsv, arch = self._seed_project("f")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.TESTING)
        mv2 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v2", stage=ModelStage.TESTING)

        old_dep = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="old",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        self.db.add(old_dep)
        self.db.commit()

        created = self.service.create_deployment(
            self.db,
            obj={
                "model_version_id": int(mv2.model_version_id),
                "name": "new",
                "platform": DeploymentPlatform.LOCAL,
            },
        )
        old_dep_db = self.service.get_deployment(self.db, int(old_dep.deployment_id))
        self.assertTrue(bool(old_dep_db.is_active))
        self.assertEqual(old_dep_db.status, DeploymentStatus.ACTIVE)
        self.assertFalse(bool(created.is_active))
        self.assertEqual(created.status, DeploymentStatus.PENDING)

    def test_update_deployment_activate_normalizes_active_flag(self) -> None:
        project, dsv, arch = self._seed_project("g")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.TESTING)
        mv2 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v2", stage=ModelStage.TESTING)

        dep1 = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="dep1",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        dep2 = Deployment(
            model_version_id=int(mv2.model_version_id),
            name="dep2",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.INACTIVE,
            is_active=False,
        )
        self.db.add_all([dep1, dep2])
        self.db.commit()

        self.service.update_deployment(
            self.db,
            int(dep2.deployment_id),
            patch={"is_active": True, "status": DeploymentStatus.ACTIVE},
        )
        dep1_db = self.service.get_deployment(self.db, int(dep1.deployment_id))
        dep2_db = self.service.get_deployment(self.db, int(dep2.deployment_id))
        self.assertFalse(bool(dep1_db.is_active))
        self.assertEqual(dep1_db.status, DeploymentStatus.INACTIVE)
        self.assertTrue(bool(dep2_db.is_active))
        self.assertEqual(dep2_db.status, DeploymentStatus.ACTIVE)

    def test_list_rollback_history_returns_only_rollback_entries(self) -> None:
        project, dsv, arch = self._seed_project("h")
        mv1 = self._create_model_version(project.project_id, dsv.version_id, arch.architecture_id, version="v1", stage=ModelStage.PRODUCTION)
        dep = Deployment(
            model_version_id=int(mv1.model_version_id),
            name="dep",
            platform=DeploymentPlatform.LOCAL,
            status=DeploymentStatus.ACTIVE,
            is_active=True,
        )
        self.db.add(dep)
        self.db.flush()
        self.db.add_all(
            [
                DeploymentLog(
                    deployment_id=int(dep.deployment_id),
                    level=LogLevel.INFO,
                    message="normal log",
                    data={"foo": "bar"},
                ),
                DeploymentLog(
                    deployment_id=int(dep.deployment_id),
                    level=LogLevel.INFO,
                    message="rollback",
                    data={
                        "action": "rollback",
                        "from_model_version_id": int(mv1.model_version_id),
                        "to_model_version_id": int(mv1.model_version_id),
                        "from_version": "v1",
                        "to_version": "v1",
                        "reason": "test",
                        "operator": "tester",
                    },
                ),
            ]
        )
        self.db.commit()

        rows = self.service.list_rollback_history(self.db, int(dep.deployment_id), limit=50)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operator"], "tester")


if __name__ == "__main__":
    unittest.main()
