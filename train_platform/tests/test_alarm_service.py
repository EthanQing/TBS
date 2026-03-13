from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from train_platform.db.base import Base
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.enums import DatasetType, DatasetVersionStatus, TaskType, TrainingRunStatus
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun
from train_platform.services.alarm_service import AlarmService
from train_platform.utils.exceptions import ConflictError, ValidationError


class AlarmServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.db = self.SessionLocal()
        self.svc = AlarmService()
        self.project, self.dataset_version, self.arch = self._seed_project_context()
        self.svc.ensure_default_rules(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _seed_project_context(self):
        ds = Dataset(
            name="alarm-dataset",
            dataset_type=DatasetType.DETECTION,
            format="yolo",
            storage_path="datasets/alarm",
        )
        self.db.add(ds)
        self.db.flush()

        dsv = DatasetVersion(
            dataset_id=int(ds.dataset_id),
            version=1,
            status=DatasetVersionStatus.FINALIZED,
            manifest_path="datasets/alarm/manifest.ndjson",
        )
        self.db.add(dsv)
        self.db.flush()
        ds.active_version_id = int(dsv.version_id)

        proj = Project(
            name="alarm-project",
            dataset_id=int(ds.dataset_id),
            task_type=TaskType.DETECTION,
            is_active=True,
        )
        self.db.add(proj)
        self.db.flush()

        arch = ModelArchitecture(
            family="YOLOv8",
            variant="yolov8n",
            task_type=TaskType.DETECTION,
            engine="ultralytics-yolo",
        )
        self.db.add(arch)
        self.db.commit()
        return proj, dsv, arch

    def _create_run(
        self,
        *,
        status: TrainingRunStatus,
        error_message: str | None = None,
        started_delta_sec: int | None = None,
        heartbeat_delta_sec: int | None = None,
    ) -> TrainingRun:
        now = datetime.now(timezone.utc)
        started_at = now - timedelta(seconds=int(started_delta_sec)) if started_delta_sec is not None else None
        heartbeat_at = now - timedelta(seconds=int(heartbeat_delta_sec)) if heartbeat_delta_sec is not None else None
        run = TrainingRun(
            run_id=str(uuid4()),
            project_id=int(self.project.project_id),
            dataset_version_id=int(self.dataset_version.version_id),
            architecture_id=int(self.arch.architecture_id),
            name="alarm-run",
            status=status,
            progress=0,
            current_epoch=0,
            total_epochs=100,
            started_at=started_at,
            heartbeat_at=heartbeat_at,
            error_message=error_message,
        )
        self.db.add(run)
        self.db.commit()
        return run

    def test_rule_crud_validation(self) -> None:
        rules, total = self.svc.list_rules(self.db)
        self.assertGreaterEqual(total, 2)

        with self.assertRaises(ValidationError):
            self.svc.create_rule(
                self.db,
                obj={
                    "rule_type": "unknown_rule",
                    "name": "x",
                    "severity": "high",
                    "enabled": True,
                    "cooldown_seconds": 0,
                    "config": {},
                },
            )

        failed_rule = next(r for r in rules if r.rule_type == AlarmService.RULE_TYPE_TRAINING_FAILED)
        updated = self.svc.update_rule(self.db, int(failed_rule.rule_id), patch={"enabled": False})
        self.assertFalse(bool(updated.enabled))

        with self.assertRaises(ValidationError):
            self.svc.update_rule(
                self.db,
                int(failed_rule.rule_id),
                patch={"severity": "invalid"},
            )

    def test_trigger_failed_and_auto_resolve(self) -> None:
        run = self._create_run(status=TrainingRunStatus.FAILED, error_message="boom")

        out1 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out1["triggered_new"]), 1)

        active, total_active = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_ACTIVE, source_id=run.run_id
        )
        self.assertEqual(total_active, 1)
        self.assertEqual(active[0].rule_type, AlarmService.RULE_TYPE_TRAINING_FAILED)

        run.status = TrainingRunStatus.COMPLETED
        run.error_message = None
        self.db.commit()

        out2 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out2["resolved"]), 1)

        _, total_active2 = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_ACTIVE, source_id=run.run_id
        )
        self.assertEqual(total_active2, 0)
        hist, total_hist = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_RESOLVED, source_id=run.run_id
        )
        self.assertEqual(total_hist, 1)
        self.assertIsNotNone(hist[0].resolved_at)

    def test_trigger_stale_and_resolve_after_heartbeat_recover(self) -> None:
        run = self._create_run(
            status=TrainingRunStatus.RUNNING,
            started_delta_sec=1000,
            heartbeat_delta_sec=1000,
        )

        out1 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out1["triggered_new"]), 1)

        active, total_active = self.svc.list_alerts(
            self.db,
            status=AlarmService.STATUS_ACTIVE,
            rule_type=AlarmService.RULE_TYPE_TRAINING_STALE,
            source_id=run.run_id,
        )
        self.assertEqual(total_active, 1)
        alert = active[0]
        self.assertEqual(alert.rule_type, AlarmService.RULE_TYPE_TRAINING_STALE)

        run.heartbeat_at = datetime.now(timezone.utc)
        self.db.commit()
        out2 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out2["resolved"]), 1)

        _, total_active2 = self.svc.list_alerts(
            self.db,
            status=AlarmService.STATUS_ACTIVE,
            rule_type=AlarmService.RULE_TYPE_TRAINING_STALE,
            source_id=run.run_id,
        )
        self.assertEqual(total_active2, 0)

    def test_dedupe_and_cooldown(self) -> None:
        run = self._create_run(status=TrainingRunStatus.FAILED, error_message="err")
        out1 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out1["triggered_new"]), 1)

        out2 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out2["triggered_new"]), 0)

        active, total_active = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_ACTIVE, rule_type=AlarmService.RULE_TYPE_TRAINING_FAILED, source_id=run.run_id
        )
        self.assertEqual(total_active, 1)
        self.assertEqual(int(active[0].trigger_count), 1)

        active[0].last_triggered_at = datetime.now(timezone.utc) - timedelta(seconds=601)
        self.db.commit()
        out3 = self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        self.assertEqual(int(out3["touched_active"]), 1)

        active2, _ = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_ACTIVE, rule_type=AlarmService.RULE_TYPE_TRAINING_FAILED, source_id=run.run_id
        )
        self.assertEqual(int(active2[0].trigger_count), 2)

    def test_ack_keeps_active_status(self) -> None:
        run = self._create_run(status=TrainingRunStatus.FAILED, error_message="boom")
        self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        active, total_active = self.svc.list_alerts(
            self.db, status=AlarmService.STATUS_ACTIVE, source_id=run.run_id
        )
        self.assertEqual(total_active, 1)
        acked = self.svc.ack_alert(self.db, int(active[0].alert_id), acked_by="tester")
        self.assertEqual(acked.status, AlarmService.STATUS_ACTIVE)
        self.assertEqual(acked.acked_by, "tester")
        self.assertIsNotNone(acked.acked_at)

        run.status = TrainingRunStatus.COMPLETED
        self.db.commit()
        self.svc.evaluate_training_rules(self.db, run_ids=[run.run_id])
        hist, total_hist = self.svc.list_alerts(self.db, status=AlarmService.STATUS_RESOLVED, source_id=run.run_id)
        self.assertEqual(total_hist, 1)
        with self.assertRaises(ConflictError):
            self.svc.ack_alert(self.db, int(hist[0].alert_id), acked_by="tester")


if __name__ == "__main__":
    unittest.main()
