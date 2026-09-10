from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import train_platform.models.v3  # noqa: F401 - register complete V3 metadata
from train_platform.api.deps import get_db
from train_platform.api.v3.alarms import router as alarms_router
from train_platform.domains.monitoring.alarms.catalog import (
    RULE_TYPE_TRAINING_FAILED,
    RULE_TYPE_TRAINING_STALE,
    SOURCE_TRAINING_RUN,
    STATUS_ACTIVE,
    STATUS_RESOLVED,
)
from train_platform.domains.monitoring.alarms.service import (
    ack_alert,
    create_rule,
    delete_rule,
    get_summary,
)
from train_platform.domains.monitoring.alarms.training import evaluate_training_alerts
from train_platform.models.v3.alarm import AlarmAlert, AlarmRule
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.base import V3Base
from train_platform.models.v3.enums import DatasetType, TaskType, TrainingRunStatus
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError


@pytest.fixture
def alarm_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    V3Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    db = session_factory()
    try:
        yield db, engine, session_factory
    finally:
        db.close()
        engine.dispose()


def rule_payload(rule_type: str, **overrides):
    payload = {
        "rule_type": rule_type,
        "name": f"custom {rule_type}",
        "description": "custom description",
        "severity": "critical",
        "enabled": True,
        "cooldown_seconds": 17,
        "config": {},
    }
    payload.update(overrides)
    return payload


def add_active_alert(db, rule: AlarmRule, *, source_id: str = "run-1") -> AlarmAlert:
    now = datetime.now(timezone.utc)
    alert = AlarmAlert(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        severity=rule.severity,
        status=STATUS_ACTIVE,
        title="preserved title",
        message="preserved message",
        source_type=SOURCE_TRAINING_RUN,
        source_id=source_id,
        trigger_count=1,
        first_triggered_at=now,
        last_triggered_at=now,
        payload={"preserved": True},
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def run_init_db(monkeypatch, engine, session_factory) -> None:
    init_db_module = importlib.import_module("train_platform.db.init_db")
    monkeypatch.setattr(init_db_module, "engine", engine)
    monkeypatch.setattr(init_db_module, "SessionLocal", session_factory)
    init_db_module.init_db()


def test_empty_database_evaluate_and_startup_do_not_create_rules(alarm_db, monkeypatch):
    db, engine, session_factory = alarm_db

    result = evaluate_training_alerts(db, run_ids=[])
    assert result["evaluated_runs"] == 0
    assert result["triggered_new"] == 0
    assert result["resolved"] == 0
    assert db.query(AlarmRule).count() == 0

    run_init_db(monkeypatch, engine, session_factory)
    db.expire_all()
    assert db.query(AlarmRule).count() == 0


def test_deleted_rule_stays_deleted_and_can_be_recreated(alarm_db, monkeypatch):
    db, engine, session_factory = alarm_db
    rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))

    with pytest.raises(ConflictError):
        create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))

    delete_rule(db, rule.rule_id)
    assert db.query(AlarmRule).count() == 0

    evaluate_training_alerts(db, run_ids=[])
    assert db.query(AlarmRule).count() == 0

    run_init_db(monkeypatch, engine, session_factory)
    db.expire_all()
    assert db.query(AlarmRule).count() == 0

    recreated = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))
    assert recreated.rule_type == RULE_TYPE_TRAINING_FAILED


def test_alarm_rule_api_delete_is_persistent_across_evaluate_and_startup(alarm_db, monkeypatch):
    db, engine, session_factory = alarm_db
    db.close()
    app = FastAPI()
    app.include_router(alarms_router, prefix="/api/v3")

    def override_get_db():
        request_db = session_factory()
        try:
            yield request_db
        finally:
            request_db.close()

    app.dependency_overrides[get_db] = override_get_db

    @app.exception_handler(ConflictError)
    async def conflict_handler(_request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    payload = rule_payload(RULE_TYPE_TRAINING_FAILED)
    with TestClient(app) as client:
        catalog = client.get("/api/v3/alarms/rule-types")
        assert catalog.status_code == 200
        assert {item["rule_type"] for item in catalog.json()} == {
            RULE_TYPE_TRAINING_FAILED,
            RULE_TYPE_TRAINING_STALE,
        }

        response = client.get("/api/v3/alarms/rules")
        assert response.status_code == 200
        assert response.json()["items"] == []

        created = client.post("/api/v3/alarms/rules", json=payload)
        assert created.status_code == 201
        rule_id = created.json()["rule_id"]
        assert client.post("/api/v3/alarms/rules", json=payload).status_code == 409

        assert client.delete(f"/api/v3/alarms/rules/{rule_id}").status_code == 200
        assert client.get("/api/v3/alarms/rules").json()["items"] == []

        evaluated = client.post("/api/v3/alarms/evaluate", json={"run_ids": []})
        assert evaluated.status_code == 200
        assert evaluated.json()["evaluated_runs"] == 0
        assert client.get("/api/v3/alarms/rules").json()["items"] == []

        run_init_db(monkeypatch, engine, session_factory)
        assert client.get("/api/v3/alarms/rules").json()["items"] == []
        assert client.post("/api/v3/alarms/rules", json=payload).status_code == 201


def test_startup_preserves_existing_rule_configuration(alarm_db, monkeypatch):
    db, engine, session_factory = alarm_db
    failed = create_rule(
        db,
        obj=rule_payload(
            RULE_TYPE_TRAINING_FAILED,
            name="failed custom",
            description="failed description",
            severity="low",
            enabled=False,
            cooldown_seconds=41,
            config={"custom": "value"},
        ),
    )
    stale = create_rule(
        db,
        obj=rule_payload(
            RULE_TYPE_TRAINING_STALE,
            name="stale custom",
            description="stale description",
            severity="medium",
            cooldown_seconds=73,
            config={"stale_after_seconds": 9},
        ),
    )
    expected = {
        failed.rule_id: ("failed custom", "failed description", "low", False, 41, {"custom": "value"}),
        stale.rule_id: ("stale custom", "stale description", "medium", True, 73, {"stale_after_seconds": 9}),
    }

    run_init_db(monkeypatch, engine, session_factory)
    db.expire_all()

    rows = db.query(AlarmRule).order_by(AlarmRule.rule_id).all()
    assert len(rows) == 2
    for row in rows:
        assert (row.name, row.description, row.severity, row.enabled, row.cooldown_seconds, row.config) == expected[row.rule_id]


@pytest.mark.parametrize("enabled", [True, False])
def test_delete_resolves_active_alert_and_preserves_history(alarm_db, enabled):
    db, _, _ = alarm_db
    rule = create_rule(
        db,
        obj=rule_payload(RULE_TYPE_TRAINING_FAILED, enabled=enabled),
    )
    alert = add_active_alert(db, rule)
    alert_id = alert.alert_id

    delete_rule(db, rule.rule_id)
    db.expire_all()

    assert db.query(AlarmRule).filter(AlarmRule.rule_id == rule.rule_id).first() is None
    preserved = db.query(AlarmAlert).filter(AlarmAlert.alert_id == alert_id).one()
    assert preserved.rule_id is None
    assert preserved.status == STATUS_RESOLVED
    assert preserved.resolved_at is not None
    assert preserved.rule_type == RULE_TYPE_TRAINING_FAILED
    assert preserved.title == "preserved title"
    assert preserved.message == "preserved message"
    assert preserved.source_id == "run-1"
    assert preserved.payload == {"preserved": True}
    assert get_summary(db)["active_total"] == 0


def test_ack_fields_survive_rule_deletion(alarm_db):
    db, _, _ = alarm_db
    rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))
    alert = add_active_alert(db, rule)
    ack_alert(db, alert.alert_id, acked_by="operator")
    acked_at = alert.acked_at
    assert alert.status == STATUS_ACTIVE

    delete_rule(db, rule.rule_id)
    db.expire_all()

    preserved = db.query(AlarmAlert).filter(AlarmAlert.alert_id == alert.alert_id).one()
    assert preserved.status == STATUS_RESOLVED
    assert preserved.resolved_at is not None
    assert preserved.acked_at == acked_at
    assert preserved.acked_by == "operator"


def test_recreated_rule_starts_a_new_alert_lifecycle(alarm_db):
    db, _, _ = alarm_db
    dataset = StandardDataset(
        name="alarm dataset",
        dataset_type=DatasetType.DETECTION,
        format="yolo",
        storage_path="datasets/alarm",
    )
    architecture = ModelArchitecture(
        family="alarm family",
        variant="alarm variant",
        task_type=TaskType.DETECTION,
        engine="ultralytics-yolo",
    )
    db.add_all([dataset, architecture])
    db.flush()
    project = Project(
        name="alarm project",
        standard_dataset_id=dataset.standard_dataset_id,
        task_type=TaskType.DETECTION,
    )
    db.add(project)
    db.flush()
    run = TrainingRun(
        run_id="failed-run",
        project_id=project.project_id,
        standard_dataset_id=dataset.standard_dataset_id,
        architecture_id=architecture.architecture_id,
        name="failed run",
        status=TrainingRunStatus.FAILED,
        error_message="boom",
        finished_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    first_rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))
    first_result = evaluate_training_alerts(db, run_ids=[run.run_id])
    assert first_result["triggered_new"] == 1
    first_alert = db.query(AlarmAlert).filter(AlarmAlert.rule_id == first_rule.rule_id).one()

    delete_rule(db, first_rule.rule_id)
    db.refresh(first_alert)
    assert first_alert.status == STATUS_RESOLVED
    assert first_alert.rule_id is None
    assert evaluate_training_alerts(db, run_ids=[run.run_id])["triggered_new"] == 0

    second_rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))
    second_result = evaluate_training_alerts(db, run_ids=[run.run_id])
    assert second_result["triggered_new"] == 1
    second_alert = db.query(AlarmAlert).filter(AlarmAlert.rule_id == second_rule.rule_id).one()
    assert second_alert.status == STATUS_ACTIVE
    assert second_alert.alert_id != first_alert.alert_id


def test_delete_only_resolves_alerts_owned_by_deleted_rule(alarm_db):
    db, _, _ = alarm_db
    failed_rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_FAILED))
    stale_rule = create_rule(db, obj=rule_payload(RULE_TYPE_TRAINING_STALE))
    failed_alert = add_active_alert(db, failed_rule, source_id="failed-run")
    stale_alert = add_active_alert(db, stale_rule, source_id="stale-run")

    delete_rule(db, failed_rule.rule_id)
    db.expire_all()

    assert db.get(AlarmAlert, failed_alert.alert_id).status == STATUS_RESOLVED
    assert db.get(AlarmAlert, stale_alert.alert_id).status == STATUS_ACTIVE
    assert db.get(AlarmAlert, stale_alert.alert_id).rule_id == stale_rule.rule_id
    assert get_summary(db)["active_total"] == 1


def test_no_enabled_rules_leave_existing_orphan_alert_unchanged(alarm_db):
    db, _, _ = alarm_db
    create_rule(
        db,
        obj=rule_payload(RULE_TYPE_TRAINING_FAILED, enabled=False),
    )
    now = datetime.now(timezone.utc)
    orphan = AlarmAlert(
        rule_id=None,
        rule_type=RULE_TYPE_TRAINING_FAILED,
        severity="high",
        status=STATUS_ACTIVE,
        title="legacy orphan",
        message="legacy",
        source_type=SOURCE_TRAINING_RUN,
        source_id="legacy-run",
        trigger_count=1,
        first_triggered_at=now,
        last_triggered_at=now,
        payload={},
    )
    db.add(orphan)
    db.commit()

    result = evaluate_training_alerts(db, run_ids=[])

    assert result["resolved"] == 0
    assert result["active_total"] == 1
    assert db.get(AlarmAlert, orphan.alert_id).status == STATUS_ACTIVE
