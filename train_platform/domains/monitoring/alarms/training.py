from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.alarm import AlarmAlert, AlarmRule
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun

from .catalog import (
    ALLOWED_RULE_TYPES,
    RULE_TYPE_TRAINING_FAILED,
    RULE_TYPE_TRAINING_STALE,
    SOURCE_TRAINING_RUN,
    STATUS_ACTIVE,
    resolve_stale_after_seconds,
)
from .service import ensure_default_rules


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def evaluate_failed(rule: AlarmRule, run: Optional[TrainingRun], now: datetime) -> dict[str, Any]:
    title = "训练任务失败"
    if not run or run.status != TrainingRunStatus.FAILED:
        return {"matched": False, "title": title, "message": "", "payload": {}}

    error_message = str(getattr(run, "error_message", "") or "").strip() or "Unknown training failure"
    finished_at = _ensure_aware_utc(getattr(run, "finished_at", None))
    return {
        "matched": True,
        "title": title,
        "message": f"run_id={run.run_id} 失败：{error_message}",
        "payload": {
            "run_id": str(run.run_id),
            "status": str(getattr(run.status, "value", run.status)),
            "error_message": error_message,
            "finished_at": finished_at.isoformat() if finished_at else None,
        },
    }


def evaluate_stale(rule: AlarmRule, run: Optional[TrainingRun], now: datetime) -> dict[str, Any]:
    title = "训练任务心跳超时"
    stale_after = resolve_stale_after_seconds(getattr(rule, "config", None))
    if not run or run.status != TrainingRunStatus.RUNNING:
        return {"matched": False, "title": title, "message": "", "payload": {}}

    heartbeat_at = _ensure_aware_utc(getattr(run, "heartbeat_at", None))
    started_at = _ensure_aware_utc(getattr(run, "started_at", None))
    pivot = heartbeat_at or started_at
    if pivot is None:
        return {"matched": False, "title": title, "message": "", "payload": {}}

    age = (now - pivot).total_seconds()
    if age <= float(stale_after):
        return {"matched": False, "title": title, "message": "", "payload": {}}

    return {
        "matched": True,
        "title": title,
        "message": f"run_id={run.run_id} 心跳超时 {int(age)}s（阈值 {int(stale_after)}s）",
        "payload": {
            "run_id": str(run.run_id),
            "status": str(getattr(run.status, "value", run.status)),
            "stale_after_seconds": int(stale_after),
            "age_seconds": int(age),
            "heartbeat_at": heartbeat_at.isoformat() if heartbeat_at else None,
            "started_at": started_at.isoformat() if started_at else None,
        },
    }


def _evaluate_rule(*, rule: AlarmRule, run: Optional[TrainingRun], now: datetime) -> dict[str, Any]:
    if str(rule.rule_type) == RULE_TYPE_TRAINING_FAILED:
        return evaluate_failed(rule=rule, run=run, now=now)
    if str(rule.rule_type) == RULE_TYPE_TRAINING_STALE:
        return evaluate_stale(rule=rule, run=run, now=now)
    return {"matched": False, "title": "未知规则", "message": "", "payload": {}}


def _collect_target_run_ids(db: Session, *, run_ids: Optional[Iterable[str]]) -> set[str]:
    explicit_ids = {str(value).strip() for value in (run_ids or []) if str(value).strip()}
    if explicit_ids:
        return explicit_ids

    target_ids = {
        str(row[0])
        for row in (
            db.query(AlarmAlert.source_id)
            .filter(AlarmAlert.status == STATUS_ACTIVE)
            .filter(AlarmAlert.source_type == SOURCE_TRAINING_RUN)
            .filter(AlarmAlert.rule_type.in_(sorted(ALLOWED_RULE_TYPES)))
            .all()
        )
        if str(row[0]).strip()
    }
    target_ids.update(
        str(row[0])
        for row in (
            db.query(TrainingRun.run_id)
            .filter(TrainingRun.status.in_([TrainingRunStatus.FAILED, TrainingRunStatus.RUNNING]))
            .all()
        )
        if str(row[0]).strip()
    )
    return target_ids


def _active_total(db: Session) -> int:
    return int(
        db.query(AlarmAlert)
        .filter(AlarmAlert.status == STATUS_ACTIVE)
        .filter(AlarmAlert.source_type == SOURCE_TRAINING_RUN)
        .count()
    )


def _should_touch_active(*, active: AlarmAlert, rule: AlarmRule, now: datetime) -> bool:
    cooldown = max(0, int(rule.cooldown_seconds or 0))
    if cooldown == 0:
        return True
    last = (
        _ensure_aware_utc(active.last_triggered_at)
        or _ensure_aware_utc(active.first_triggered_at)
        or now
    )
    return (now - last).total_seconds() >= float(cooldown)


def evaluate_training_alerts(
    db: Session,
    *,
    run_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    ensure_default_rules(db)
    rules = (
        db.query(AlarmRule)
        .filter(AlarmRule.enabled == True)  # noqa: E712
        .filter(AlarmRule.rule_type.in_(sorted(ALLOWED_RULE_TYPES)))
        .order_by(AlarmRule.rule_id.asc())
        .all()
    )
    now = _utcnow()
    if not rules:
        return {
            "evaluated_runs": 0,
            "triggered_new": 0,
            "touched_active": 0,
            "resolved": 0,
            "active_total": _active_total(db),
            "timestamp": now,
        }

    target_ids = _collect_target_run_ids(db, run_ids=run_ids)
    if not target_ids:
        return {
            "evaluated_runs": 0,
            "triggered_new": 0,
            "touched_active": 0,
            "resolved": 0,
            "active_total": _active_total(db),
            "timestamp": now,
        }

    sorted_target_ids = sorted(target_ids)
    run_map = {
        str(run.run_id): run
        for run in db.query(TrainingRun).filter(TrainingRun.run_id.in_(sorted_target_ids)).all()
    }
    rule_types = [str(rule.rule_type) for rule in rules]
    active_alerts = (
        db.query(AlarmAlert)
        .filter(AlarmAlert.status == STATUS_ACTIVE)
        .filter(AlarmAlert.source_type == SOURCE_TRAINING_RUN)
        .filter(AlarmAlert.source_id.in_(sorted_target_ids))
        .filter(AlarmAlert.rule_type.in_(rule_types))
        .all()
    )
    active_index = {(str(alert.rule_type), str(alert.source_id)): alert for alert in active_alerts}

    triggered_new = 0
    touched_active = 0
    resolved = 0
    for source_id in sorted_target_ids:
        run = run_map.get(source_id)
        for rule in rules:
            key = (str(rule.rule_type), source_id)
            active = active_index.get(key)
            match = _evaluate_rule(rule=rule, run=run, now=now)
            if bool(match["matched"]):
                if active is None:
                    created = AlarmAlert(
                        rule_id=int(rule.rule_id),
                        rule_type=str(rule.rule_type),
                        severity=str(rule.severity),
                        status=STATUS_ACTIVE,
                        title=str(match["title"]),
                        message=str(match["message"]),
                        source_type=SOURCE_TRAINING_RUN,
                        source_id=source_id,
                        trigger_count=1,
                        first_triggered_at=now,
                        last_triggered_at=now,
                        resolved_at=None,
                        payload=match["payload"],
                    )
                    db.add(created)
                    db.flush()
                    active_index[key] = created
                    triggered_new += 1
                elif _should_touch_active(active=active, rule=rule, now=now):
                    active.last_triggered_at = now
                    active.trigger_count = int(active.trigger_count or 0) + 1
                    active.title = str(match["title"])
                    active.message = str(match["message"])
                    active.payload = match["payload"]
                    active.severity = str(rule.severity)
                    touched_active += 1
            elif active is not None:
                active.status = STATUS_RESOLVED
                active.resolved_at = now
                active_index.pop(key, None)
                resolved += 1

    db.commit()
    return {
        "evaluated_runs": len(target_ids),
        "triggered_new": triggered_new,
        "touched_active": touched_active,
        "resolved": resolved,
        "active_total": _active_total(db),
        "timestamp": now,
    }


def evaluate_training_alerts_best_effort(
    db: Session,
    run_ids: Optional[Iterable[str]] = None,
) -> None:
    try:
        evaluate_training_alerts(db, run_ids=run_ids)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("Training alarm evaluation failed", exc_info=True)
