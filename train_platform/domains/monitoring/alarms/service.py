from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.models.v3.alarm import AlarmAlert, AlarmRule
from train_platform.utils.exceptions import ConflictError, NotFoundError

from .catalog import (
    ALLOWED_SEVERITIES,
    RULE_CATALOG,
    STATUS_ACTIVE,
    validate_cooldown,
    validate_rule_type,
    validate_severity,
    normalize_config,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_default_rules(db: Session) -> None:
    existing = {str(row[0]) for row in db.query(AlarmRule.rule_type).all()}
    defaults = []
    for rule_type, meta in RULE_CATALOG.items():
        if rule_type in existing:
            continue
        defaults.append(
            AlarmRule(
                rule_type=rule_type,
                name=str(meta["name"]),
                description=str(meta["description"]),
                severity=str(meta["default_severity"]),
                enabled=bool(meta["default_enabled"]),
                cooldown_seconds=int(meta["default_cooldown_seconds"]),
                config={},
            )
        )
    if defaults:
        db.add_all(defaults)
        db.commit()


def list_rules(
    db: Session,
    *,
    enabled: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[AlarmRule], int]:
    query = db.query(AlarmRule)
    if enabled is not None:
        query = query.filter(AlarmRule.enabled == bool(enabled))
    total = int(query.count())
    items = (
        query.order_by(AlarmRule.rule_id.asc())
        .offset(max(0, int(skip)))
        .limit(max(1, int(limit)))
        .all()
    )
    for row in items:
        if not isinstance(row.config, dict):
            row.config = {}
    return items, total


def create_rule(db: Session, *, obj: dict[str, Any]) -> AlarmRule:
    rule_type = validate_rule_type(obj.get("rule_type"))
    if db.query(AlarmRule).filter(AlarmRule.rule_type == rule_type).first():
        raise ConflictError(f"Rule already exists for type: {rule_type}")

    severity = validate_severity(obj.get("severity"))
    cooldown = validate_cooldown(obj.get("cooldown_seconds"))
    config = normalize_config(rule_type, obj.get("config"))
    row = AlarmRule(
        rule_type=rule_type,
        name=str(obj.get("name") or "").strip() or str(RULE_CATALOG[rule_type]["name"]),
        description=str(obj.get("description") or "").strip() or None,
        severity=severity,
        enabled=bool(obj.get("enabled", True)),
        cooldown_seconds=cooldown,
        config=config,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_rule(db: Session, rule_id: int) -> AlarmRule:
    row = db.query(AlarmRule).filter(AlarmRule.rule_id == int(rule_id)).first()
    if not row:
        raise NotFoundError("Alarm rule not found")
    if not isinstance(row.config, dict):
        row.config = {}
    return row


def update_rule(db: Session, rule_id: int, *, patch: dict[str, Any]) -> AlarmRule:
    row = get_rule(db, int(rule_id))
    if "name" in patch and patch["name"] is not None:
        row.name = str(patch["name"]).strip()
    if "description" in patch:
        raw_description = patch["description"]
        row.description = str(raw_description).strip() if raw_description is not None else None
    if "severity" in patch and patch["severity"] is not None:
        row.severity = validate_severity(patch["severity"])
    if "enabled" in patch and patch["enabled"] is not None:
        row.enabled = bool(patch["enabled"])
    if "cooldown_seconds" in patch and patch["cooldown_seconds"] is not None:
        row.cooldown_seconds = validate_cooldown(patch["cooldown_seconds"])
    if "config" in patch and patch["config"] is not None:
        row.config = normalize_config(str(row.rule_type), patch["config"])
    db.commit()
    db.refresh(row)
    return row


def delete_rule(db: Session, rule_id: int) -> None:
    row = get_rule(db, int(rule_id))
    db.delete(row)
    db.commit()


def list_alerts(
    db: Session,
    *,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    rule_type: Optional[str] = None,
    source_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[AlarmAlert], int]:
    query = db.query(AlarmAlert)
    if status:
        query = query.filter(AlarmAlert.status == str(status))
    if severity:
        query = query.filter(AlarmAlert.severity == str(severity))
    if rule_type:
        query = query.filter(AlarmAlert.rule_type == str(rule_type))
    if source_id:
        query = query.filter(AlarmAlert.source_id == str(source_id))

    total = int(query.count())
    items = (
        query.order_by(AlarmAlert.last_triggered_at.desc(), AlarmAlert.alert_id.desc())
        .offset(max(0, int(skip)))
        .limit(max(1, int(limit)))
        .all()
    )
    for row in items:
        if not isinstance(row.payload, dict):
            row.payload = {}
    return items, total


def ack_alert(db: Session, alert_id: int, *, acked_by: Optional[str] = None) -> AlarmAlert:
    row = db.query(AlarmAlert).filter(AlarmAlert.alert_id == int(alert_id)).first()
    if not row:
        raise NotFoundError("Alarm alert not found")
    if str(row.status) != STATUS_ACTIVE:
        raise ConflictError("Only active alerts can be acknowledged")
    if not isinstance(row.payload, dict):
        row.payload = {}
    row.acked_at = _utcnow()
    row.acked_by = str(acked_by).strip() if acked_by else None
    db.commit()
    db.refresh(row)
    return row


def get_summary(db: Session) -> dict[str, Any]:
    query = db.query(AlarmAlert).filter(AlarmAlert.status == STATUS_ACTIVE)
    total = int(query.count())
    rows = query.with_entities(AlarmAlert.severity, func.count(AlarmAlert.alert_id)).group_by(AlarmAlert.severity).all()
    by_severity = {str(severity): int(count) for severity, count in rows}
    for severity in ALLOWED_SEVERITIES:
        by_severity.setdefault(severity, 0)
    return {"active_total": total, "by_severity": by_severity}
