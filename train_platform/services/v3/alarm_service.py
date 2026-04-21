from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.models.v3.alarm import AlarmAlert, AlarmRule
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class AlarmService:
    STATUS_ACTIVE = "active"
    STATUS_RESOLVED = "resolved"
    SOURCE_TRAINING_RUN = "training_run"

    RULE_TYPE_TRAINING_FAILED = "training_run_failed"
    RULE_TYPE_TRAINING_STALE = "training_run_stale"

    ALLOWED_RULE_TYPES = {
        RULE_TYPE_TRAINING_FAILED,
        RULE_TYPE_TRAINING_STALE,
    }
    ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}

    RULE_CATALOG: Dict[str, Dict[str, Any]] = {
        RULE_TYPE_TRAINING_FAILED: {
            "name": "训练任务失败",
            "description": "当训练任务状态变为 failed 时触发。",
            "default_severity": "high",
            "default_enabled": True,
            "default_cooldown_seconds": 300,
            "config_schema": {},
        },
        RULE_TYPE_TRAINING_STALE: {
            "name": "训练任务心跳超时",
            "description": "训练任务处于 running 且心跳超过阈值未更新时触发。",
            "default_severity": "high",
            "default_enabled": True,
            "default_cooldown_seconds": 300,
            "config_schema": {
                "stale_after_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "default": int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120")),
                    "description": "覆盖系统默认心跳超时秒数。",
                }
            },
        },
    }

    def list_rule_types(self) -> List[Dict[str, Any]]:
        out = []
        for rule_type, meta in self.RULE_CATALOG.items():
            out.append({"rule_type": rule_type, **meta})
        return out

    def ensure_default_rules(self, db: Session) -> None:
        existing = {str(x[0]) for x in db.query(AlarmRule.rule_type).all()}
        dirty = False
        for rule_type, meta in self.RULE_CATALOG.items():
            if rule_type in existing:
                continue
            db.add(
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
            dirty = True
        if dirty:
            db.commit()

    def list_rules(self, db: Session, *, enabled: Optional[bool] = None, skip: int = 0, limit: int = 100) -> tuple[List[AlarmRule], int]:
        q = db.query(AlarmRule)
        if enabled is not None:
            q = q.filter(AlarmRule.enabled == bool(enabled))
        total = int(q.count())
        items = q.order_by(AlarmRule.rule_id.asc()).offset(max(0, int(skip))).limit(max(1, int(limit))).all()
        for row in items:
            if not isinstance(row.config, dict):
                row.config = {}
        return items, total

    def create_rule(self, db: Session, *, obj: Dict[str, Any]) -> AlarmRule:
        rule_type = str(obj.get("rule_type") or "").strip()
        if rule_type not in self.ALLOWED_RULE_TYPES:
            raise ValidationError(f"Unsupported rule_type: {rule_type}")
        if db.query(AlarmRule).filter(AlarmRule.rule_type == rule_type).first():
            raise ConflictError(f"Rule already exists for type: {rule_type}")

        severity = self._validate_severity(obj.get("severity"))
        cooldown = self._validate_cooldown(obj.get("cooldown_seconds"))
        config = self._normalize_config(rule_type, obj.get("config"))

        row = AlarmRule(
            rule_type=rule_type,
            name=str(obj.get("name") or "").strip() or self.RULE_CATALOG[rule_type]["name"],
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

    def get_rule(self, db: Session, rule_id: int) -> AlarmRule:
        row = db.query(AlarmRule).filter(AlarmRule.rule_id == int(rule_id)).first()
        if not row:
            raise NotFoundError("Alarm rule not found")
        if not isinstance(row.config, dict):
            row.config = {}
        return row

    def update_rule(self, db: Session, rule_id: int, *, patch: Dict[str, Any]) -> AlarmRule:
        row = self.get_rule(db, int(rule_id))

        if "name" in patch and patch["name"] is not None:
            row.name = str(patch["name"]).strip()
        if "description" in patch:
            raw = patch["description"]
            row.description = str(raw).strip() if raw is not None else None
        if "severity" in patch and patch["severity"] is not None:
            row.severity = self._validate_severity(patch["severity"])
        if "enabled" in patch and patch["enabled"] is not None:
            row.enabled = bool(patch["enabled"])
        if "cooldown_seconds" in patch and patch["cooldown_seconds"] is not None:
            row.cooldown_seconds = self._validate_cooldown(patch["cooldown_seconds"])
        if "config" in patch and patch["config"] is not None:
            row.config = self._normalize_config(str(row.rule_type), patch["config"])

        db.commit()
        db.refresh(row)
        return row

    def delete_rule(self, db: Session, rule_id: int) -> None:
        row = self.get_rule(db, int(rule_id))
        db.delete(row)
        db.commit()

    def list_alerts(
        self,
        db: Session,
        *,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        rule_type: Optional[str] = None,
        source_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[List[AlarmAlert], int]:
        q = db.query(AlarmAlert)
        if status:
            q = q.filter(AlarmAlert.status == str(status))
        if severity:
            q = q.filter(AlarmAlert.severity == str(severity))
        if rule_type:
            q = q.filter(AlarmAlert.rule_type == str(rule_type))
        if source_id:
            q = q.filter(AlarmAlert.source_id == str(source_id))

        total = int(q.count())
        items = (
            q.order_by(AlarmAlert.last_triggered_at.desc(), AlarmAlert.alert_id.desc())
            .offset(max(0, int(skip)))
            .limit(max(1, int(limit)))
            .all()
        )
        for row in items:
            if not isinstance(row.payload, dict):
                row.payload = {}
        return items, total

    def ack_alert(self, db: Session, alert_id: int, *, acked_by: Optional[str] = None) -> AlarmAlert:
        row = db.query(AlarmAlert).filter(AlarmAlert.alert_id == int(alert_id)).first()
        if not row:
            raise NotFoundError("Alarm alert not found")
        if str(row.status) != self.STATUS_ACTIVE:
            raise ConflictError("Only active alerts can be acknowledged")
        if not isinstance(row.payload, dict):
            row.payload = {}
        row.acked_at = _utcnow()
        row.acked_by = str(acked_by).strip() if acked_by else None
        db.commit()
        db.refresh(row)
        return row

    def get_summary(self, db: Session) -> Dict[str, Any]:
        q = db.query(AlarmAlert).filter(AlarmAlert.status == self.STATUS_ACTIVE)
        total = int(q.count())
        rows = q.with_entities(AlarmAlert.severity, func.count(AlarmAlert.alert_id)).group_by(AlarmAlert.severity).all()
        by_severity = {str(s): int(c) for s, c in rows}
        for s in self.ALLOWED_SEVERITIES:
            by_severity.setdefault(s, 0)
        return {"active_total": total, "by_severity": by_severity}

    def evaluate_training_rules(self, db: Session, *, run_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        self.ensure_default_rules(db)
        rules = (
            db.query(AlarmRule)
            .filter(AlarmRule.enabled == True)  # noqa: E712
            .filter(AlarmRule.rule_type.in_(sorted(self.ALLOWED_RULE_TYPES)))
            .order_by(AlarmRule.rule_id.asc())
            .all()
        )
        if not rules:
            return {
                "evaluated_runs": 0,
                "triggered_new": 0,
                "touched_active": 0,
                "resolved": 0,
                "active_total": int(
                    db.query(AlarmAlert)
                    .filter(AlarmAlert.status == self.STATUS_ACTIVE)
                    .filter(AlarmAlert.source_type == self.SOURCE_TRAINING_RUN)
                    .count()
                ),
                "timestamp": _utcnow(),
            }

        target_ids = self._collect_target_run_ids(db, run_ids=run_ids)
        now = _utcnow()
        if not target_ids:
            return {
                "evaluated_runs": 0,
                "triggered_new": 0,
                "touched_active": 0,
                "resolved": 0,
                "active_total": int(
                    db.query(AlarmAlert)
                    .filter(AlarmAlert.status == self.STATUS_ACTIVE)
                    .filter(AlarmAlert.source_type == self.SOURCE_TRAINING_RUN)
                    .count()
                ),
                "timestamp": now,
            }

        run_map = {
            str(r.run_id): r
            for r in db.query(TrainingRun).filter(TrainingRun.run_id.in_(sorted(target_ids))).all()
        }
        active_alerts = (
            db.query(AlarmAlert)
            .filter(AlarmAlert.status == self.STATUS_ACTIVE)
            .filter(AlarmAlert.source_type == self.SOURCE_TRAINING_RUN)
            .filter(AlarmAlert.source_id.in_(sorted(target_ids)))
            .filter(AlarmAlert.rule_type.in_([str(r.rule_type) for r in rules]))
            .all()
        )
        active_index: Dict[Tuple[str, str], AlarmAlert] = {
            (str(a.rule_type), str(a.source_id)): a for a in active_alerts
        }

        triggered_new = 0
        touched_active = 0
        resolved = 0

        for source_id in sorted(target_ids):
            run = run_map.get(source_id)
            for rule in rules:
                key = (str(rule.rule_type), source_id)
                active = active_index.get(key)
                matched, title, message, payload = self._match_rule(rule=rule, run=run, now=now)
                if matched:
                    if active is None:
                        created = AlarmAlert(
                            rule_id=int(rule.rule_id),
                            rule_type=str(rule.rule_type),
                            severity=str(rule.severity),
                            status=self.STATUS_ACTIVE,
                            title=title,
                            message=message,
                            source_type=self.SOURCE_TRAINING_RUN,
                            source_id=source_id,
                            trigger_count=1,
                            first_triggered_at=now,
                            last_triggered_at=now,
                            resolved_at=None,
                            payload=payload,
                        )
                        db.add(created)
                        db.flush()
                        active_index[key] = created
                        triggered_new += 1
                    else:
                        if self._should_touch_active(active=active, rule=rule, now=now):
                            active.last_triggered_at = now
                            active.trigger_count = int(active.trigger_count or 0) + 1
                            active.title = title
                            active.message = message
                            active.payload = payload
                            active.severity = str(rule.severity)
                            touched_active += 1
                else:
                    if active is not None:
                        active.status = self.STATUS_RESOLVED
                        active.resolved_at = now
                        resolved += 1
                        active_index.pop(key, None)

        db.commit()
        active_total = int(
            db.query(AlarmAlert)
            .filter(AlarmAlert.status == self.STATUS_ACTIVE)
            .filter(AlarmAlert.source_type == self.SOURCE_TRAINING_RUN)
            .count()
        )
        return {
            "evaluated_runs": len(target_ids),
            "triggered_new": triggered_new,
            "touched_active": touched_active,
            "resolved": resolved,
            "active_total": active_total,
            "timestamp": now,
        }

    @classmethod
    def try_evaluate_training_rules(cls, db: Session, *, run_ids: Iterable[str]) -> None:
        try:
            cls().evaluate_training_rules(db, run_ids=run_ids)
        except Exception:
            # Alarm evaluation should not block core train/infer workflows.
            pass

    def _collect_target_run_ids(self, db: Session, *, run_ids: Optional[Iterable[str]]) -> set[str]:
        explicit_ids = {str(x).strip() for x in (run_ids or []) if str(x).strip()}
        if explicit_ids:
            return explicit_ids

        out: set[str] = set()
        active_rows = (
            db.query(AlarmAlert.source_id)
            .filter(AlarmAlert.status == self.STATUS_ACTIVE)
            .filter(AlarmAlert.source_type == self.SOURCE_TRAINING_RUN)
            .filter(AlarmAlert.rule_type.in_(sorted(self.ALLOWED_RULE_TYPES)))
            .all()
        )
        out.update(str(x[0]) for x in active_rows if str(x[0]).strip())

        run_rows = (
            db.query(TrainingRun.run_id)
            .filter(TrainingRun.status.in_([TrainingRunStatus.FAILED, TrainingRunStatus.RUNNING]))
            .all()
        )
        out.update(str(x[0]) for x in run_rows if str(x[0]).strip())
        return out

    def _match_rule(
        self,
        *,
        rule: AlarmRule,
        run: Optional[TrainingRun],
        now: datetime,
    ) -> tuple[bool, str, str, Dict[str, Any]]:
        rule_type = str(rule.rule_type)
        if rule_type == self.RULE_TYPE_TRAINING_FAILED:
            if run and run.status == TrainingRunStatus.FAILED:
                err = str(getattr(run, "error_message", "") or "").strip() or "Unknown training failure"
                title = "训练任务失败"
                message = f"run_id={run.run_id} 失败：{err}"
                payload = {
                    "run_id": str(run.run_id),
                    "status": str(getattr(run.status, "value", run.status)),
                    "error_message": err,
                    "finished_at": _ensure_aware_utc(getattr(run, "finished_at", None)).isoformat()
                    if getattr(run, "finished_at", None)
                    else None,
                }
                return True, title, message, payload
            return False, "训练任务失败", "", {}

        if rule_type == self.RULE_TYPE_TRAINING_STALE:
            stale_after = self._resolve_stale_after_seconds(rule)
            if not run or run.status != TrainingRunStatus.RUNNING:
                return False, "训练任务心跳超时", "", {}

            hb = _ensure_aware_utc(getattr(run, "heartbeat_at", None))
            started = _ensure_aware_utc(getattr(run, "started_at", None))
            pivot = hb or started
            if pivot is None:
                return False, "训练任务心跳超时", "", {}

            age = (now - pivot).total_seconds()
            if age <= float(stale_after):
                return False, "训练任务心跳超时", "", {}

            title = "训练任务心跳超时"
            message = f"run_id={run.run_id} 心跳超时 {int(age)}s（阈值 {int(stale_after)}s）"
            payload = {
                "run_id": str(run.run_id),
                "status": str(getattr(run.status, "value", run.status)),
                "stale_after_seconds": int(stale_after),
                "age_seconds": int(age),
                "heartbeat_at": hb.isoformat() if hb else None,
                "started_at": started.isoformat() if started else None,
            }
            return True, title, message, payload

        return False, "未知规则", "", {}

    def _resolve_stale_after_seconds(self, rule: AlarmRule) -> int:
        fallback = int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120"))
        cfg = rule.config if isinstance(rule.config, dict) else {}
        raw = cfg.get("stale_after_seconds")
        if raw is None:
            return max(1, int(fallback))
        try:
            v = int(raw)
        except Exception:
            raise ValidationError("config.stale_after_seconds must be an integer")
        if v < 1 or v > 86400:
            raise ValidationError("config.stale_after_seconds must be between 1 and 86400")
        return v

    def _normalize_config(self, rule_type: str, raw: Any) -> Dict[str, Any]:
        if raw is None:
            cfg: Dict[str, Any] = {}
        elif isinstance(raw, dict):
            cfg = dict(raw)
        else:
            raise ValidationError("config must be a JSON object")

        if rule_type == self.RULE_TYPE_TRAINING_STALE and "stale_after_seconds" in cfg:
            try:
                v = int(cfg.get("stale_after_seconds"))
            except Exception:
                raise ValidationError("config.stale_after_seconds must be an integer")
            if v < 1 or v > 86400:
                raise ValidationError("config.stale_after_seconds must be between 1 and 86400")
            cfg["stale_after_seconds"] = v
        return cfg

    def _validate_severity(self, raw: Any) -> str:
        v = str(raw or "").strip().lower()
        if v not in self.ALLOWED_SEVERITIES:
            raise ValidationError(f"Invalid severity: {raw}")
        return v

    def _validate_cooldown(self, raw: Any) -> int:
        try:
            v = int(raw)
        except Exception:
            raise ValidationError("cooldown_seconds must be an integer")
        if v < 0 or v > 86400:
            raise ValidationError("cooldown_seconds must be between 0 and 86400")
        return v

    def _should_touch_active(self, *, active: AlarmAlert, rule: AlarmRule, now: datetime) -> bool:
        cooldown = max(0, int(rule.cooldown_seconds or 0))
        if cooldown == 0:
            return True

        last = _ensure_aware_utc(active.last_triggered_at) or _ensure_aware_utc(active.first_triggered_at) or now
        return (now - last).total_seconds() >= float(cooldown)
