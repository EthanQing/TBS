from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.schemas.v3.alarms import (
    AlarmAckRequest,
    AlarmAlertOut,
    AlarmEvaluateOut,
    AlarmEvaluateRequest,
    AlarmRuleCreate,
    AlarmRuleOut,
    AlarmRuleTypeOut,
    AlarmRuleUpdate,
    AlarmSummaryOut,
)
from train_platform.schemas.v3.common import DeleteResponse, Page, PageMeta
from train_platform.services.v3.alarm_service import AlarmService


router = APIRouter(prefix="/alarms", tags=["alarms"])


@router.get("/rule-types", response_model=list[AlarmRuleTypeOut])
def list_alarm_rule_types():
    return AlarmService().list_rule_types()


@router.get("/rules", response_model=Page[AlarmRuleOut])
def list_alarm_rules(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    enabled: bool | None = Query(None),
    db: Session = Depends(get_db),
):
    svc = AlarmService()
    svc.ensure_default_rules(db)
    skip = (int(page) - 1) * int(page_size)
    items, total = svc.list_rules(db, enabled=enabled, skip=skip, limit=page_size)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=total)}


@router.post("/rules", response_model=AlarmRuleOut, status_code=201)
def create_alarm_rule(payload: AlarmRuleCreate, db: Session = Depends(get_db)):
    return AlarmService().create_rule(db, obj=payload.model_dump())


@router.patch("/rules/{rule_id}", response_model=AlarmRuleOut)
def update_alarm_rule(rule_id: int, payload: AlarmRuleUpdate, db: Session = Depends(get_db)):
    patch = payload.model_dump(exclude_unset=True)
    return AlarmService().update_rule(db, int(rule_id), patch=patch)


@router.delete("/rules/{rule_id}", response_model=DeleteResponse)
def delete_alarm_rule(rule_id: int, db: Session = Depends(get_db)):
    AlarmService().delete_rule(db, int(rule_id))
    return DeleteResponse(ok=True, message="Alarm rule deleted")


@router.post("/evaluate", response_model=AlarmEvaluateOut)
def evaluate_alarms(payload: AlarmEvaluateRequest | None = Body(default=None), db: Session = Depends(get_db)):
    run_ids = payload.run_ids if payload is not None else []
    out = AlarmService().evaluate_training_rules(db, run_ids=run_ids)
    return AlarmEvaluateOut.model_validate(out)


@router.get("/active", response_model=Page[AlarmAlertOut])
def list_active_alarms(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    severity: str | None = Query(None),
    rule_type: str | None = Query(None),
    source_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    skip = (int(page) - 1) * int(page_size)
    items, total = AlarmService().list_alerts(
        db,
        status=AlarmService.STATUS_ACTIVE,
        severity=severity,
        rule_type=rule_type,
        source_id=source_id,
        skip=skip,
        limit=page_size,
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=total)}


@router.post("/active/{alert_id}/ack", response_model=AlarmAlertOut)
def ack_alarm(alert_id: int, payload: AlarmAckRequest | None = Body(default=None), db: Session = Depends(get_db)):
    acked_by = payload.acked_by if payload is not None else None
    return AlarmService().ack_alert(db, int(alert_id), acked_by=acked_by)


@router.get("/history", response_model=Page[AlarmAlertOut])
def list_alarm_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    severity: str | None = Query(None),
    rule_type: str | None = Query(None),
    source_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    skip = (int(page) - 1) * int(page_size)
    items, total = AlarmService().list_alerts(
        db,
        status=AlarmService.STATUS_RESOLVED,
        severity=severity,
        rule_type=rule_type,
        source_id=source_id,
        skip=skip,
        limit=page_size,
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=total)}


@router.get("/summary", response_model=AlarmSummaryOut)
def get_alarm_summary(db: Session = Depends(get_db)):
    out = AlarmService().get_summary(db)
    return AlarmSummaryOut.model_validate(out)
