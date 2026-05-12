from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AlarmSeverity = Literal["critical", "high", "medium", "low"]
AlarmStatus = Literal["active", "resolved"]
AlarmRuleType = Literal["training_run_failed", "training_run_stale"]
AlarmSourceType = Literal["training_run"]


class AlarmRuleTypeOut(BaseModel):
    rule_type: AlarmRuleType
    name: str
    description: str
    default_severity: AlarmSeverity
    default_enabled: bool
    default_cooldown_seconds: int
    config_schema: Dict[str, Any] = Field(default_factory=dict)


class AlarmRuleBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = Field(None, max_length=2000)
    severity: AlarmSeverity = "high"
    enabled: bool = True
    cooldown_seconds: int = Field(300, ge=0, le=86400)
    config: Dict[str, Any] = Field(default_factory=dict)


class AlarmRuleCreate(AlarmRuleBase):
    rule_type: AlarmRuleType


class AlarmRuleUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    description: Optional[str] = Field(None, max_length=2000)
    severity: Optional[AlarmSeverity] = None
    enabled: Optional[bool] = None
    cooldown_seconds: Optional[int] = Field(None, ge=0, le=86400)
    config: Optional[Dict[str, Any]] = None


class AlarmRuleOut(BaseModel):
    rule_id: int
    rule_type: AlarmRuleType
    name: str
    description: Optional[str] = None
    severity: AlarmSeverity
    enabled: bool
    cooldown_seconds: int
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AlarmAlertOut(BaseModel):
    alert_id: int
    rule_id: Optional[int] = None
    rule_type: AlarmRuleType
    severity: AlarmSeverity
    status: AlarmStatus
    title: str
    message: Optional[str] = None
    source_type: AlarmSourceType
    source_id: str
    trigger_count: int
    first_triggered_at: datetime
    last_triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
    acked_by: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AlarmEvaluateRequest(BaseModel):
    run_ids: List[str] = Field(default_factory=list)


class AlarmEvaluateOut(BaseModel):
    evaluated_runs: int
    triggered_new: int
    touched_active: int
    resolved: int
    active_total: int
    timestamp: datetime


class AlarmAckRequest(BaseModel):
    acked_by: Optional[str] = Field(None, max_length=128)


class AlarmSummaryOut(BaseModel):
    active_total: int
    by_severity: Dict[str, int] = Field(default_factory=dict)
