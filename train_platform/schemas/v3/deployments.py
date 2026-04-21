from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from train_platform.models.v3.enums import (
    DeploymentPlatform,
    DeploymentRunPhase,
    DeploymentRunStatus,
    DeploymentStatus,
    DeploymentTriggerType,
    LogLevel,
    ModelStage,
)


class DeploymentCreate(BaseModel):
    model_version_id: int
    name: str = Field(..., min_length=1, max_length=255)
    platform: DeploymentPlatform
    config: Optional[Dict[str, Any]] = None
    health_check_url: Optional[str] = Field(None, max_length=500)


class DeploymentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[DeploymentStatus] = None
    endpoint_url: Optional[str] = Field(None, max_length=500)
    health_check_url: Optional[str] = Field(None, max_length=500)
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class DeploymentOut(BaseModel):
    deployment_id: int
    model_version_id: int
    name: str
    platform: DeploymentPlatform
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    api_key_hint: Optional[str] = None
    is_active: bool
    deployed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DeploymentLogCreate(BaseModel):
    level: LogLevel = LogLevel.INFO
    message: str
    data: Optional[Dict[str, Any]] = None


class DeploymentLogOut(BaseModel):
    log_id: int
    deployment_id: int
    level: LogLevel
    message: str
    data: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DeploymentExecuteCreate(BaseModel):
    operator: str = Field("admin", min_length=1, max_length=128)
    reason: Optional[str] = Field(None, max_length=1000)
    rotate_api_key: bool = True
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)


class DeploymentRunOut(BaseModel):
    run_id: str
    deployment_id: int
    project_id: int
    model_version_id: int
    trigger_type: DeploymentTriggerType
    status: DeploymentRunStatus
    phase: DeploymentRunPhase
    current_step: Optional[str] = None
    progress: int
    cancel_requested: bool
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DeploymentExecuteOut(BaseModel):
    run: DeploymentRunOut
    issued_api_key: Optional[str] = None
    api_key_hint: Optional[str] = None


class DeploymentRunRetryOut(BaseModel):
    run: DeploymentRunOut
    issued_api_key: Optional[str] = None
    api_key_hint: Optional[str] = None


class DeploymentRunWsMessage(BaseModel):
    type: str
    data: Dict[str, Any]


class ServingInferRequest(BaseModel):
    image_url: Optional[str] = None
    conf: float = Field(0.5, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    show_labels: bool = True
    show_confidence: bool = True


class ServingInferResponse(BaseModel):
    deployment_id: int
    model_version_id: int
    engine: Optional[str] = None
    family: Optional[str] = None
    variant: Optional[str] = None
    run_id: Optional[str] = None
    input_path: Optional[str] = None
    input_meta: Optional[Dict[str, Any]] = None
    inference_time_ms: Optional[float] = None
    names: Optional[Dict[str, Any]] = None
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    raw_output: Optional[Dict[str, Any]] = None


class ServingJobCreate(BaseModel):
    mode: str = Field(..., pattern="^(batch|video)$")
    deployment_id: int


class ServingJobOut(BaseModel):
    enabled: bool = False
    status: str
    message: str
    mode: Optional[str] = None
    job_id: Optional[str] = None


class DeploymentRollbackCreate(BaseModel):
    target_model_version_id: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1, max_length=1000)
    operator: str = Field("admin", min_length=1, max_length=128)


class DeploymentRollbackCandidateOut(BaseModel):
    model_version_id: int
    run_id: str
    version: str
    stage: ModelStage
    weights_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DeploymentRollbackCandidatesOut(BaseModel):
    deployment: DeploymentOut
    current_model_version_id: int
    candidates: List[DeploymentRollbackCandidateOut]


class DeploymentRollbackHistoryOut(BaseModel):
    log_id: int
    deployment_id: int
    created_at: datetime
    operator: Optional[str] = None
    reason: Optional[str] = None
    from_model_version_id: Optional[int] = None
    to_model_version_id: Optional[int] = None
    from_version: Optional[str] = None
    to_version: Optional[str] = None


class DeploymentRollbackOut(BaseModel):
    deployment: DeploymentOut
    event: DeploymentRollbackHistoryOut

