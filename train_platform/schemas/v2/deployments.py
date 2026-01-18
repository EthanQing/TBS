from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import DeploymentPlatform, DeploymentStatus, LogLevel


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

