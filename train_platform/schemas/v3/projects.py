from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from train_platform.models.v3.enums import TaskType


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    standard_dataset_id: int
    task_type: TaskType
    description: Optional[str] = None
    created_by: Optional[str] = None
    tags: Optional[dict] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tags: Optional[dict] = None
    is_active: Optional[bool] = None


class ProjectOut(BaseModel):
    project_id: int
    name: str
    description: Optional[str] = None
    standard_dataset_id: int
    task_type: TaskType
    created_by: Optional[str] = None
    tags: Optional[dict] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectModelSizeOut(BaseModel):
    project_id: int
    completed_models_count: int = 0
    total_size_mb: float = 0.0

    model_config = {"from_attributes": True}


class ProjectTrainingAlertRunOut(BaseModel):
    run_id: str
    name: Optional[str] = None
    status: Optional[str] = None
    progress: int = 0
    current_epoch: int = 0
    total_epochs: Optional[int] = None
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ProjectTrainingAlertOut(BaseModel):
    project_id: int
    running_count: int = 0
    latest_running_run: Optional[ProjectTrainingAlertRunOut] = None
    unreviewed_completed_count: int = 0
    latest_unreviewed_completed_run: Optional[ProjectTrainingAlertRunOut] = None


class ProjectCompareBaselineRunOut(BaseModel):
    run_id: str
    name: Optional[str] = None
    status: Optional[str] = None
    architecture_id: Optional[int] = None
    engine: Optional[str] = None


class ProjectCompareBaselineOut(BaseModel):
    project_id: int
    framework_key: str
    baseline_run_id: Optional[str] = None
    baseline_run: Optional[ProjectCompareBaselineRunOut] = None


class ProjectCompareBaselineSetIn(BaseModel):
    framework_key: str = Field(..., min_length=1, max_length=128)
    baseline_run_id: str = Field(..., min_length=1, max_length=36)
