from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import TaskType


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_id: int
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
    dataset_id: int
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

