from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import TaskType


class ArchitectureCreate(BaseModel):
    family: str = Field(..., min_length=1, max_length=50)
    variant: str = Field(..., min_length=1, max_length=100)
    task_type: TaskType
    engine: str = Field("ultralytics-yolo", min_length=1, max_length=64)
    pretrained_path: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    default_params: Optional[dict] = None


class ArchitectureOut(BaseModel):
    architecture_id: int
    family: str
    variant: str
    task_type: TaskType
    engine: str
    pretrained_path: Optional[str] = None
    description: Optional[str] = None
    default_params: Optional[dict] = None
    created_at: datetime

    model_config = {"from_attributes": True}

