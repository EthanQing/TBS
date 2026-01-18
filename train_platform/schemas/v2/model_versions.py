from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from train_platform.models.enums import ModelStage


class ModelVersionCreate(BaseModel):
    run_id: str = Field(..., description="来源训练 run_id")
    version: str = Field(..., min_length=1, max_length=50, description="模型版本号（建议语义化，如 v1.0.0）")
    stage: ModelStage = Field(ModelStage.DEVELOPMENT)
    description: Optional[str] = None


class ModelVersionUpdate(BaseModel):
    version: Optional[str] = Field(None, min_length=1, max_length=50)
    stage: Optional[ModelStage] = None
    description: Optional[str] = None


class ModelVersionOut(BaseModel):
    model_version_id: int
    project_id: int
    run_id: str
    version: str
    stage: ModelStage
    description: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    weights_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

