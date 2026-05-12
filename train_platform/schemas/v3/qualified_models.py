from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class QualifiedModelCreate(BaseModel):
    model_version_id: int = Field(..., ge=1, description="要标记为合格的模型版本 ID")
    qualified_by: Optional[str] = Field(None, max_length=128, description="标记人/操作人")
    note: Optional[str] = Field(None, max_length=2000, description="主观判断备注")


class QualifiedModelOut(BaseModel):
    qualified_model_id: int
    model_version_id: int
    project_id: int
    run_id: str
    standard_dataset_id: int
    qualified_by: Optional[str] = None
    note: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    weights_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QualifiedModelMarkResponse(BaseModel):
    created: bool = Field(..., description="本次请求是否新建了合格模型记录")
    message: str
    item: QualifiedModelOut
