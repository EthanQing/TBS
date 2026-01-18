from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class InferenceRunCreate(BaseModel):
    model_version_id: int
    deployment_id: Optional[int] = None
    input_path: Optional[str] = Field(None, min_length=1, max_length=500, description="本地路径或 /static/temp/... URL 或 token")
    image_url: Optional[str] = Field(None, description="http(s) 图片 URL")
    conf: float = Field(0.5, gt=0, le=1)
    iou: float = Field(0.45, gt=0, le=1)
    input_meta: Optional[Dict[str, Any]] = None


class InferenceRunOut(BaseModel):
    inference_id: int
    model_version_id: int
    deployment_id: Optional[int] = None
    input_path: str
    input_meta: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class InferenceUploadOut(BaseModel):
    token: str
    path: str
