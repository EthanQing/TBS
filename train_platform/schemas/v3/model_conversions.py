from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ModelConversionPerfItem(BaseModel):
    latency_ms: Optional[float] = None
    throughput_img_s: Optional[float] = None
    size_mb: Optional[float] = None


class ModelConversionPerfOut(BaseModel):
    device: Optional[str] = None
    onnx_provider: Optional[str] = None
    imgsz: Optional[int] = None
    pt: Optional[ModelConversionPerfItem] = None
    onnx: Optional[ModelConversionPerfItem] = None


class ModelConversionOut(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    logs: list[str] = []
    output_url: Optional[str] = None
    output_filename: Optional[str] = None
    performance: Optional[ModelConversionPerfOut] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
