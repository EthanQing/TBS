from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DatasetConversionOut(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    stage: Optional[str] = None
    processed: Optional[int] = None
    total: Optional[int] = None
    logs: list[str] = []
    output_url: Optional[str] = None
    output_filename: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

