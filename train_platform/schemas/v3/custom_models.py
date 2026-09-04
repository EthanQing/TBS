from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class CustomModelPackageOut(BaseModel):
    package_id: int
    name: str
    version: str
    schema_version: int
    sdk_version: str
    entrypoint_module: str
    entrypoint_class: str
    runtime_profile: str
    source_sha256: str
    archive_path: str
    manifest_json: dict[str, Any]
    created_at: datetime
    retired_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CustomModelPackageListResponse(BaseModel):
    items: list[CustomModelPackageOut]
    total: int


__all__ = [
    "CustomModelPackageListResponse",
    "CustomModelPackageOut",
]
