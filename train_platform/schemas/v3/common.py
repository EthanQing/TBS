from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class PageMeta(BaseModel):
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=500)
    total: int = Field(..., ge=0)


class Page(BaseModel, Generic[T]):
    items: List[T]
    meta: PageMeta


class IdResponse(BaseModel):
    id: str


class DeleteResponse(BaseModel):
    ok: bool = True
    message: Optional[str] = None

