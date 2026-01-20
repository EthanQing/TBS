from __future__ import annotations

from pydantic import BaseModel


class PretrainUploadOut(BaseModel):
    token: str
    path: str
    filename: str
