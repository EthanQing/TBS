from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(prefix="/datasets", tags=["datasets"])

# V3 no longer exposes the mixed `/datasets` resource.
# Use `/illegal-datasets` and `/standard-datasets` instead.
