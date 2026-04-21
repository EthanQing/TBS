from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.v3.chart_config import ChartConfig

router = APIRouter(prefix="/chart-configs", tags=["chart-configs"])


@router.get("/{scope}")
def get_chart_config(scope: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the saved chart configuration for the given scope, or empty dict."""
    row = db.query(ChartConfig).filter(ChartConfig.scope == scope).first()
    return row.config if row else {}


@router.put("/{scope}")
def save_chart_config(
    scope: str,
    payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Create or update the chart configuration for the given scope."""
    row = db.query(ChartConfig).filter(ChartConfig.scope == scope).first()
    if row:
        row.config = payload
        row.updated_at = datetime.now(timezone.utc)
    else:
        row = ChartConfig(scope=scope, config=payload)
        db.add(row)
    db.commit()
    return {"ok": True}
