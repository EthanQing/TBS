from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run_meta import TrainingRunMeta
from train_platform.repositories.v3.training_run_meta_repo import TrainingRunMetaRepository
from train_platform.utils.exceptions import ValidationError

from .service import TrainingRunService


def _get_run(db: Session, run_id: str):
    return TrainingRunService().get_run(db, run_id)


def get_meta(db: Session, run_id: str) -> TrainingRunMeta:
    _get_run(db, run_id)
    meta = TrainingRunMetaRepository().get_by_run_id(db, run_id)
    if meta:
        return meta

    meta = TrainingRunMeta(run_id=str(run_id))
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return meta


def update_meta(db: Session, run_id: str, *, patch: dict[str, Any]) -> TrainingRunMeta:
    _get_run(db, run_id)
    meta = TrainingRunMetaRepository().get_by_run_id(db, run_id)
    if not meta:
        meta = TrainingRunMeta(run_id=str(run_id))
        db.add(meta)
        db.flush()

    if "creator" in patch:
        meta.creator = patch["creator"]
    if "group" in patch:
        meta.group_name = patch["group"]
    if "tags" in patch:
        meta.tags = patch["tags"]
    if "notes" in patch:
        meta.notes = patch["notes"]
    if "extra" in patch:
        meta.extra = patch["extra"]

    db.commit()
    db.refresh(meta)
    return meta


def mark_project_card_reviewed(db: Session, run_id: str, *, source: str | None = None) -> dict[str, Any]:
    run = _get_run(db, run_id)
    if bool(getattr(run, "hidden", False)) or run.status != TrainingRunStatus.COMPLETED:
        raise ValidationError("Only visible completed training runs can be marked as reviewed")

    repo = TrainingRunMetaRepository()
    meta = repo.get_by_run_id(db, run_id)
    if not meta:
        meta = TrainingRunMeta(run_id=str(run_id))
        db.add(meta)
        db.flush()

    extra = dict(meta.extra) if isinstance(meta.extra, dict) else {}
    reviewed_at = str(extra.get("project_card_reviewed_at") or "").strip()
    if not reviewed_at:
        reviewed_at = datetime.now(timezone.utc).isoformat()
        extra["project_card_reviewed_at"] = reviewed_at
    source_norm = str(source or "").strip()
    if source_norm:
        extra["project_card_review_source"] = source_norm[:64]
    meta.extra = extra

    db.commit()
    db.refresh(meta)

    try:
        reviewed_dt = datetime.fromisoformat(reviewed_at)
    except Exception:
        reviewed_dt = datetime.now(timezone.utc)
    return {
        "run_id": str(run.run_id),
        "reviewed": True,
        "reviewed_at": reviewed_dt,
        "source": extra.get("project_card_review_source"),
    }


__all__ = ["get_meta", "mark_project_card_reviewed", "update_meta"]
