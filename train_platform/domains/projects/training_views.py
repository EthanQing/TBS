from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.project import Project
from train_platform.models.v3.training_run import TrainingRun, TrainingRunResult
from train_platform.models.v3.training_run_meta import TrainingRunMeta

from .service import ProjectService


def _normalize_project_ids(db: Session, project_ids: list[int] | None) -> list[int]:
    if project_ids is None:
        return [int(row[0]) for row in db.query(Project.project_id).order_by(Project.project_id.asc()).all()]

    ids: list[int] = []
    seen: set[int] = set()
    for raw in project_ids:
        try:
            project_id = int(raw)
        except Exception:
            continue
        if project_id <= 0 or project_id in seen:
            continue
        seen.add(project_id)
        ids.append(project_id)
    return ids


def _reviewed_at_from_extra(extra: Any) -> str | None:
    data = extra if isinstance(extra, dict) else {}
    value = data.get("project_card_reviewed_at")
    return str(value).strip() if value else None


def _run_payload(run: TrainingRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": str(run.run_id),
        "name": run.name,
        "status": str(getattr(run.status, "value", run.status) or ""),
        "progress": int(getattr(run, "progress", 0) or 0),
        "current_epoch": int(getattr(run, "current_epoch", 0) or 0),
        "total_epochs": getattr(run, "total_epochs", None),
        "updated_at": getattr(run, "updated_at", None),
        "finished_at": getattr(run, "finished_at", None),
    }


def _is_newer(sort_key: object, current_key: object) -> bool:
    if current_key is None:
        return True
    if sort_key is None:
        return False
    return sort_key > current_key


def list_training_activity(db: Session, project_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Return project-facing running and unreviewed-completed training activity."""

    ids = _normalize_project_ids(db, project_ids)
    if not ids:
        return []

    rows = (
        db.query(TrainingRun, TrainingRunMeta)
        .outerjoin(TrainingRunMeta, TrainingRunMeta.run_id == TrainingRun.run_id)
        .filter(TrainingRun.project_id.in_(ids))
        .filter(TrainingRun.hidden == False)  # noqa: E712
        .filter(TrainingRun.status.in_([TrainingRunStatus.RUNNING, TrainingRunStatus.COMPLETED]))
        .all()
    )

    by_project = {
        project_id: {
            "project_id": project_id,
            "running_count": 0,
            "latest_running_run": None,
            "unreviewed_completed_count": 0,
            "latest_unreviewed_completed_run": None,
        }
        for project_id in ids
    }
    latest_running_sort: dict[int, object] = {}
    latest_completed_sort: dict[int, object] = {}

    for run, meta in rows:
        project_id = int(getattr(run, "project_id", 0) or 0)
        bucket = by_project.get(project_id)
        if bucket is None:
            continue

        if run.status == TrainingRunStatus.RUNNING:
            bucket["running_count"] = int(bucket["running_count"]) + 1
            sort_key = getattr(run, "updated_at", None) or getattr(run, "started_at", None) or getattr(run, "created_at", None)
            if bucket["latest_running_run"] is None or _is_newer(sort_key, latest_running_sort.get(project_id)):
                latest_running_sort[project_id] = sort_key
                bucket["latest_running_run"] = _run_payload(run)
            continue

        if _reviewed_at_from_extra(getattr(meta, "extra", None) if meta is not None else None):
            continue
        bucket["unreviewed_completed_count"] = int(bucket["unreviewed_completed_count"]) + 1
        sort_key = getattr(run, "finished_at", None) or getattr(run, "updated_at", None) or getattr(run, "created_at", None)
        if bucket["latest_unreviewed_completed_run"] is None or _is_newer(sort_key, latest_completed_sort.get(project_id)):
            latest_completed_sort[project_id] = sort_key
            bucket["latest_unreviewed_completed_run"] = _run_payload(run)

    return [by_project[project_id] for project_id in ids]


def _aggregate_model_sizes(db: Session, project_ids: list[int]) -> list[dict[str, Any]]:
    if not project_ids:
        return []

    rows = (
        db.query(
            TrainingRun.project_id.label("project_id"),
            func.count(TrainingRun.run_id).label("completed_models_count"),
            func.coalesce(func.sum(TrainingRunResult.model_size_mb), 0).label("total_size_mb"),
        )
        .select_from(TrainingRun)
        .outerjoin(TrainingRunResult, TrainingRunResult.run_id == TrainingRun.run_id)
        .filter(TrainingRun.project_id.in_(project_ids))
        .filter(TrainingRun.hidden == False)  # noqa: E712
        .filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
        .group_by(TrainingRun.project_id)
        .all()
    )

    by_id: dict[int, tuple[int, float]] = {}
    for row in rows:
        project_id = int(getattr(row, "project_id", 0) or 0)
        count = int(getattr(row, "completed_models_count", 0) or 0)
        total = getattr(row, "total_size_mb", 0) or 0
        try:
            total_size = float(total)
        except Exception:
            total_size = 0.0
        by_id[project_id] = (count, float(round(total_size, 2)))

    return [
        {
            "project_id": project_id,
            "completed_models_count": by_id.get(project_id, (0, 0.0))[0],
            "total_size_mb": by_id.get(project_id, (0, 0.0))[1],
        }
        for project_id in project_ids
    ]


def list_model_sizes(db: Session, project_ids: list[int] | None = None) -> list[dict[str, Any]]:
    return _aggregate_model_sizes(db, _normalize_project_ids(db, project_ids))


def get_model_size(db: Session, project_id: int) -> dict[str, Any]:
    project_id = int(project_id)
    ProjectService().get_project(db, project_id)
    result = _aggregate_model_sizes(db, [project_id])
    return result[0]


__all__ = ["get_model_size", "list_model_sizes", "list_training_activity"]
