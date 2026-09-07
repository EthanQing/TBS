from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.project import Project
from train_platform.models.v3.training_run import TrainingRun
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

BASELINE_TAG = "compare_baseline"


def _get_project(db: Session, project_id: int) -> Project:
    project = db.query(Project).filter(Project.project_id == int(project_id)).first()
    if not project:
        raise NotFoundError("Project not found")
    return project


def normalize_framework_key(framework_key: str) -> str:
    raw = str(framework_key or "").strip().lower()
    if not raw:
        raise ValidationError("framework_key is required")
    if raw in ("pytorch", "paddle"):
        return raw
    if raw.startswith("engine:") and len(raw) > len("engine:"):
        return raw
    raise ValidationError("framework_key must be pytorch, paddle, or engine:<name>")


def extract_baselines(tags: Any) -> dict[str, str]:
    data = tags if isinstance(tags, dict) else {}
    bucket = data.get(BASELINE_TAG)
    if not isinstance(bucket, dict):
        return {}
    result: dict[str, str] = {}
    for framework_key, run_id in bucket.items():
        key = str(framework_key or "").strip().lower()
        normalized_run_id = str(run_id or "").strip()
        if key and normalized_run_id:
            result[key] = normalized_run_id
    return result


def preserve_baselines(existing_tags: Any, requested_tags: Any) -> dict[str, Any] | None:
    """Apply ordinary tags input while keeping compare_baseline system-owned."""

    existing = existing_tags if isinstance(existing_tags, dict) else {}
    result = dict(requested_tags) if isinstance(requested_tags, dict) else None
    if BASELINE_TAG in existing:
        if result is None:
            result = {}
        result[BASELINE_TAG] = existing[BASELINE_TAG]
    elif result is not None:
        result.pop(BASELINE_TAG, None)
    return result


def _resolve_framework_from_engine(engine: str | None) -> str:
    raw = str(engine or "").strip().lower()
    if not raw:
        return "engine:unknown"
    if raw == "ultralytics-yolo":
        return "pytorch"
    if raw == "paddle-det":
        return "paddle"
    return f"engine:{raw}"


def _run_payload(run: TrainingRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": str(run.run_id),
        "name": run.name,
        "status": str(getattr(run.status, "value", run.status) or ""),
        "architecture_id": int(run.architecture_id),
        "engine": str(getattr(run.architecture, "engine", "") or "").strip().lower() or None,
    }


def _build_baseline_response(db: Session, project: Project, framework_key: str) -> dict[str, Any]:
    baseline_map = extract_baselines(project.tags)
    run_id = baseline_map.get(framework_key)
    baseline_run = None
    if run_id:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
        if run is not None and int(run.project_id) == int(project.project_id):
            baseline_run = _run_payload(run)
    return {
        "project_id": int(project.project_id),
        "framework_key": framework_key,
        "baseline_run_id": run_id,
        "baseline_run": baseline_run,
    }


def get_compare_baseline(db: Session, project_id: int, framework_key: str) -> dict[str, Any]:
    project = _get_project(db, int(project_id))
    return _build_baseline_response(db, project, normalize_framework_key(framework_key))


def set_compare_baseline(db: Session, project_id: int, framework_key: str, baseline_run_id: str) -> dict[str, Any]:
    project = _get_project(db, int(project_id))
    key = normalize_framework_key(framework_key)
    run = db.query(TrainingRun).filter(TrainingRun.run_id == str(baseline_run_id).strip()).first()
    if run is None:
        raise NotFoundError("Training run not found")
    if int(run.project_id) != int(project.project_id):
        raise ConflictError("Baseline run does not belong to this project")
    architecture = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()
    if architecture is None:
        raise NotFoundError("Architecture not found")
    if _resolve_framework_from_engine(getattr(architecture, "engine", None)) != key:
        raise ConflictError("Baseline run framework does not match framework_key")

    tags = dict(project.tags) if isinstance(project.tags, dict) else {}
    baseline_map = extract_baselines(tags)
    baseline_map[key] = str(run.run_id)
    tags[BASELINE_TAG] = baseline_map
    project.tags = tags
    db.commit()
    db.refresh(project)
    return _build_baseline_response(db, project, key)


def clear_compare_baseline(db: Session, project_id: int, framework_key: str) -> dict[str, Any]:
    project = _get_project(db, int(project_id))
    key = normalize_framework_key(framework_key)
    tags = dict(project.tags) if isinstance(project.tags, dict) else {}
    baseline_map = extract_baselines(tags)
    if key in baseline_map:
        baseline_map.pop(key, None)
        if baseline_map:
            tags[BASELINE_TAG] = baseline_map
        else:
            tags.pop(BASELINE_TAG, None)
        project.tags = tags
        db.commit()
        db.refresh(project)
    return {
        "project_id": int(project.project_id),
        "framework_key": key,
        "baseline_run_id": None,
        "baseline_run": None,
    }


__all__ = [
    "BASELINE_TAG",
    "clear_compare_baseline",
    "extract_baselines",
    "get_compare_baseline",
    "normalize_framework_key",
    "preserve_baselines",
    "set_compare_baseline",
]
