from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.enums import TrainingRunStatus
from train_platform.models.project import Project
from train_platform.models.training_run import TrainingRun, TrainingRunResult
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.projects import (
    ProjectCompareBaselineOut,
    ProjectCompareBaselineSetIn,
    ProjectCreate,
    ProjectModelSizeOut,
    ProjectOut,
    ProjectUpdate,
)
from train_platform.utils.exceptions import ValidationError
from train_platform.services.project_service import ProjectService


router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/model-sizes", response_model=list[ProjectModelSizeOut])
def list_project_model_sizes(
    project_ids: str | None = Query(
        None,
        description="Comma separated project_id list. If omitted, returns stats for all projects.",
    ),
    db: Session = Depends(get_db),
):
    ids: list[int] = []
    if project_ids:
        for part in str(project_ids).split(","):
            s = part.strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except Exception:
                continue
        # de-dup while keeping order
        seen = set()
        ids = [x for x in ids if not (x in seen or seen.add(x))]
    else:
        ids = [int(x[0]) for x in db.query(Project.project_id).order_by(Project.project_id.asc()).all()]

    if not ids:
        return []

    rows = (
        db.query(
            TrainingRun.project_id.label("project_id"),
            func.count(TrainingRun.run_id).label("completed_models_count"),
            func.coalesce(func.sum(TrainingRunResult.model_size_mb), 0).label("total_size_mb"),
        )
        .select_from(TrainingRun)
        .outerjoin(TrainingRunResult, TrainingRunResult.run_id == TrainingRun.run_id)
        .filter(TrainingRun.project_id.in_(ids))
        .filter(TrainingRun.hidden == False)  # noqa: E712
        .filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
        .group_by(TrainingRun.project_id)
        .all()
    )

    by_id: dict[int, tuple[int, float]] = {}
    for r in rows:
        pid = int(getattr(r, "project_id", 0) or 0)
        cnt = int(getattr(r, "completed_models_count", 0) or 0)
        total = getattr(r, "total_size_mb", 0) or 0
        try:
            total_f = float(total)
        except Exception:
            total_f = 0.0
        by_id[pid] = (cnt, float(round(total_f, 2)))

    out: list[ProjectModelSizeOut] = []
    for pid in ids:
        cnt, total = by_id.get(int(pid), (0, 0.0))
        out.append(ProjectModelSizeOut(project_id=int(pid), completed_models_count=int(cnt), total_size_mb=float(total)))
    return out


@router.get("", response_model=Page[ProjectOut])
def list_projects(
    page: int = 1,
    page_size: int = 50,
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    q = db.query(Project)
    if dataset_id is not None:
        q = q.filter(Project.dataset_id == int(dataset_id))
    total = q.count()
    items = ProjectService().list_projects(db, skip=skip, limit=page_size, dataset_id=dataset_id)
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    return ProjectService().create_project(db, obj=payload.model_dump())


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    return ProjectService().get_project(db, project_id)


@router.get("/{project_id}/compare-baseline", response_model=ProjectCompareBaselineOut)
def get_project_compare_baseline(
    project_id: int,
    framework_key: str = Query(..., description="pytorch|paddle|engine:<name>"),
    db: Session = Depends(get_db),
):
    return ProjectService().get_compare_baseline(db, int(project_id), str(framework_key))


@router.put("/{project_id}/compare-baseline", response_model=ProjectCompareBaselineOut)
def set_project_compare_baseline(project_id: int, payload: ProjectCompareBaselineSetIn, db: Session = Depends(get_db)):
    run_id = str(payload.baseline_run_id or "").strip()
    if not run_id:
        raise ValidationError("baseline_run_id is required")
    return ProjectService().set_compare_baseline(
        db,
        int(project_id),
        str(payload.framework_key),
        run_id,
    )


@router.delete("/{project_id}/compare-baseline", response_model=ProjectCompareBaselineOut)
def clear_project_compare_baseline(
    project_id: int,
    framework_key: str = Query(..., description="pytorch|paddle|engine:<name>"),
    db: Session = Depends(get_db),
):
    return ProjectService().clear_compare_baseline(db, int(project_id), str(framework_key))


@router.get("/{project_id}/model-size", response_model=ProjectModelSizeOut)
def get_project_model_size(project_id: int, db: Session = Depends(get_db)):
    # 404 if project not found
    ProjectService().get_project(db, int(project_id))

    row = (
        db.query(
            func.count(TrainingRun.run_id).label("completed_models_count"),
            func.coalesce(func.sum(TrainingRunResult.model_size_mb), 0).label("total_size_mb"),
        )
        .select_from(TrainingRun)
        .outerjoin(TrainingRunResult, TrainingRunResult.run_id == TrainingRun.run_id)
        .filter(TrainingRun.project_id == int(project_id))
        .filter(TrainingRun.hidden == False)  # noqa: E712
        .filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
        .first()
    )

    cnt = int(getattr(row, "completed_models_count", 0) or 0) if row is not None else 0
    total = getattr(row, "total_size_mb", 0) if row is not None else 0
    try:
        total_f = float(total or 0)
    except Exception:
        total_f = 0.0
    return ProjectModelSizeOut(project_id=int(project_id), completed_models_count=cnt, total_size_mb=float(round(total_f, 2)))


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)):
    return ProjectService().update_project(db, project_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{project_id}", response_model=DeleteResponse)
def delete_project(
    project_id: int,
    force: bool = Query(False, description="Delete project and all related training runs/model versions"),
    db: Session = Depends(get_db),
):
    ProjectService().delete_project(db, project_id, force=bool(force))
    return DeleteResponse(ok=True, message="Project deleted")

