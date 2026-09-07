from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.domains.projects.baselines import (
    clear_compare_baseline,
    get_compare_baseline,
    set_compare_baseline,
)
from train_platform.domains.projects.deletion import delete_project as delete_project_use_case
from train_platform.domains.projects.service import ProjectService
from train_platform.domains.projects.training_views import (
    get_model_size,
    list_model_sizes,
    list_training_activity,
)
from train_platform.schemas.v3.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v3.projects import (
    ProjectCompareBaselineOut,
    ProjectCompareBaselineSetIn,
    ProjectCreate,
    ProjectModelSizeOut,
    ProjectOut,
    ProjectTrainingAlertOut,
    ProjectUpdate,
)
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/projects", tags=["projects"])


def _parse_project_ids(project_ids: str | None) -> list[int]:
    ids: list[int] = []
    if not project_ids:
        return ids
    seen: set[int] = set()
    for part in str(project_ids).split(","):
        s = part.strip()
        if not s:
            continue
        try:
            pid = int(s)
        except Exception:
            continue
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
    return ids


@router.get("/training-alerts", response_model=list[ProjectTrainingAlertOut])
def list_project_training_alerts(
    project_ids: str | None = Query(
        None,
        description="Comma separated project_id list. If omitted, returns alerts for all projects.",
    ),
    db: Session = Depends(get_db),
):
    return list_training_activity(db, _parse_project_ids(project_ids) if project_ids else None)


@router.get("/model-sizes", response_model=list[ProjectModelSizeOut])
def list_project_model_sizes(
    project_ids: str | None = Query(
        None,
        description="Comma separated project_id list. If omitted, returns stats for all projects.",
    ),
    db: Session = Depends(get_db),
):
    return list_model_sizes(db, _parse_project_ids(project_ids) if project_ids else None)


@router.get("", response_model=Page[ProjectOut])
def list_projects(
    page: int = 1,
    page_size: int = 50,
    standard_dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    service = ProjectService()
    total = service.count_projects(db, standard_dataset_id=standard_dataset_id)
    items = service.list_projects(db, skip=skip, limit=page_size, standard_dataset_id=standard_dataset_id)
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
    return get_compare_baseline(db, int(project_id), str(framework_key))


@router.put("/{project_id}/compare-baseline", response_model=ProjectCompareBaselineOut)
def set_project_compare_baseline(project_id: int, payload: ProjectCompareBaselineSetIn, db: Session = Depends(get_db)):
    run_id = str(payload.baseline_run_id or "").strip()
    if not run_id:
        raise ValidationError("baseline_run_id is required")
    return set_compare_baseline(db, int(project_id), str(payload.framework_key), run_id)


@router.delete("/{project_id}/compare-baseline", response_model=ProjectCompareBaselineOut)
def clear_project_compare_baseline(
    project_id: int,
    framework_key: str = Query(..., description="pytorch|paddle|engine:<name>"),
    db: Session = Depends(get_db),
):
    return clear_compare_baseline(db, int(project_id), str(framework_key))


@router.get("/{project_id}/model-size", response_model=ProjectModelSizeOut)
def get_project_model_size(project_id: int, db: Session = Depends(get_db)):
    return get_model_size(db, int(project_id))


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)):
    return ProjectService().update_project(db, project_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{project_id}", response_model=DeleteResponse)
def delete_project(
    project_id: int,
    force: bool = Query(False, description="Delete project and all related training runs/model versions"),
    db: Session = Depends(get_db),
):
    delete_project_use_case(db, project_id, force=bool(force))
    return DeleteResponse(ok=True, message="Project deleted")
