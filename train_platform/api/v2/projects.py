from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.models.project import Project
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.projects import ProjectCreate, ProjectOut, ProjectUpdate
from train_platform.services.project_service import ProjectService


router = APIRouter(prefix="/projects", tags=["projects"])


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


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)):
    return ProjectService().update_project(db, project_id, patch=payload.model_dump(exclude_unset=True))


@router.delete("/{project_id}", response_model=DeleteResponse)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    ProjectService().delete_project(db, project_id)
    return DeleteResponse(ok=True, message="Project deleted")

