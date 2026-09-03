from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.model_assets.versions.deletion import delete_model_versions_with_dependents
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.platform.filesystem import remove_tree
from train_platform.utils.exceptions import ConflictError

from .service import ProjectService


def delete_project(db: Session, project_id: int, *, force: bool = False) -> None:
    project = ProjectService().get_project(db, int(project_id))
    runs = db.query(TrainingRun).filter(TrainingRun.project_id == int(project.project_id)).all()
    model_versions = db.query(ModelVersion).filter(ModelVersion.project_id == int(project.project_id)).all()

    if not force:
        ProjectService.validate_delete(runs=runs, model_versions=model_versions)

    if force:
        running_runs = [run for run in runs if run.status == TrainingRunStatus.RUNNING]
        if running_runs:
            raise ConflictError("Cannot force delete project; active training runs must finish or cancel before deletion")

    run_dirs = [settings.training_dir / str(run.run_id) for run in runs]
    delete_model_versions_with_dependents(db, model_versions)
    for run in runs:
        db.delete(run)
    db.flush()
    db.delete(project)
    db.commit()

    for run_dir in run_dirs:
        remove_tree(run_dir, ignore_errors=True)


__all__ = ["delete_project"]
