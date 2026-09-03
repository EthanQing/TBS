from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.enums import DeploymentStatus, ModelStage
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.project import Project
from train_platform.utils.exceptions import NotFoundError, ValidationError


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ActivationResult:
    deployment: Deployment
    previous_model_version_id: int


def _deployment(db: Session, deployment_id: int, *, lock: bool = True) -> Deployment:
    query = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).populate_existing()
    if lock:
        query = query.with_for_update()
    row = query.first()
    if not row:
        raise NotFoundError("Deployment not found")
    return row


def _model_version(db: Session, model_version_id: int, *, lock: bool = True) -> ModelVersion:
    query = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).populate_existing()
    if lock:
        query = query.with_for_update()
    row = query.first()
    if not row:
        raise NotFoundError("Model version not found")
    return row


def _deactivate_peers(db: Session, *, project_id: int, keep_deployment_id: int) -> None:
    peers = (
        db.query(Deployment)
        .join(ModelVersion, ModelVersion.model_version_id == Deployment.model_version_id)
        .filter(
            ModelVersion.project_id == int(project_id),
            Deployment.deployment_id != int(keep_deployment_id),
            or_(Deployment.is_active.is_(True), Deployment.status == DeploymentStatus.ACTIVE),
        )
        .populate_existing()
        .with_for_update()
        .all()
    )
    for peer in peers:
        peer.is_active = False
        if peer.status == DeploymentStatus.ACTIVE:
            peer.status = DeploymentStatus.INACTIVE


def activate_deployment(
    db: Session,
    *,
    deployment_id: int,
    model_version_id: int | None = None,
) -> ActivationResult:
    """Activate a deployment and synchronize the project's model stage.

    The project row is locked before peer lookup so concurrent activation and
    deployment-run admission serialize on the same project relationship.
    """
    initial_deployment = (
        db.query(Deployment)
        .filter(Deployment.deployment_id == int(deployment_id))
        .first()
    )
    if not initial_deployment:
        raise NotFoundError("Deployment not found")
    initial_current = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_version_id == int(initial_deployment.model_version_id))
        .first()
    )
    if not initial_current:
        raise NotFoundError("Model version not found")
    target_id = int(model_version_id if model_version_id is not None else initial_deployment.model_version_id)
    initial_target = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_version_id == target_id)
        .first()
    )
    if not initial_target:
        raise NotFoundError("Model version not found")
    if int(initial_target.project_id) != int(initial_current.project_id):
        raise ValidationError("Model version does not belong to the deployment project")

    project = (
        db.query(Project)
        .filter(Project.project_id == int(initial_current.project_id))
        .populate_existing()
        .with_for_update()
        .first()
    )
    if not project:
        raise NotFoundError("Project not found")
    deployment = _deployment(db, deployment_id)
    current = _model_version(db, int(deployment.model_version_id))
    previous_model_version_id = int(deployment.model_version_id)
    target_id = int(model_version_id if model_version_id is not None else deployment.model_version_id)
    target = _model_version(db, target_id)
    if int(current.project_id) != int(project.project_id) or int(target.project_id) != int(project.project_id):
        raise ValidationError("Model version does not belong to the deployment project")

    _deactivate_peers(db, project_id=int(project.project_id), keep_deployment_id=int(deployment.deployment_id))
    deployment.model_version_id = target_id
    deployment.is_active = True
    deployment.status = DeploymentStatus.ACTIVE
    deployment.deployed_at = utcnow()

    db.query(ModelVersion).filter(
        ModelVersion.project_id == int(project.project_id),
        ModelVersion.model_version_id != target_id,
        ModelVersion.stage == ModelStage.PRODUCTION,
    ).update({ModelVersion.stage: ModelStage.TESTING}, synchronize_session=False)
    target.stage = ModelStage.PRODUCTION
    return ActivationResult(
        deployment=deployment,
        previous_model_version_id=previous_model_version_id,
    )


def prepare_deployment_for_run(db: Session, *, deployment_id: int) -> Deployment:
    """Mark a non-active deployment as deploying while preserving live serving."""
    deployment = _deployment(db, deployment_id)
    if not (deployment.is_active and deployment.status == DeploymentStatus.ACTIVE):
        deployment.status = DeploymentStatus.DEPLOYING
    return deployment


def mark_deployment_failed(db: Session, *, deployment_id: int) -> Deployment:
    deployment = _deployment(db, deployment_id)
    if deployment.status == DeploymentStatus.DEPLOYING and not deployment.is_active:
        deployment.status = DeploymentStatus.FAILED
    return deployment


def mark_deployment_cancelled(db: Session, *, deployment_id: int) -> Deployment:
    deployment = _deployment(db, deployment_id)
    if deployment.status == DeploymentStatus.DEPLOYING and not deployment.is_active:
        deployment.status = DeploymentStatus.PENDING
    return deployment


def patch_lifecycle(
    db: Session,
    *,
    deployment_id: int,
    status: DeploymentStatus | None = None,
    is_active: bool | None = None,
    status_provided: bool = False,
    is_active_provided: bool = False,
) -> Deployment:
    """Apply the public lifecycle patch without bypassing activation rules."""
    deployment = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
    if not deployment:
        raise NotFoundError("Deployment not found")
    if status_provided and status == DeploymentStatus.ACTIVE and is_active_provided and is_active is False:
        raise ValidationError("ACTIVE deployment must be active")
    if status_provided and status != DeploymentStatus.ACTIVE and is_active_provided and is_active is True:
        raise ValidationError("Only ACTIVE deployment can be active")

    if status_provided and status == DeploymentStatus.ACTIVE:
        return activate_deployment(db, deployment_id=int(deployment.deployment_id)).deployment
    if is_active_provided and is_active is True:
        return activate_deployment(db, deployment_id=int(deployment.deployment_id)).deployment

    deployment = _deployment(db, deployment_id)
    if is_active_provided and is_active is False:
        deployment.is_active = False
        if deployment.status == DeploymentStatus.ACTIVE:
            deployment.status = DeploymentStatus.INACTIVE

    if status_provided and status is not None:
        deployment.status = status
        deployment.is_active = False
    return deployment


__all__ = [
    "activate_deployment",
    "ActivationResult",
    "mark_deployment_cancelled",
    "mark_deployment_failed",
    "patch_lifecycle",
    "prepare_deployment_for_run",
    "utcnow",
]
