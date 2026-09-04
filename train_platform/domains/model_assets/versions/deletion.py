from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import DeploymentRunStatus
from train_platform.models.v3.inference import InferenceRun
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.project import Project
from train_platform.utils.exceptions import ConflictError


def delete_model_versions_with_dependents(db: Session, model_versions: Iterable[ModelVersion]) -> None:
    """Stage dependent rows for deletion; the caller owns the transaction commit."""

    versions = list(model_versions)
    model_version_ids = [
        int(version.model_version_id)
        for version in versions
        if getattr(version, "model_version_id", None) is not None
    ]
    if not model_version_ids:
        return

    # Collect affected project IDs
    affected_project_ids = {
        int(v.project_id)
        for v in versions
        if getattr(v, "project_id", None) is not None
    }
    db_project_ids = (
        db.query(ModelVersion.project_id)
        .filter(ModelVersion.model_version_id.in_(model_version_ids))
        .all()
    )
    for row in db_project_ids:
        if row and row[0] is not None:
            affected_project_ids.add(int(row[0]))

    # Lock affected Project rows in stable sorted order to serialize against concurrent admission
    for project_id in sorted(affected_project_ids):
        db.query(Project).filter(Project.project_id == project_id).populate_existing().with_for_update().first()

    # Re-query affected deployments under lock
    deployments = (
        db.query(Deployment)
        .filter(Deployment.model_version_id.in_(model_version_ids))
        .populate_existing()
        .with_for_update()
        .all()
    )
    affected_deployment_ids = [int(deployment.deployment_id) for deployment in deployments]

    # Active run must cover deleted model version IDs OR affected deployment IDs
    active_run_filters = [DeploymentRun.model_version_id.in_(model_version_ids)]
    if affected_deployment_ids:
        active_run_filters.append(DeploymentRun.deployment_id.in_(affected_deployment_ids))

    active_run = (
        db.query(DeploymentRun)
        .filter(
            or_(*active_run_filters),
            DeploymentRun.status.in_((DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING)),
        )
        .first()
    )
    if active_run is not None:
        raise ConflictError(
            "Cannot delete model versions while an active deployment run exists; "
            "finish or cancel the deployment run first"
        )

    deployment_ids = affected_deployment_ids
    inference_filters = [InferenceRun.model_version_id.in_(model_version_ids)]
    if deployment_ids:
        inference_filters.append(InferenceRun.deployment_id.in_(deployment_ids))

    for inference in db.query(InferenceRun).filter(or_(*inference_filters)).all():
        db.delete(inference)
    db.flush()

    for deployment in deployments:
        db.delete(deployment)
    db.flush()

    target_versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_version_id.in_(model_version_ids))
        .populate_existing()
        .with_for_update()
        .all()
    )
    for version in target_versions:
        db.delete(version)
    db.flush()


__all__ = ["delete_model_versions_with_dependents"]

