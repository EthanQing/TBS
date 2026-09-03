from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import or_
from sqlalchemy.orm import Session

from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.inference import InferenceRun
from train_platform.models.v3.model_registry import ModelVersion


def delete_model_versions_with_dependents(db: Session, model_versions: Iterable[ModelVersion]) -> None:
    """Stage dependent rows for deletion; the caller owns the transaction commit."""

    versions = list(model_versions)
    model_version_ids = [int(version.model_version_id) for version in versions]
    if not model_version_ids:
        return

    deployments = (
        db.query(Deployment)
        .filter(Deployment.model_version_id.in_(model_version_ids))
        .all()
    )
    deployment_ids = [int(deployment.deployment_id) for deployment in deployments]
    inference_filters = [InferenceRun.model_version_id.in_(model_version_ids)]
    if deployment_ids:
        inference_filters.append(InferenceRun.deployment_id.in_(deployment_ids))

    for inference in db.query(InferenceRun).filter(or_(*inference_filters)).all():
        db.delete(inference)
    db.flush()

    for deployment in deployments:
        db.delete(deployment)
    db.flush()

    for version in versions:
        db.delete(version)
    db.flush()


__all__ = ["delete_model_versions_with_dependents"]
