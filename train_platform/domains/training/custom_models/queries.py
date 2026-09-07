from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.v3.custom_model_package import CustomModelPackage
from train_platform.utils.exceptions import NotFoundError


def get_package(db: Session, package_id: int) -> CustomModelPackage:
    """Fetch a single CustomModelPackage by package_id or raise NotFoundError."""
    pkg = db.query(CustomModelPackage).filter(CustomModelPackage.package_id == int(package_id)).first()
    if not pkg:
        raise NotFoundError(f"CustomModelPackage with id {package_id} not found")
    return pkg


def list_packages(
    db: Session,
    *,
    include_retired: bool = False,
    name: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[CustomModelPackage], int]:
    """List CustomModelPackages with pagination and optional retired filtering."""
    query = db.query(CustomModelPackage)
    if not include_retired:
        query = query.filter(CustomModelPackage.retired_at.is_(None))
    if name:
        query = query.filter(CustomModelPackage.name == str(name).strip())

    total = int(query.count())
    items = (
        query.order_by(CustomModelPackage.created_at.desc())
        .offset(max(0, int(skip)))
        .limit(max(0, int(limit)))
        .all()
    )
    return items, total


__all__ = ["get_package", "list_packages"]
