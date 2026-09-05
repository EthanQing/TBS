from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.custom_model_package import CustomModelPackage
from train_platform.models.v3.enums import TaskType
from train_platform.utils.exceptions import ConflictError, ValidationError

from .registry import get_plugin


def _normalize_and_validate_engine(value: str | None) -> str:
    engine = str(value or "").strip().lower() or "ultralytics-yolo"
    try:
        plugin = get_plugin(engine)
    except Exception as exc:
        raise ValidationError(f"Unknown architecture engine: {engine}") from exc
    return str(getattr(plugin, "plugin_id", engine) or engine).strip().lower()


def list_architectures(
    db: Session,
    *,
    family: str | None = None,
    task_type: TaskType | None = None,
    engine: str | None = None,
    custom_model_package_id: int | None = None,
    q: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[ModelArchitecture]:
    query = db.query(ModelArchitecture)
    if family:
        query = query.filter(func.lower(ModelArchitecture.family) == str(family).strip().lower())
    if task_type:
        query = query.filter(ModelArchitecture.task_type == task_type)
    if engine:
        normalized_engine = _normalize_and_validate_engine(engine)
        query = query.filter(func.lower(ModelArchitecture.engine) == normalized_engine)
    if custom_model_package_id is not None:
        query = query.filter(ModelArchitecture.custom_model_package_id == int(custom_model_package_id))
    if q:
        like = f"%{str(q).strip()}%"
        query = query.filter(ModelArchitecture.variant.ilike(like))
    return query.order_by(ModelArchitecture.family.asc(), ModelArchitecture.variant.asc()).offset(skip).limit(limit).all()


def create_architecture(
    db: Session,
    *,
    family: str,
    variant: str,
    task_type: TaskType,
    engine: str | None = "ultralytics-yolo",
    custom_model_package_id: int | None = None,
    pretrained_path: str | None = None,
    description: str | None = None,
    default_params: dict | None = None,
) -> ModelArchitecture:
    family = str(family or "").strip()
    variant = str(variant or "").strip()
    if not family or not variant:
        raise ValidationError("family and variant are required")

    normalized_engine = _normalize_and_validate_engine(engine)

    # Validate custom_model_package_id constraint per engine
    if normalized_engine == "custom-source":
        if custom_model_package_id is None:
            raise ValidationError("custom_model_package_id is required when engine is 'custom-source'")
        pkg = (
            db.query(CustomModelPackage)
            .populate_existing()
            .with_for_update()
            .filter(CustomModelPackage.package_id == int(custom_model_package_id))
            .first()
        )
        if not pkg:
            raise ValidationError(f"Referenced CustomModelPackage id={custom_model_package_id} does not exist")
        if pkg.retired_at is not None:
            raise ValidationError(
                f"Referenced CustomModelPackage id={custom_model_package_id} is retired and cannot be linked to new architectures"
            )
    else:
        if custom_model_package_id is not None:
            raise ValidationError(f"custom_model_package_id must be NULL for built-in engine '{normalized_engine}'")

    exists = (
        db.query(ModelArchitecture)
        .filter(
            func.lower(ModelArchitecture.family) == family.lower(),
            func.lower(ModelArchitecture.variant) == variant.lower(),
            ModelArchitecture.task_type == task_type,
        )
        .first()
    )
    if exists:
        raise ConflictError("Architecture already exists")

    row = ModelArchitecture(
        family=family,
        variant=variant,
        task_type=task_type,
        engine=normalized_engine,
        custom_model_package_id=int(custom_model_package_id) if custom_model_package_id is not None else None,
        pretrained_path=pretrained_path,
        description=description,
        default_params=default_params,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


__all__ = ["create_architecture", "list_architectures"]

