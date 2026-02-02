from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.db.session import SessionLocal


def init_db() -> None:
    """
    Seed minimal reference data.

    NOTE: Schema migrations should be managed via Alembic (see alembic.ini).
    This function intentionally does NOT call Base.metadata.create_all().
    """
    with SessionLocal() as db:
        _seed_architectures(db)


def _seed_architectures(db: Session) -> None:
    from train_platform.models.architecture import ModelArchitecture
    from train_platform.db.seed_data import DEFAULT_ARCHITECTURES

    # Fetch existing variants to avoid duplicates
    existing_variants_and_families = {
        (row[0], row[1]) for row in db.query(ModelArchitecture.variant, ModelArchitecture.family).all()
    }

    to_add = []
    for d in DEFAULT_ARCHITECTURES:
        if (d["variant"], d["family"]) not in existing_variants_and_families:
            to_add.append(ModelArchitecture(**d))

    if to_add:
        db.add_all(to_add)
        db.commit()
