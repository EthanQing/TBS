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

    # Only seed when table is empty.
    # If user already has rows (custom architectures), leave them untouched.
    has_any = db.query(ModelArchitecture.architecture_id).limit(1).first()
    if has_any:
        return

    db.add_all([ModelArchitecture(**d) for d in DEFAULT_ARCHITECTURES])
    db.commit()
