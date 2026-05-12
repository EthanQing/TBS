from __future__ import annotations

import logging

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from train_platform.db.session import SessionLocal, engine

logger = logging.getLogger("train_platform.db.init")


def init_db() -> None:
    """
    Seed minimal reference data.

    NOTE: Schema migrations should be managed via Alembic (see alembic.ini).
    This function intentionally does NOT call Base.metadata.create_all().
    """
    _ensure_v3_schema_ready()
    with SessionLocal() as db:
        _seed_architectures(db)
        _seed_alarm_rules(db)


def _ensure_v3_schema_ready() -> None:
    from train_platform.models import v3 as _models_v3  # noqa: F401
    from train_platform.models.v3 import V3Base

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    required_tables = set(V3Base.metadata.tables.keys())
    missing_tables = sorted(required_tables - existing_tables)
    if not missing_tables:
        return

    preview = ", ".join(missing_tables[:8])
    if len(missing_tables) > 8:
        preview = f"{preview}, ... (+{len(missing_tables) - 8} more)"

    raise RuntimeError(
        "V3 database schema is incomplete. "
        f"Missing tables: {preview}. "
        "Please run 'alembic -c alembic.ini upgrade head'. "
        "If Alembic is already at head but tables are still missing, rebuild the dev database and run the migration again."
    )


def _seed_architectures(db: Session) -> None:
    from train_platform.models.v3.architecture import ModelArchitecture
    from train_platform.db.seed_data import DEFAULT_ARCHITECTURES

    # Fetch existing rows to avoid duplicates
    existing_keys = {
        (str(row[0]), str(row[1]), str(row[2]))
        for row in db.query(ModelArchitecture.variant, ModelArchitecture.family, ModelArchitecture.task_type).all()
    }

    to_add = []
    for d in DEFAULT_ARCHITECTURES:
        key = (str(d["variant"]), str(d["family"]), str(d["task_type"]))
        if key not in existing_keys:
            to_add.append(ModelArchitecture(**d))

    if to_add:
        db.add_all(to_add)
        db.commit()
        families = sorted({str(item.family) for item in to_add if getattr(item, "family", None)})
        logger.info(
            "Seeded %s model architectures on startup. Families=%s",
            len(to_add),
            ",".join(families),
        )
    else:
        logger.info("Model architectures already seeded; no new rows added.")


def _seed_alarm_rules(db: Session) -> None:
    try:
        from train_platform.services.v3.alarm_service import AlarmService

        AlarmService().ensure_default_rules(db)
    except Exception as e:
        logger.warning("Failed to seed alarm rules: %s", e)
