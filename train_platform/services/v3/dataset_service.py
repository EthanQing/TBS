from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.services.v3.illegal_dataset_service import IllegalDatasetService
from train_platform.services.v3.standard_dataset_service import StandardDatasetService
from train_platform.utils.exceptions import NotFoundError, ValidationError


class DatasetService:
    """Compatibility shim for removed mixed dataset semantics in V3."""

    def get_dataset(self, db: Session, dataset_id: int):
        try:
            return StandardDatasetService().get_dataset(db, int(dataset_id))
        except Exception:
            pass
        try:
            return IllegalDatasetService().get_dataset(db, int(dataset_id))
        except Exception:
            pass
        raise NotFoundError("Dataset not found")

    def __getattr__(self, name: str):  # pragma: no cover - compatibility only
        raise ValidationError(
            "Mixed V3 dataset service was removed. Use IllegalDatasetService or StandardDatasetService explicitly."
        )
