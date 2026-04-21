from __future__ import annotations

# Compatibility re-exports for the removed mixed `/datasets` V3 surface.
# Use `illegal_datasets` and `standard_datasets` schemas directly in new code.

from train_platform.schemas.v3.illegal_datasets import (  # noqa: F401
    DatasetFileOut,
    DatasetImageAnnotationsOut,
    DatasetImageUploadOut,
    DatasetStatisticsOut,
    DatasetViewOut,
    IllegalDatasetCreate as DatasetCreate,
    IllegalDatasetDetailOut as DatasetDetailOut,
    IllegalDatasetEventOut as DatasetEventOut,
    IllegalDatasetOut as DatasetOut,
    IllegalDatasetUpdate as DatasetUpdate,
    IllegalDatasetVersionOut as DatasetVersionOut,
)

__all__ = [
    "DatasetCreate",
    "DatasetUpdate",
    "DatasetOut",
    "DatasetDetailOut",
    "DatasetEventOut",
    "DatasetVersionOut",
    "DatasetStatisticsOut",
    "DatasetFileOut",
    "DatasetViewOut",
    "DatasetImageAnnotationsOut",
    "DatasetImageUploadOut",
]
