from __future__ import annotations

# Compatibility aliases for legacy imports.
# V3 separates the dataset domain into illegal/standard datasets.

from train_platform.models.v3.illegal_dataset import IllegalDatasetVersion as DatasetVersion
from train_platform.models.v3.standard_dataset import StandardDataset as Dataset

__all__ = ["Dataset", "DatasetVersion"]
