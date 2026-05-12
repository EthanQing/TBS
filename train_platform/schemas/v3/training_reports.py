from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ReportBasicInfo(BaseModel):
    """Training run basic information."""

    run_id: str
    name: Optional[str] = None
    framework_label: str
    framework_key: str
    engine: str
    status: str
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class ReportDataset(BaseModel):
    """Dataset information."""

    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None


class ReportArchitecture(BaseModel):
    """Model architecture selection."""

    architecture_id: int
    family: str
    variant: str
    task_type: str
    description: Optional[str] = None
    pretrained_path: Optional[str] = None


class ReportParameters(BaseModel):
    """Training hyper-parameters."""

    epochs: int
    batch_size: int
    image_size: int
    learning_rate: Optional[float] = None
    patience: int
    device: str
    workers: int
    optimizer: str
    use_pretrained: bool
    save_period: Optional[int] = None
    augmentation: Optional[Dict[str, Any]] = None
    loss_weights: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None


class ReportMetrics(BaseModel):
    """Final model metrics."""

    best_metrics: Optional[Dict[str, Any]] = None
    final_metrics: Optional[Dict[str, Any]] = None
    core_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Selected key metrics extracted from best/final metric snapshots.",
    )


class ReportArtifacts(BaseModel):
    """Model artifacts."""

    best_weights_path: Optional[str] = None
    last_weights_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    flops: Optional[int] = None


class TrainingRunReportOut(BaseModel):
    """Structured training result report."""

    basic: ReportBasicInfo
    dataset: ReportDataset
    architecture: ReportArchitecture
    parameters: ReportParameters
    metrics: ReportMetrics
    artifacts: ReportArtifacts
