from __future__ import annotations

# Import order matters only for type-checkers; SQLAlchemy resolves relationships by string name.

from train_platform.models.architecture import ModelArchitecture
from train_platform.models.alarm import AlarmAlert, AlarmRule
from train_platform.models.chart_config import ChartConfig
from train_platform.models.dataset_event import DatasetEvent
from train_platform.models.dataset import Dataset, DatasetVersion
from train_platform.models.dataset_image import DatasetImage
from train_platform.models.deployment import Deployment, DeploymentLog
from train_platform.models.deployment_run import DeploymentRun
from train_platform.models.enums import (
    DatasetType,
    DatasetSplit,
    DatasetVersionStatus,
    DeploymentPlatform,
    DeploymentRunPhase,
    DeploymentRunStatus,
    DeploymentTriggerType,
    DeploymentStatus,
    LogLevel,
    ModelStage,
    TaskType,
    TrainingRunStatus,
)
from train_platform.models.inference import InferenceRun
from train_platform.models.model_registry import ModelVersion
from train_platform.models.project import Project
from train_platform.models.training_run import (
    TrainingRun,
    TrainingRunArtifact,
    TrainingRunEpochMetric,
    TrainingRunEvent,
    TrainingRunParameters,
    TrainingRunResult,
)
from train_platform.models.training_run_meta import TrainingRunMeta

__all__ = [
    # core
    "ChartConfig",
    "AlarmRule",
    "AlarmAlert",
    "Dataset",
    "DatasetVersion",
    "DatasetEvent",
    "DatasetImage",
    "Project",
    "ModelArchitecture",
    "TrainingRun",
    "TrainingRunParameters",
    "TrainingRunResult",
    "TrainingRunEpochMetric",
    "TrainingRunArtifact",
    "TrainingRunEvent",
    "TrainingRunMeta",
    "ModelVersion",
    "Deployment",
    "DeploymentLog",
    "DeploymentRun",
    "InferenceRun",
    # enums
    "DatasetType",
    "DatasetSplit",
    "TaskType",
    "DatasetVersionStatus",
    "TrainingRunStatus",
    "ModelStage",
    "DeploymentPlatform",
    "DeploymentTriggerType",
    "DeploymentStatus",
    "DeploymentRunStatus",
    "DeploymentRunPhase",
    "LogLevel",
]
