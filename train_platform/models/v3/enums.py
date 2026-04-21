from __future__ import annotations

import enum


class DatasetType(str, enum.Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


class TaskType(str, enum.Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


class DatasetVersionStatus(str, enum.Enum):
    CREATED = "created"
    FINALIZED = "finalized"
    FAILED = "failed"


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TrainingRunStatus(str, enum.Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class DeploymentPlatform(str, enum.Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DELETING = "deleting"


class DeploymentTriggerType(str, enum.Enum):
    MANUAL = "manual"


class DeploymentRunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentRunPhase(str, enum.Enum):
    PREPARING = "preparing"
    VALIDATE_ARTIFACTS = "validate_artifacts"
    MATERIALIZE_RUNTIME = "materialize_runtime"
    SMOKE_TEST = "smoke_test"
    ACTIVATE = "activate"
    DONE = "done"
    CANCELLED = "cancelled"


class ModelStage(str, enum.Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
