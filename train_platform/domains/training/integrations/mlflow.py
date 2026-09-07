from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.training_run_meta import TrainingRunMeta


def _truthy(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in ("1", "true", "yes", "on")


def mlflow_enabled() -> bool:
    if os.getenv("MLFLOW_ENABLE") is not None:
        return _truthy(os.getenv("MLFLOW_ENABLE"))
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def to_file_uri(path: str | os.PathLike) -> str:
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return f"file:{str(path)}"


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI") or to_file_uri(settings.training_dir / "mlruns")


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "TrainPlatform")


def _default_artifact_location(tracking_uri: str) -> Optional[str]:
    if tracking_uri.startswith("file:") or "://" not in tracking_uri:
        return (settings.training_dir / "mlruns").as_posix()
    return None


def _get_mlflow_client() -> tuple[Any, str] | tuple[None, None]:
    try:
        from mlflow.tracking import MlflowClient

        tracking_uri = get_tracking_uri()
        return MlflowClient(tracking_uri=tracking_uri), tracking_uri
    except Exception:
        return None, None


def _get_or_create_experiment_id(client: Any, tracking_uri: str) -> str:
    name = get_experiment_name()
    experiment = client.get_experiment_by_name(name)
    if experiment:
        return str(experiment.experiment_id)
    artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION") or _default_artifact_location(tracking_uri)
    if artifact_location:
        return str(client.create_experiment(name, artifact_location=artifact_location))
    return str(client.create_experiment(name))


def get_mlflow_binding(db: Session, run_id: str) -> Optional[dict[str, str]]:
    meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()
    if not meta or not isinstance(meta.extra, dict):
        return None
    binding = {}
    for key in ("mlflow_run_id", "mlflow_experiment_id", "mlflow_tracking_uri"):
        value = meta.extra.get(key)
        if value:
            binding[key] = str(value)
    return binding or None


def set_mlflow_binding(db: Session, run_id: str, binding: Dict[str, Any]) -> None:
    """Mutate TrainingRunMeta in the caller's transaction; never commit or rollback."""

    updates = {
        key: str(binding[key])
        for key in ("mlflow_run_id", "mlflow_experiment_id", "mlflow_tracking_uri")
        if binding.get(key)
    }
    if not updates:
        return
    meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()
    if not meta:
        meta = TrainingRunMeta(run_id=str(run_id))
        db.add(meta)
    extra = dict(meta.extra) if isinstance(meta.extra, dict) else {}
    extra.update(updates)
    meta.extra = extra


@dataclass
class MlflowRunLogger:
    client: Any
    run_id: str
    experiment_id: str
    tracking_uri: str
    binding_to_persist: Optional[dict[str, str]] = None

    def log_metrics(self, metrics: Dict[str, float], *, step: int) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            try:
                self.client.log_metric(self.run_id, str(key), float(value), step=int(step))
            except Exception:
                continue

    def log_params(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        for key, value in params.items():
            try:
                self.client.log_param(self.run_id, str(key), str(value))
            except Exception:
                continue

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not tags:
            return
        for key, value in tags.items():
            try:
                self.client.set_tag(self.run_id, str(key), str(value))
            except Exception:
                continue

    def terminate(self, status: str = "FINISHED") -> None:
        try:
            self.client.set_terminated(self.run_id, status=str(status or "FINISHED"))
        except Exception:
            pass


def initialize_mlflow_logger(
    run: Any,
    *,
    existing_binding: Mapping[str, Any] | None = None,
    dataset_path: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> Optional[MlflowRunLogger]:
    """Initialize tracking from a caller-loaded binding.

    This function only performs external MLflow operations. The caller owns
    loading and persisting the narrow TrainingRunMeta binding.
    """

    if not mlflow_enabled():
        return None
    client, tracking_uri = _get_mlflow_client()
    if client is None or tracking_uri is None:
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        pass

    # External setup is optional. A failing tracking server or experiment must
    # never prevent the trainer from starting.
    try:
        experiment_id = _get_or_create_experiment_id(client, tracking_uri)
        mlflow_run_id = existing_binding.get("mlflow_run_id") if existing_binding else None
        if mlflow_run_id:
            try:
                client.get_run(str(mlflow_run_id))
            except Exception:
                mlflow_run_id = None
        created = False
        if not mlflow_run_id:
            tags = {
                "train.run_id": str(run.run_id),
                "train.project_id": str(getattr(run, "project_id", "")),
                "train.standard_dataset_id": str(getattr(run, "standard_dataset_id", "")),
                "train.architecture_id": str(getattr(run, "architecture_id", "")),
                "train.name": str(getattr(run, "name", "")),
            }
            if dataset_path:
                tags["train.dataset_path"] = str(dataset_path)
            if run_dir:
                tags["train.run_dir"] = str(run_dir)
            info = client.create_run(experiment_id, tags=tags)
            mlflow_run_id = info.info.run_id
            created = True
    except Exception:
        return None

    binding_to_persist = None
    if created:
        binding_to_persist = {
            "mlflow_run_id": str(mlflow_run_id),
            "mlflow_experiment_id": str(experiment_id),
            "mlflow_tracking_uri": str(tracking_uri),
        }

    logger = MlflowRunLogger(
        client=client,
        run_id=str(mlflow_run_id),
        experiment_id=str(experiment_id),
        tracking_uri=str(tracking_uri),
        binding_to_persist=binding_to_persist,
    )
    if created:
        params: Dict[str, Any] = {}
        parameters = getattr(run, "parameters", None)
        if parameters is not None:
            params.update(
                {
                    "epochs": getattr(parameters, "epochs", None),
                    "batch_size": getattr(parameters, "batch_size", None),
                    "image_size": getattr(parameters, "image_size", None),
                    "learning_rate": getattr(parameters, "learning_rate", None),
                    "lr_scheduler": getattr(parameters, "lr_scheduler", "linear"),
                    "patience": getattr(parameters, "patience", None),
                    "device": getattr(parameters, "device", None),
                    "workers": getattr(parameters, "workers", None),
                    "use_pretrained": getattr(parameters, "use_pretrained", None),
                    "optimizer": getattr(parameters, "optimizer", None),
                }
            )
            additional = getattr(parameters, "additional_params", None) or {}
            if isinstance(additional, dict):
                for key, value in additional.items():
                    params.setdefault(key, value)
        logger.log_params({key: value for key, value in params.items() if value is not None})
    return logger


def resolve_mlflow_run_id(db: Session, run_id: str) -> Optional[str]:
    try:
        binding = get_mlflow_binding(db, run_id)
    except Exception:
        return None
    return binding.get("mlflow_run_id") if binding else None


def fetch_mlflow_epoch_metrics(db: Session, run_id: str, *, limit: int = 5000) -> Optional[list[dict]]:
    if not mlflow_enabled():
        return None
    client, _tracking_uri = _get_mlflow_client()
    if client is None:
        return None
    mlflow_run_id = resolve_mlflow_run_id(db, run_id)
    if not mlflow_run_id:
        return None
    try:
        run = client.get_run(str(mlflow_run_id))
    except Exception:
        return None

    metric_keys = list(getattr(run.data, "metrics", {}).keys())
    if not metric_keys:
        return []
    metrics_by_epoch: dict[int, dict[str, float]] = {}
    timestamps_by_epoch: dict[int, int] = {}
    for key in metric_keys:
        try:
            history = client.get_metric_history(str(mlflow_run_id), key)
        except Exception:
            history = []
        for metric in history or []:
            try:
                epoch = int(getattr(metric, "step", 0) or 0)
                value = float(getattr(metric, "value", 0.0))
            except (TypeError, ValueError):
                continue
            metrics_by_epoch.setdefault(epoch, {})[str(key)] = value
            try:
                timestamp = int(getattr(metric, "timestamp", 0) or 0)
            except (TypeError, ValueError):
                timestamp = 0
            if timestamp:
                timestamps_by_epoch[epoch] = max(timestamps_by_epoch.get(epoch, 0), timestamp)

    rows: list[dict] = []
    for index, epoch in enumerate(sorted(metrics_by_epoch.keys())):
        timestamp = timestamps_by_epoch.get(epoch) or int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        rows.append(
            {
                "metric_id": index + 1,
                "run_id": str(run_id),
                "epoch": int(epoch),
                "metrics": metrics_by_epoch[epoch],
                "created_at": datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc),
            }
        )
    return rows[: int(limit)] if limit and len(rows) > int(limit) else rows


__all__ = [
    "MlflowRunLogger",
    "fetch_mlflow_epoch_metrics",
    "get_experiment_name",
    "get_mlflow_binding",
    "get_tracking_uri",
    "initialize_mlflow_logger",
    "mlflow_enabled",
    "resolve_mlflow_run_id",
    "set_mlflow_binding",
    "to_file_uri",
]
