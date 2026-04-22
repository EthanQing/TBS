from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.v3.training_run_meta import TrainingRunMeta


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def mlflow_enabled() -> bool:
    if os.getenv("MLFLOW_ENABLE") is not None:
        return _truthy(os.getenv("MLFLOW_ENABLE"))
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def to_file_uri(path: str | os.PathLike) -> str:
    """Convert a file path to a file URI (file:///...) handling Windows paths correctly."""
    try:
        from pathlib import Path
        return Path(path).resolve().as_uri()
    except Exception:
        return f"file:{str(path)}"


def get_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Use proper file URI for Windows compatibility
    return to_file_uri(settings.training_dir / "mlruns")


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "TrainPlatform")


def _default_artifact_location(tracking_uri: str) -> Optional[str]:
    if tracking_uri.startswith("file:") or "://" not in tracking_uri:
        return (settings.training_dir / "mlruns").as_posix()
    return None


def _get_mlflow_client() -> tuple[Any, str] | tuple[None, None]:
    try:
        from mlflow.tracking import MlflowClient
    except Exception:
        return None, None

    uri = get_tracking_uri()
    try:
        client = MlflowClient(tracking_uri=uri)
    except Exception:
        return None, None
    return client, uri


def _get_or_create_experiment_id(client: Any, tracking_uri: str) -> str:
    name = get_experiment_name()
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION") or _default_artifact_location(tracking_uri)
    if artifact_location:
        return client.create_experiment(name, artifact_location=artifact_location)
    return client.create_experiment(name)


def _get_meta_extra(run_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()
        if meta and isinstance(meta.extra, dict):
            return dict(meta.extra)
        return None
    finally:
        db.close()


def _update_meta_extra(run_id: str, updates: dict) -> None:
    if not updates:
        return
    db = SessionLocal()
    try:
        meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()
        if not meta:
            meta = TrainingRunMeta(run_id=str(run_id))
            db.add(meta)
            db.flush()
        extra = meta.extra if isinstance(meta.extra, dict) else {}
        extra.update(updates)
        meta.extra = extra
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@dataclass
class MlflowRunLogger:
    client: Any
    run_id: str
    experiment_id: str
    tracking_uri: str

    def log_metrics(self, metrics: Dict[str, float], *, step: int) -> None:
        if not metrics:
            return
        try:
            for key, val in metrics.items():
                try:
                    self.client.log_metric(self.run_id, str(key), float(val), step=int(step))
                except Exception:
                    continue
        except Exception:
            return

    def log_params(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        for key, val in params.items():
            try:
                self.client.log_param(self.run_id, str(key), str(val))
            except Exception:
                continue

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not tags:
            return
        for key, val in tags.items():
            try:
                self.client.set_tag(self.run_id, str(key), str(val))
            except Exception:
                continue

    def terminate(self, status: str = "FINISHED") -> None:
        try:
            self.client.set_terminated(self.run_id, status=str(status or "FINISHED"))
        except Exception:
            return


def init_mlflow_logger(run: Any, *, dataset_path: Optional[str] = None, run_dir: Optional[str] = None) -> Optional[MlflowRunLogger]:
    if not mlflow_enabled():
        return None

    client, tracking_uri = _get_mlflow_client()
    if client is None:
        return None

    # Sync global MLflow state for external libraries (like Ultralytics)
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        pass

    experiment_id = _get_or_create_experiment_id(client, tracking_uri)

    extra = _get_meta_extra(str(run.run_id))
    mlflow_run_id = None
    if isinstance(extra, dict):
        mlflow_run_id = extra.get("mlflow_run_id")

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

        _update_meta_extra(
            str(run.run_id),
            {
                "mlflow_run_id": mlflow_run_id,
                "mlflow_experiment_id": experiment_id,
                "mlflow_tracking_uri": tracking_uri,
            },
        )

    logger = MlflowRunLogger(
        client=client,
        run_id=str(mlflow_run_id),
        experiment_id=str(experiment_id),
        tracking_uri=str(tracking_uri),
    )

    if created:
        params = {}
        p = getattr(run, "parameters", None)
        if p is not None:
            params.update(
                {
                    "epochs": getattr(p, "epochs", None),
                    "batch_size": getattr(p, "batch_size", None),
                    "image_size": getattr(p, "image_size", None),
                    "learning_rate": getattr(p, "learning_rate", None),
                    "patience": getattr(p, "patience", None),
                    "device": getattr(p, "device", None),
                    "workers": getattr(p, "workers", None),
                    "use_pretrained": getattr(p, "use_pretrained", None),
                    "optimizer": getattr(p, "optimizer", None),
                }
            )
            add = getattr(p, "additional_params", None) or {}
            if isinstance(add, dict):
                for k, v in add.items():
                    params.setdefault(k, v)
        params = {k: v for k, v in params.items() if v is not None}
        logger.log_params(params)

    return logger


def resolve_mlflow_run_id(db: Any, run_id: str) -> Optional[str]:
    meta = db.query(TrainingRunMeta).filter(TrainingRunMeta.run_id == str(run_id)).first()
    if not meta or not isinstance(meta.extra, dict):
        return None
    val = meta.extra.get("mlflow_run_id")
    return str(val) if val else None


def fetch_mlflow_epoch_metrics(db: Any, run_id: str, *, limit: int = 5000) -> Optional[list[dict]]:
    if not mlflow_enabled():
        return None

    client, tracking_uri = _get_mlflow_client()
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
    ts_by_epoch: dict[int, int] = {}

    for key in metric_keys:
        try:
            history = client.get_metric_history(str(mlflow_run_id), key)
        except Exception:
            history = []
        for m in history or []:
            try:
                step = int(getattr(m, "step", 0) or 0)
            except Exception:
                step = 0
            try:
                val = float(getattr(m, "value", 0.0))
            except Exception:
                continue
            metrics_by_epoch.setdefault(step, {})[str(key)] = val
            try:
                ts = int(getattr(m, "timestamp", 0) or 0)
            except Exception:
                ts = 0
            if ts:
                ts_by_epoch[step] = max(ts_by_epoch.get(step, 0), ts)

    rows: list[dict] = []
    for idx, epoch in enumerate(sorted(metrics_by_epoch.keys())):
        metrics = metrics_by_epoch.get(epoch, {})
        ts_ms = ts_by_epoch.get(epoch, 0)
        if not ts_ms:
            ts_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        created_at = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        rows.append(
            {
                "metric_id": idx + 1,
                "run_id": str(run_id),
                "epoch": int(epoch),
                "metrics": metrics,
                "created_at": created_at,
            }
        )

    if limit and len(rows) > int(limit):
        rows = rows[: int(limit)]

    return rows
