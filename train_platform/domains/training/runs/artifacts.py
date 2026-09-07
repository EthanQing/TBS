from __future__ import annotations

from pathlib import Path
from typing import Mapping

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.domains.training.frameworks.contract import (
    TrainingArtifactReport,
    validate_artifact_path,
)
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunArtifact,
    TrainingRunEpochMetric,
    TrainingRunResult,
)


LOSS_METRIC_TERMS: tuple[str, ...] = ("loss", "l1", "dfl")


def _metric_number(value) -> float | None:
    """Return a JSON-safe finite number, or None for non-numeric metrics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        n = float(value)
    except Exception:
        return None
    return n if n == n else None


def is_lower_better_metric(key: str) -> bool:
    key_lower = str(key or "").lower()
    return any(term in key_lower for term in LOSS_METRIC_TERMS)


def _json_compatible(value):
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def compute_epoch_metric_snapshots(db: Session, run_id: str) -> tuple[dict[str, float] | None, dict | None]:
    rows = (
        db.query(TrainingRunEpochMetric)
        .filter(TrainingRunEpochMetric.run_id == str(run_id))
        .order_by(TrainingRunEpochMetric.epoch.asc())
        .all()
    )
    if not rows:
        return None, None
    final_metrics = rows[-1].metrics if isinstance(rows[-1].metrics, dict) else {}
    best: dict[str, float] = {}
    for row in rows:
        metrics = row.metrics if isinstance(row.metrics, dict) else {}
        for key, value in metrics.items():
            number = _metric_number(value)
            if number is None:
                continue
            current_best = best.get(key)
            if current_best is None:
                best[key] = number
            elif is_lower_better_metric(key):
                best[key] = min(current_best, number)
            else:
                best[key] = max(current_best, number)
    return best, final_metrics


def register_reported_artifact(
    db: Session,
    run_id: str,
    report: TrainingArtifactReport,
    *,
    expected_pid: int | None = None,
) -> TrainingRunArtifact | None:
    """Validate and persist an artifact reported by a running custom trainer."""

    if not isinstance(report, TrainingArtifactReport):
        raise TypeError("artifact report must be a TrainingArtifactReport")

    run_id = str(run_id)
    relative_path = validate_artifact_path(report.path)
    base = settings.training_dir.resolve()
    output_root_path = base / run_id / "custom_model" / "output"
    output_root = output_root_path.resolve(strict=False)
    if output_root != base and base not in output_root.parents:
        raise ValueError("custom training output directory escapes the training directory")
    candidate = output_root / relative_path
    resolved = candidate.resolve(strict=False)
    if resolved != output_root and output_root not in resolved.parents:
        raise ValueError("reported artifact path escapes the custom training output directory")
    if not candidate.exists():
        raise ValueError(f"reported artifact does not exist: {relative_path}")
    if not candidate.is_file():
        raise ValueError(f"reported artifact is not a regular file: {relative_path}")

    run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
    if not run or run.status != TrainingRunStatus.RUNNING:
        return None
    if expected_pid is not None and (run.pid is None or int(run.pid) != int(expected_pid)):
        return None

    try:
        size_bytes = int(candidate.stat().st_size)
    except OSError:
        size_bytes = None

    meta = _json_compatible(report.meta or {})
    meta["source"] = "reported"
    meta.pop("format", None)
    if report.format is not None:
        meta["format"] = report.format
    role = report.role
    kind = "weights" if role in {"best_weights", "last_weights"} else "artifact"
    stored_path = candidate.relative_to(base).as_posix()

    query = db.query(TrainingRunArtifact).filter(
        TrainingRunArtifact.run_id == run_id,
        TrainingRunArtifact.role == role,
    )
    if role not in {"best_weights", "last_weights"}:
        query = query.filter(TrainingRunArtifact.path == stored_path)
    artifact = query.order_by(TrainingRunArtifact.artifact_id.desc()).first()
    if artifact is None:
        artifact = TrainingRunArtifact(run_id=run_id, role=role)
        db.add(artifact)
    artifact.kind = kind
    artifact.role = role
    artifact.name = candidate.name
    artifact.path = stored_path
    artifact.size_bytes = size_bytes
    artifact.sha256 = None
    artifact.meta = meta
    db.commit()
    return artifact


def index_completion_artifacts(db: Session, run_id: str) -> None:
    """Index completion artifacts and update the shared training result row."""

    base = settings.training_dir
    run_dir = base / str(run_id)
    existing_artifacts = db.query(TrainingRunArtifact).filter(TrainingRunArtifact.run_id == str(run_id)).all()
    for artifact in existing_artifacts:
        if not isinstance(artifact.meta, Mapping) or artifact.meta.get("source") != "reported":
            db.delete(artifact)
    db.flush()

    candidates: list[tuple[str, str, Path]] = [
        ("weights", "best.pt", run_dir / "weights" / "best.pt"),
        ("weights", "last.pt", run_dir / "weights" / "last.pt"),
        ("weights", "best.pdparams", run_dir / "weights" / "best.pdparams"),
        ("weights", "last.pdparams", run_dir / "weights" / "last.pdparams"),
        ("weights", "best.pdopt", run_dir / "weights" / "best.pdopt"),
        ("weights", "last.pdopt", run_dir / "weights" / "last.pdopt"),
        ("export", "best.onnx", run_dir / "weights" / "best.onnx"),
        ("export", "last.onnx", run_dir / "weights" / "last.onnx"),
        ("csv", "results.csv", run_dir / "results.csv"),
        ("config", "args.yaml", run_dir / "args.yaml"),
        ("config", "results.yaml", run_dir / "results.yaml"),
        ("log", "train.stdout.log", run_dir / "logs" / "train.stdout.log"),
        ("log", "train.stderr.log", run_dir / "logs" / "train.stderr.log"),
    ]
    for name in (
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ):
        candidates.append(("plot", name, run_dir / name))

    reported_roles = {
        str(artifact.role)
        for artifact in existing_artifacts
        if isinstance(artifact.meta, Mapping)
        and artifact.meta.get("source") == "reported"
        and artifact.role in {"best_weights", "last_weights"}
    }
    indexed_builtin_roles: set[str] = set()
    for kind, name, abs_path in candidates:
        if not abs_path.exists() or not abs_path.is_file():
            continue
        try:
            rel = abs_path.relative_to(base).as_posix()
        except Exception:
            rel = str(abs_path)
        try:
            size_bytes = int(abs_path.stat().st_size)
        except Exception:
            size_bytes = None
        role = (
            "best_weights"
            if name in {"best.pt", "best.pdparams"}
            else "last_weights"
            if name in {"last.pt", "last.pdparams"}
            else None
        )
        if role is not None:
            if role in reported_roles or role in indexed_builtin_roles:
                role = None
            else:
                indexed_builtin_roles.add(role)
        db.add(
            TrainingRunArtifact(
                run_id=str(run_id),
                kind=kind,
                role=role,
                name=name,
                path=rel,
                size_bytes=size_bytes,
            )
        )

    db.flush()

    result = db.query(TrainingRunResult).filter(TrainingRunResult.run_id == str(run_id)).first()
    if not result:
        result = TrainingRunResult(run_id=str(run_id))
        db.add(result)
    result.results_dir = str(run_id)

    role_artifacts = (
        db.query(TrainingRunArtifact)
        .filter(
            TrainingRunArtifact.run_id == str(run_id),
            TrainingRunArtifact.role.in_(("best_weights", "last_weights")),
        )
        .order_by(TrainingRunArtifact.artifact_id.desc())
        .all()
    )
    by_role: dict[str, TrainingRunArtifact] = {}
    for artifact in role_artifacts:
        by_role.setdefault(str(artifact.role), artifact)
    best = by_role.get("best_weights")
    last = by_role.get("last_weights")
    result.best_weights_path = best.path if best else None
    result.last_weights_path = last.path if last else None
    size_source = best or last
    if size_source:
        try:
            size_path = base / str(size_source.path)
            result.model_size_mb = round(size_path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass

    best_metrics, final_metrics = compute_epoch_metric_snapshots(db, str(run_id))
    if best_metrics is not None and final_metrics is not None:
        result.best_metrics = best_metrics
        result.final_metrics = final_metrics


__all__ = [
    "compute_epoch_metric_snapshots",
    "index_completion_artifacts",
    "is_lower_better_metric",
    "register_reported_artifact",
]
