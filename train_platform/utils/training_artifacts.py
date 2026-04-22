from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.training_run import TrainingRunArtifact, TrainingRunResult


def index_completion_artifacts(db: Session, run_id: str) -> None:
    """
    Index common Ultralytics artifacts for UI download/inspection.

    This function is idempotent (replaces existing artifact rows for the run).
    """
    base = settings.training_dir
    run_dir = base / str(run_id)

    db.query(TrainingRunArtifact).filter(TrainingRunArtifact.run_id == str(run_id)).delete()

    candidates: list[tuple[str, str, Path]] = [
        ("weights", "best.pt", run_dir / "weights" / "best.pt"),
        ("weights", "last.pt", run_dir / "weights" / "last.pt"),
        ("weights", "best.pdparams", run_dir / "weights" / "best.pdparams"),
        ("weights", "last.pdparams", run_dir / "weights" / "last.pdparams"),
        ("weights", "best.pdopt", run_dir / "weights" / "best.pdopt"),
        ("weights", "last.pdopt", run_dir / "weights" / "last.pdopt"),
        # Common export outputs (generated on-demand via /training-runs/{id}/export)
        ("export", "best.onnx", run_dir / "weights" / "best.onnx"),
        ("export", "last.onnx", run_dir / "weights" / "last.onnx"),
        ("csv", "results.csv", run_dir / "results.csv"),
        ("config", "args.yaml", run_dir / "args.yaml"),
        ("config", "results.yaml", run_dir / "results.yaml"),
        ("log", "train.stdout.log", run_dir / "logs" / "train.stdout.log"),
        ("log", "train.stderr.log", run_dir / "logs" / "train.stderr.log"),
    ]

    plot_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    for name in plot_names:
        candidates.append(("plot", name, run_dir / name))

    for kind, name, abs_path in candidates:
        if not abs_path.exists() or not abs_path.is_file():
            continue
        try:
            rel = abs_path.relative_to(base).as_posix()
        except Exception:
            rel = str(abs_path)

        size_bytes = None
        try:
            size_bytes = int(abs_path.stat().st_size)
        except Exception:
            size_bytes = None

        db.add(
            TrainingRunArtifact(
                run_id=str(run_id),
                kind=kind,
                name=name,
                path=rel,
                size_bytes=size_bytes,
            )
        )

    # Update/Upsert TrainingRunResult for the V3 model registry flow.
    res = db.query(TrainingRunResult).filter(TrainingRunResult.run_id == str(run_id)).first()
    if not res:
        res = TrainingRunResult(run_id=str(run_id))
        db.add(res)

    res.results_dir = str(run_id)

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    best_pd = run_dir / "weights" / "best.pdparams"
    last_pd = run_dir / "weights" / "last.pdparams"

    best = (
        best_pt
        if best_pt.exists()
        else best_pd
        if best_pd.exists()
        else None
    )
    last = (
        last_pt
        if last_pt.exists()
        else last_pd
        if last_pd.exists()
        else None
    )

    res.best_weights_path = best.relative_to(base).as_posix() if best else None
    res.last_weights_path = last.relative_to(base).as_posix() if last else None

    size_source = best or last
    if size_source and size_source.exists():
        try:
            res.model_size_mb = round(size_source.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass
