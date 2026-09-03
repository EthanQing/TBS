from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from io import BytesIO

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.training_run import TrainingRunArtifact
from train_platform.platform.runtime import ModelWorkerClient, ModelWorkerError
from train_platform.utils.exceptions import NotFoundError, ValidationError

from .reports import build_report
from .service import TrainingRunService


@dataclass(frozen=True)
class TrainingExport:
    run_id: str
    format: str
    weights: str
    artifact: TrainingRunArtifact | None = None


@dataclass(frozen=True)
class ExportDownload:
    path: Path | None = None
    content: bytes | None = None
    filename: str | None = None


def _normalize_export(format: str | None, weights: str | None) -> tuple[str, str]:
    fmt = str(format or "pt").strip().lower()
    weights_key = str(weights or "best").strip().lower()
    if fmt not in ("pt", "onnx"):
        raise ValidationError("Unsupported export format")
    if weights_key not in ("best", "last"):
        raise ValidationError("weights must be 'best' or 'last'")
    return fmt, weights_key


def _safe_run_path(run_id: str, filename: str) -> Path:
    base = settings.training_dir.resolve()
    path = (settings.training_dir / str(run_id) / "weights" / filename).resolve(strict=False)
    if base not in path.parents:
        raise ValidationError("Unsafe weights path")
    return path


def export_training_run(
    db: Session,
    run_id: str,
    *,
    format: str | None = "pt",
    weights: str | None = "best",
    dynamic: bool = False,
    opset: int | None = None,
    imgsz: int | None = None,
) -> TrainingExport:
    run = TrainingRunService().get_run(db, run_id)
    fmt, weights_key = _normalize_export(format, weights)
    src_weights = _safe_run_path(str(run.run_id), f"{weights_key}.pt")
    if not src_weights.exists():
        raise ValidationError("Weights not found")

    if fmt == "pt":
        return TrainingExport(str(run.run_id), fmt, weights_key)

    out_name = f"{weights_key}.onnx"
    out_onnx = _safe_run_path(str(run.run_id), out_name)
    if not out_onnx.exists() or out_onnx.stat().st_size <= 0:
        try:
            ModelWorkerClient().export_ultralytics_onnx(
                src_pt=src_weights,
                out_onnx=out_onnx,
                dynamic=bool(dynamic),
                opset=opset,
                imgsz=imgsz,
            )
        except ModelWorkerError as exc:
            raise ValidationError(f"Failed to reach inference worker: {exc}") from exc

        if not out_onnx.exists() or out_onnx.stat().st_size <= 0:
            newest: Path | None = None
            run_root = (settings.training_dir / str(run.run_id)).resolve(strict=False)
            try:
                candidates = list(out_onnx.parent.glob("*.onnx"))
                if not candidates:
                    candidates = list(run_root.rglob("*.onnx"))
                for candidate in candidates:
                    if newest is None or candidate.stat().st_mtime > newest.stat().st_mtime:
                        newest = candidate
            except Exception:
                newest = None

            if newest and newest.exists() and newest != out_onnx and newest.stat().st_size > 0:
                try:
                    import shutil

                    shutil.copy2(newest, out_onnx)
                except Exception:
                    pass

        if not out_onnx.exists() or out_onnx.stat().st_size <= 0:
            raise ValidationError("ONNX export failed: non-empty output file not found")

    relative_path = out_onnx.relative_to(settings.training_dir).as_posix()
    db.query(TrainingRunArtifact).filter(
        TrainingRunArtifact.run_id == str(run.run_id),
        TrainingRunArtifact.kind == "export",
        TrainingRunArtifact.name == out_name,
    ).delete()
    try:
        size_bytes = int(out_onnx.stat().st_size)
    except Exception:
        size_bytes = None
    artifact = TrainingRunArtifact(
        run_id=str(run.run_id),
        kind="export",
        name=out_name,
        path=relative_path,
        size_bytes=size_bytes,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return TrainingExport(str(run.run_id), fmt, weights_key, artifact)


def download_export(
    db: Session,
    run_id: str,
    *,
    format: str | None = "pt",
    weights: str | None = "best",
    include_report: bool = False,
) -> ExportDownload:
    run = TrainingRunService().get_run(db, run_id)
    fmt, weights_key = _normalize_export(format, weights)
    extension = "onnx" if fmt == "onnx" else "pt"
    path = _safe_run_path(str(run.run_id), f"{weights_key}.{extension}")
    if not path.exists() or not path.is_file():
        raise NotFoundError(f"Export file not found: {path.name}")
    if not include_report:
        return ExportDownload(path=path)

    report = build_report(db, str(run.run_id))
    from train_platform.utils.training_report_docx import build_training_report_docx, build_training_report_filename

    report_filename = build_training_report_filename(report)
    report_content = build_training_report_docx(report)
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(path, arcname=path.name)
        archive.writestr(report_filename, report_content)
    filename = f"{run.run_id}_{weights_key}_{fmt}_with_report.zip"
    return ExportDownload(content=archive_buffer.getvalue(), filename=filename)


__all__ = ["ExportDownload", "TrainingExport", "download_export", "export_training_run"]
