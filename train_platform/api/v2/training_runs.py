from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Query
from fastapi import WebSocket, WebSocketDisconnect
import requests
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.models.dataset import DatasetVersion
from train_platform.models.enums import TrainingRunStatus
from train_platform.models.training_run import TrainingRun, TrainingRunArtifact, TrainingRunEpochMetric, TrainingRunEvent
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.training_runs import (
    TrainingRunArtifactOut,
    TrainingRunCompareRequest,
    TrainingRunCompareResponse,
    TrainingRunEpochMetricOut,
    TrainingRunEventOut,
    TrainingRunExportOut,
    TrainingRunExportRequest,
    TrainingRunLogTailOut,
    TrainingRunCreate,
    TrainingRunMetaOut,
    TrainingRunMetaUpdate,
    TrainingRunOut,
    TrainingRunUpdate,
)
from train_platform.services.training_run_service import TrainingRunService
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.mlflow_utils import fetch_mlflow_epoch_metrics


router = APIRouter(prefix="/training-runs", tags=["training-runs"])


def _export_onnx_via_worker(
    src_pt: Path,
    out_onnx: Path,
    *,
    dynamic: bool,
    opset: int | None,
    imgsz: int | None,
) -> None:
    worker_url = os.getenv("INFERENCE_WORKER_URL", "http://inference-worker:18002").rstrip("/")
    timeout = float(os.getenv("INFERENCE_WORKER_TIMEOUT", "1200"))
    payload = {
        "src_pt": str(src_pt),
        "out_onnx": str(out_onnx),
        "dynamic": bool(dynamic),
        "opset": int(opset) if opset is not None else None,
        "imgsz": int(imgsz) if imgsz is not None else None,
    }

    try:
        resp = requests.post(f"{worker_url}/internal/training-runs/export-onnx", json=payload, timeout=timeout)
    except Exception as e:
        raise ValidationError(f"Failed to reach inference worker: {e}") from e

    if resp.status_code != 200:
        raise ValidationError(f"Inference worker error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception as e:
        raise ValidationError(f"Inference worker returned non-JSON response: {e}") from e

    err = data.get("error")
    if err:
        raise ValidationError(err)


@router.get("", response_model=Page[TrainingRunOut])
def list_training_runs(
    page: int = 1,
    page_size: int = 50,
    project_id: int | None = Query(None),
    dataset_id: int | None = Query(None),
    architecture_id: int | None = Query(None),
    status: str | None = Query(None, description="created/queued/running/completed/failed/cancelled/deleted"),
    include_hidden: bool = Query(False),
    db: Session = Depends(get_db),
):
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 500)
    skip = (page - 1) * page_size

    st = None
    if status:
        try:
            st = TrainingRunStatus(str(status))
        except Exception:
            raise ValidationError("Invalid status")

    q = db.query(TrainingRun)
    if not include_hidden:
        q = q.filter(TrainingRun.hidden == False)  # noqa: E712
    if project_id is not None:
        q = q.filter(TrainingRun.project_id == int(project_id))
    if dataset_id is not None:
        q = q.join(TrainingRun.dataset_version).filter(DatasetVersion.dataset_id == int(dataset_id))
    if architecture_id is not None:
        q = q.filter(TrainingRun.architecture_id == int(architecture_id))
    if st is not None:
        q = q.filter(TrainingRun.status == st)
    total = q.count()

    items = TrainingRunService().list_runs(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        architecture_id=architecture_id,
        status=st,
        skip=skip,
        limit=page_size,
        include_hidden=include_hidden,
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=TrainingRunOut, status_code=201)
def create_training_run(payload: TrainingRunCreate, db: Session = Depends(get_db)):
    return TrainingRunService().create_run(db, obj=payload.model_dump())


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_training_run(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().get_run(db, run_id)


@router.patch("/{run_id}", response_model=TrainingRunOut)
def update_training_run(run_id: str, payload: TrainingRunUpdate, db: Session = Depends(get_db)):
    return TrainingRunService().update_run(db, run_id, patch=payload.model_dump(exclude_unset=True))


@router.post("/{run_id}/queue", response_model=TrainingRunOut)
def queue_training_run(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().queue_run(db, run_id)


@router.post("/{run_id}/cancel", response_model=TrainingRunOut)
def cancel_training_run(run_id: str, reason: str | None = Body(None), db: Session = Depends(get_db)):
    return TrainingRunService().request_cancel(db, run_id, reason=reason)


@router.delete("/{run_id}", response_model=TrainingRunOut)
def delete_training_run(
    run_id: str,
    force: bool = Query(False, description="Delete training run and related model versions/deployments"),
    db: Session = Depends(get_db),
):
    return TrainingRunService().delete_run(db, run_id, force=bool(force))


@router.get("/{run_id}/events", response_model=list[TrainingRunEventOut])
def list_training_run_events(run_id: str, limit: int = Query(200, ge=1, le=5000), db: Session = Depends(get_db)):
    return TrainingRunService().list_events(db, run_id, limit=limit)


@router.get("/{run_id}/metrics/epochs", response_model=list[TrainingRunEpochMetricOut])
def list_training_run_epoch_metrics(
    run_id: str,
    limit: int = Query(5000, ge=1, le=100000),
    source: str | None = Query(None, description="auto|db|mlflow"),
    db: Session = Depends(get_db),
):
    source_norm = str(source or "auto").strip().lower()
    if source_norm in ("auto", "mlflow"):
        rows = fetch_mlflow_epoch_metrics(db, run_id, limit=int(limit))
        if rows is not None:
            return rows
    return TrainingRunService().list_epoch_metrics(db, run_id, limit=limit)


@router.get("/{run_id}/artifacts", response_model=list[TrainingRunArtifactOut])
def list_training_run_artifacts(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().list_artifacts(db, run_id)


@router.post("/{run_id}/export", response_model=TrainingRunExportOut)
def export_training_run(run_id: str, payload: TrainingRunExportRequest, db: Session = Depends(get_db)):
    """
    Export a training run to a deployable format.

    Currently supported:
    - pt: raw weights download (best/last)
    - onnx: Ultralytics export (YOLOv8 -> ONNX)
    """
    run = TrainingRunService().get_run(db, run_id)

    fmt = str(payload.format or "pt").strip().lower()
    weights = str(payload.weights or "best").strip().lower()
    if fmt not in ("pt", "onnx"):
        raise ValidationError("Unsupported export format")
    if weights not in ("best", "last"):
        raise ValidationError("weights must be 'best' or 'last'")

    weights_dir = (settings.training_dir / str(run.run_id) / "weights").resolve(strict=False)
    src_pt = (weights_dir / ("best.pt" if weights == "best" else "last.pt")).resolve(strict=False)
    if settings.training_dir.resolve() not in src_pt.parents:
        raise ValidationError("Unsafe weights path")
    if not src_pt.exists():
        raise ValidationError(f"Weights not found: {src_pt.name}")

    if fmt == "pt":
        rel = src_pt.relative_to(settings.training_dir).as_posix()
        url = f"/static/training/{rel}"
        return TrainingRunExportOut(run_id=str(run.run_id), format=fmt, weights=weights, download_url=url, artifact=None)

    # fmt == onnx
    out_name = "best.onnx" if weights == "best" else "last.onnx"
    out_onnx = (weights_dir / out_name).resolve(strict=False)
    if settings.training_dir.resolve() not in out_onnx.parents:
        raise ValidationError("Unsafe output path")

    if not out_onnx.exists():
        _export_onnx_via_worker(
            src_pt,
            out_onnx,
            dynamic=bool(payload.dynamic),
            opset=payload.opset,
            imgsz=payload.imgsz,
        )

        exported = None

        # Ultralytics may return a path; ensure the canonical output exists at out_onnx.
        exported_path: Path | None = None
        try:
            if exported:
                exported_path = Path(str(exported)).resolve(strict=False)
        except Exception:
            exported_path = None

        if not out_onnx.exists():
            newest: Path | None = None
            run_root = (settings.training_dir / str(run.run_id)).resolve(strict=False)
            try:
                # Prefer weights/*.onnx, fallback to any *.onnx under the run folder.
                candidates = list(weights_dir.glob("*.onnx"))
                if not candidates:
                    candidates = list(run_root.rglob("*.onnx"))
                for cand in candidates:
                    if newest is None or cand.stat().st_mtime > newest.stat().st_mtime:
                        newest = cand
            except Exception:
                newest = None

            if exported_path and exported_path.exists():
                newest = exported_path

            if newest and newest.exists() and newest != out_onnx:
                try:
                    newest.replace(out_onnx)
                except Exception:
                    import shutil

                    shutil.copy2(newest, out_onnx)

        if not out_onnx.exists():
            raise ValidationError("ONNX export failed: output file not found")

    # Upsert an artifact row so UI can list/download it later.
    rel = out_onnx.relative_to(settings.training_dir).as_posix()
    db.query(TrainingRunArtifact).filter(
        TrainingRunArtifact.run_id == str(run.run_id),
        TrainingRunArtifact.kind == "export",
        TrainingRunArtifact.name == out_name,
    ).delete()

    size_bytes = None
    try:
        size_bytes = int(out_onnx.stat().st_size)
    except Exception:
        size_bytes = None

    art = TrainingRunArtifact(run_id=str(run.run_id), kind="export", name=out_name, path=rel, size_bytes=size_bytes)
    db.add(art)
    db.commit()
    db.refresh(art)

    url = f"/static/training/{rel}"
    return TrainingRunExportOut(
        run_id=str(run.run_id),
        format=fmt,
        weights=weights,
        download_url=url,
        artifact=TrainingRunArtifactOut.model_validate(art),
    )


@router.post("/compare", response_model=TrainingRunCompareResponse)
def compare_training_runs(payload: TrainingRunCompareRequest, db: Session = Depends(get_db)):
    return TrainingRunService().compare_runs(db, payload.run_ids)


@router.get("/{run_id}/meta", response_model=TrainingRunMetaOut)
def get_training_run_meta(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().get_meta(db, run_id)


@router.patch("/{run_id}/meta", response_model=TrainingRunMetaOut)
def update_training_run_meta(run_id: str, payload: TrainingRunMetaUpdate, db: Session = Depends(get_db)):
    return TrainingRunService().update_meta(db, run_id, patch=payload.model_dump(exclude_unset=True))


@router.get("/{run_id}/logs/tail", response_model=TrainingRunLogTailOut)
def tail_training_run_logs(
    run_id: str,
    which: str = Query("stdout", description="stdout|stderr"),
    lines: int = Query(200, ge=1, le=20000),
    db: Session = Depends(get_db),
):
    which_norm = str(which or "").strip().lower()
    text = TrainingRunService().tail_logs(db, run_id, which=which_norm, lines=lines)
    return TrainingRunLogTailOut(run_id=str(run_id), which=which_norm, lines=int(lines), text=text)


@router.websocket("/{run_id}/logs/stream")
async def stream_training_run_logs(websocket: WebSocket, run_id: str):
    """
    WebSocket: stream worker stdout/stderr logs by tailing the log files.

    Path: /api/v2/training-runs/{run_id}/logs/stream?which=stdout|stderr|both&tail=200
    """

    def _read_new_lines(path: Path, pos: int, carry: str) -> tuple[int, str, list[str]]:
        try:
            if not path.exists() or not path.is_file():
                return pos, carry, []

            size = int(path.stat().st_size)
            if pos < 0 or size < pos:
                pos = 0
                carry = ""

            with open(path, "rb") as f:
                f.seek(int(pos))
                chunk = f.read()

            if not chunk:
                return pos, carry, []

            pos = int(pos) + int(len(chunk))
            text = chunk.decode("utf-8", errors="replace")
            text = (carry or "") + text

            # Keep the last partial line (if any) in carry to avoid flicker.
            if text.endswith("\n") or text.endswith("\r"):
                return pos, "", text.splitlines()

            parts = text.splitlines()
            if not parts:
                return pos, text, []
            carry = parts.pop()
            return pos, carry, parts
        except Exception:
            return pos, carry, []

    async def _send_lines(which: str, mode: str, lines: list[str]) -> None:
        if not lines:
            return
        await websocket.send_json({"type": "log", "data": {"which": which, "mode": mode, "lines": lines}})

    await websocket.accept()

    which = str(websocket.query_params.get("which") or "stdout").strip().lower()
    tail_raw = websocket.query_params.get("tail")
    try:
        tail_lines = int(tail_raw) if tail_raw is not None else 200
    except Exception:
        tail_lines = 200
    tail_lines = max(0, min(int(tail_lines), 5000))

    want_stdout = which in ("stdout", "both", "all")
    want_stderr = which in ("stderr", "both", "all")
    if not (want_stdout or want_stderr):
        want_stdout = True

    stdout_path = (settings.training_dir / str(run_id) / "logs" / "train.stdout.log").resolve(strict=False)
    stderr_path = (settings.training_dir / str(run_id) / "logs" / "train.stderr.log").resolve(strict=False)

    pos_stdout = 0
    pos_stderr = 0
    carry_stdout = ""
    carry_stderr = ""

    try:
        # Validate run exists & send an initial tail for context.
        with SessionLocal() as db:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
            if not run:
                await websocket.send_json({"type": "error", "data": {"message": "run not found"}})
                await websocket.close(code=1008)
                return

            if tail_lines > 0:
                svc = TrainingRunService()
                if want_stdout:
                    text = svc.tail_logs(db, str(run_id), which="stdout", lines=int(tail_lines))
                    await _send_lines("stdout", "tail", (text or "").splitlines())
                if want_stderr:
                    text = svc.tail_logs(db, str(run_id), which="stderr", lines=int(tail_lines))
                    await _send_lines("stderr", "tail", (text or "").splitlines())

        # Start streaming from the end (tail already sent).
        try:
            if want_stdout and stdout_path.exists():
                pos_stdout = int(stdout_path.stat().st_size)
        except Exception:
            pos_stdout = 0
        try:
            if want_stderr and stderr_path.exists():
                pos_stderr = int(stderr_path.stat().st_size)
        except Exception:
            pos_stderr = 0

        while True:
            if want_stdout:
                pos_stdout, carry_stdout, lines = _read_new_lines(stdout_path, pos_stdout, carry_stdout)
                if lines:
                    await _send_lines("stdout", "append", lines)
            if want_stderr:
                pos_stderr, carry_stderr, lines = _read_new_lines(stderr_path, pos_stderr, carry_stderr)
                if lines:
                    await _send_lines("stderr", "append", lines)

            with SessionLocal() as db:
                run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
                if not run:
                    await websocket.send_json({"type": "error", "data": {"message": "run not found"}})
                    await websocket.close(code=1008)
                    return

                if run.status in (
                    TrainingRunStatus.COMPLETED,
                    TrainingRunStatus.FAILED,
                    TrainingRunStatus.CANCELLED,
                    TrainingRunStatus.DELETED,
                ):
                    # Best-effort: flush remaining lines once before closing.
                    if want_stdout:
                        pos_stdout, carry_stdout, lines = _read_new_lines(stdout_path, pos_stdout, carry_stdout)
                        if lines:
                            await _send_lines("stdout", "append", lines)
                    if want_stderr:
                        pos_stderr, carry_stderr, lines = _read_new_lines(stderr_path, pos_stderr, carry_stderr)
                        if lines:
                            await _send_lines("stderr", "append", lines)

                    await websocket.send_json({"type": "done", "data": {"status": getattr(run.status, "value", run.status)}})
                    await asyncio.sleep(0.5)
                    await websocket.close()
                    return

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": {"message": f"{type(e).__name__}: {e}"}})
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@router.websocket("/{run_id}/metrics/stream")
async def stream_training_run_metrics(websocket: WebSocket, run_id: str):
    """
    WebSocket: realtime-ish stream via DB polling (simple, no broker needed).

    Path: /api/v2/training-runs/{run_id}/metrics/stream
    """
    await websocket.accept()

    last_metric_id = 0
    last_event_id = 0
    last_status = None
    last_progress = None
    last_epoch = None

    try:
        while True:
            with SessionLocal() as db:
                run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
                if not run:
                    await websocket.send_json({"type": "error", "data": {"message": "run not found"}})
                    await websocket.close(code=1008)
                    return

                # Push status/progress changes.
                if (
                    last_status != getattr(run.status, "value", run.status)
                    or last_progress != int(getattr(run, "progress", 0) or 0)
                    or last_epoch != int(getattr(run, "current_epoch", 0) or 0)
                ):
                    last_status = getattr(run.status, "value", run.status)
                    last_progress = int(getattr(run, "progress", 0) or 0)
                    last_epoch = int(getattr(run, "current_epoch", 0) or 0)
                    await websocket.send_json(
                        {
                            "type": "status",
                            "data": {
                                "run_id": str(run.run_id),
                                "status": last_status,
                                "progress": last_progress,
                                "current_epoch": last_epoch,
                                "total_epochs": int(getattr(run, "total_epochs", 0) or 0) or None,
                                "worker_id": getattr(run, "worker_id", None),
                            },
                        }
                    )

                # New epoch metrics
                metrics = (
                    db.query(TrainingRunEpochMetric)
                    .filter(TrainingRunEpochMetric.run_id == str(run_id), TrainingRunEpochMetric.metric_id > int(last_metric_id))
                    .order_by(TrainingRunEpochMetric.metric_id.asc())
                    .limit(200)
                    .all()
                )
                for m in metrics:
                    last_metric_id = max(int(last_metric_id), int(m.metric_id))
                    await websocket.send_json(
                        {
                            "type": "metric",
                            "data": {
                                "epoch": int(m.epoch),
                                "metrics": m.metrics,
                                "progress": int(getattr(run, "progress", 0) or 0),
                            },
                        }
                    )

                # New events
                events = (
                    db.query(TrainingRunEvent)
                    .filter(TrainingRunEvent.run_id == str(run_id), TrainingRunEvent.event_id > int(last_event_id))
                    .order_by(TrainingRunEvent.event_id.asc())
                    .limit(200)
                    .all()
                )
                for ev in events:
                    last_event_id = max(int(last_event_id), int(ev.event_id))
                    await websocket.send_json(
                        {
                            "type": "event",
                            "data": {
                                "event_id": int(ev.event_id),
                                "level": getattr(ev.level, "value", str(ev.level)),
                                "event_type": str(ev.event_type),
                                "message": ev.message,
                                "data": ev.data,
                                "created_at": ev.created_at.isoformat() if ev.created_at else None,
                            },
                        }
                    )

                # If the run reached a terminal status, keep the socket open for a short grace period
                # so clients can fetch final logs/metrics, then close.
                if run.status in (
                    TrainingRunStatus.COMPLETED,
                    TrainingRunStatus.FAILED,
                    TrainingRunStatus.CANCELLED,
                    TrainingRunStatus.DELETED,
                ):
                    # Flush any remaining metrics/events before closing.
                    metrics = (
                        db.query(TrainingRunEpochMetric)
                        .filter(TrainingRunEpochMetric.run_id == str(run_id), TrainingRunEpochMetric.metric_id > int(last_metric_id))
                        .order_by(TrainingRunEpochMetric.metric_id.asc())
                        .limit(500)
                        .all()
                    )
                    for m in metrics:
                        last_metric_id = max(int(last_metric_id), int(m.metric_id))
                        await websocket.send_json(
                            {
                                "type": "metric",
                                "data": {
                                    "epoch": int(m.epoch),
                                    "metrics": m.metrics,
                                    "progress": int(getattr(run, "progress", 0) or 0),
                                },
                            }
                        )

                    events = (
                        db.query(TrainingRunEvent)
                        .filter(TrainingRunEvent.run_id == str(run_id), TrainingRunEvent.event_id > int(last_event_id))
                        .order_by(TrainingRunEvent.event_id.asc())
                        .limit(200)
                        .all()
                    )
                    for ev in events:
                        last_event_id = max(int(last_event_id), int(ev.event_id))
                        await websocket.send_json(
                            {
                                "type": "event",
                                "data": {
                                    "event_id": int(ev.event_id),
                                    "level": getattr(ev.level, "value", str(ev.level)),
                                    "event_type": str(ev.event_type),
                                    "message": ev.message,
                                    "data": ev.data,
                                    "created_at": ev.created_at.isoformat() if ev.created_at else None,
                                },
                            }
                        )

                    await websocket.send_json({"type": "done", "data": {"status": getattr(run.status, "value", run.status)}})
                    await asyncio.sleep(0.5)
                    await websocket.close()
                    return

            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": {"message": f"{type(e).__name__}: {e}"}})
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
