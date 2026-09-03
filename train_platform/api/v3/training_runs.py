from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, Query, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.core.config import settings
from train_platform.db.session import SessionLocal
from train_platform.domains.training.runs import (
    FrameworkCompareConflict,
    TrainingRunBenchmarkService,
    TrainingRunService as TrainingRunDomainService,
    compare_runs,
    download_export,
    export_training_run as domain_export_training_run,
    get_meta,
    list_artifacts,
    list_epoch_metrics,
    list_events,
    mark_project_card_reviewed,
    tail_logs,
    update_meta,
)
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.training_run import TrainingRun, TrainingRunEpochMetric, TrainingRunEvent
from train_platform.schemas.v3.common import Page, PageMeta
from train_platform.schemas.v3.training_runs import (
    TrainingRunArtifactOut,
    TrainingRunBenchmarkInferenceRequest,
    TrainingRunBenchmarkInferenceResponse,
    TrainingRunCompareRequest,
    TrainingRunCompareResponse,
    TrainingAugmentationOptionsOut,
    TrainingLossWeightOptionsOut,
    TrainingRunEpochMetricOut,
    TrainingRunEventOut,
    TrainingRunExportOut,
    TrainingRunExportRequest,
    TrainingRunLogTailOut,
    TrainingRunCreate,
    TrainingRunMetaOut,
    TrainingRunMetaUpdate,
    TrainingRunOut,
    TrainingRunReviewOut,
    TrainingRunReviewRequest,
    TrainingRunUpdate,
)
from train_platform.services.v3.alarm_service import AlarmService
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.training_augmentations import get_training_augmentation_options
from train_platform.utils.training_loss_weights import get_training_loss_weight_options
from train_platform.utils.mlflow_utils import fetch_mlflow_epoch_metrics


router = APIRouter(prefix="/training-runs", tags=["training-runs"])
run_svc = TrainingRunDomainService()


def _evaluate_run_alarm(db: Session, run_id: str) -> None:
    AlarmService.try_evaluate_training_rules(db, run_ids=[str(run_id)])


def _training_export_download_url(run_id: str, fmt: str, weights: str, include_report: bool = False) -> str:
    url = f"/api/v3/training-runs/{run_id}/export/download?format={fmt}&weights={weights}"
    if include_report:
        url += "&include_report=1"
    return url


@router.get("", response_model=Page[TrainingRunOut])
def list_training_runs(
    page: int = 1,
    page_size: int = 50,
    project_id: int | None = Query(None),
    standard_dataset_id: int | None = Query(None),
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
    if standard_dataset_id is not None:
        q = q.filter(TrainingRun.standard_dataset_id == int(standard_dataset_id))
    if architecture_id is not None:
        q = q.filter(TrainingRun.architecture_id == int(architecture_id))
    if st is not None:
        q = q.filter(TrainingRun.status == st)
    total = q.count()

    items = run_svc.list_runs(
        db,
        project_id=project_id,
        standard_dataset_id=standard_dataset_id,
        architecture_id=architecture_id,
        status=st,
        skip=skip,
        limit=page_size,
        include_hidden=include_hidden,
    )
    return {"items": items, "meta": PageMeta(page=page, page_size=page_size, total=int(total))}


@router.post("", response_model=TrainingRunOut, status_code=201)
def create_training_run(payload: TrainingRunCreate, db: Session = Depends(get_db)):
    return run_svc.create_run(db, obj=payload.model_dump())


@router.get("/augmentation-options", response_model=TrainingAugmentationOptionsOut)
def get_training_augmentation_options_endpoint(
    architecture_id: int | None = Query(None),
    engine: str | None = Query(None),
    task_type: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if architecture_id is not None:
        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(architecture_id)).first()
        if not arch:
            raise NotFoundError("Architecture not found")
        engine = str(getattr(arch, "engine", "") or "")
        task_type = str(getattr(getattr(arch, "task_type", None), "value", getattr(arch, "task_type", "")) or "")
    return get_training_augmentation_options(
        engine=engine or "ultralytics-yolo",
        task_type=task_type or "detection",
    )


@router.get("/loss-weight-options", response_model=TrainingLossWeightOptionsOut)
def get_training_loss_weight_options_endpoint(
    architecture_id: int | None = Query(None),
    engine: str | None = Query(None),
    task_type: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if architecture_id is not None:
        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(architecture_id)).first()
        if not arch:
            raise NotFoundError("Architecture not found")
        engine = str(getattr(arch, "engine", "") or "")
        task_type = str(getattr(getattr(arch, "task_type", None), "value", getattr(arch, "task_type", "")) or "")
    return get_training_loss_weight_options(
        engine=engine or "ultralytics-yolo",
        task_type=task_type or "detection",
    )


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_training_run(run_id: str, db: Session = Depends(get_db)):
    return run_svc.get_run(db, run_id)


@router.patch("/{run_id}", response_model=TrainingRunOut)
def update_training_run(run_id: str, payload: TrainingRunUpdate, db: Session = Depends(get_db)):
    return run_svc.update_run(db, run_id, patch=payload.model_dump(exclude_unset=True))


@router.post("/{run_id}/queue", response_model=TrainingRunOut)
def queue_training_run(run_id: str, db: Session = Depends(get_db)):
    run = run_svc.queue_run(db, run_id)
    _evaluate_run_alarm(db, str(run.run_id))
    return run


@router.post("/{run_id}/resume", response_model=TrainingRunOut)
def resume_training_run(run_id: str, db: Session = Depends(get_db)):
    run = run_svc.resume_run(db, run_id)
    _evaluate_run_alarm(db, str(run.run_id))
    return run


@router.post("/{run_id}/cancel", response_model=TrainingRunOut)
def cancel_training_run(run_id: str, reason: str | None = Body(None), db: Session = Depends(get_db)):
    run = run_svc.request_cancel(db, run_id, reason=reason)
    _evaluate_run_alarm(db, str(run.run_id))
    return run


@router.post("/{run_id}/review", response_model=TrainingRunReviewOut)
def review_training_run(
    run_id: str,
    payload: TrainingRunReviewRequest | None = None,
    db: Session = Depends(get_db),
):
    return mark_project_card_reviewed(db, run_id, source=payload.source if payload else None)


@router.delete("/{run_id}", response_model=TrainingRunOut)
def delete_training_run(
    run_id: str,
    force: bool = Query(False, description="Delete training run and related model versions/deployments"),
    db: Session = Depends(get_db),
):
    run = run_svc.delete_run(db, run_id, force=bool(force))
    _evaluate_run_alarm(db, str(run.run_id))
    return run


@router.get("/{run_id}/events", response_model=list[TrainingRunEventOut])
def list_training_run_events(run_id: str, limit: int = Query(200, ge=1, le=5000), db: Session = Depends(get_db)):
    return list_events(db, run_id, limit=limit)


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
        if source_norm == "mlflow":
            return rows or []
        # auto: fallback to DB when MLflow has no usable points yet
        if rows:
            return rows
    return list_epoch_metrics(db, run_id, limit=limit)


@router.get("/{run_id}/artifacts", response_model=list[TrainingRunArtifactOut])
def list_training_run_artifacts(run_id: str, db: Session = Depends(get_db)):
    return list_artifacts(db, run_id)


@router.post("/{run_id}/export", response_model=TrainingRunExportOut)
def export_training_run(run_id: str, payload: TrainingRunExportRequest, db: Session = Depends(get_db)):
    exported = domain_export_training_run(
        db,
        run_id,
        format=payload.format,
        weights=payload.weights,
        dynamic=bool(payload.dynamic),
        opset=payload.opset,
        imgsz=payload.imgsz,
    )
    url = _training_export_download_url(
        str(exported.run_id),
        exported.format,
        exported.weights,
        include_report=bool(payload.include_report),
    )
    return TrainingRunExportOut(
        run_id=exported.run_id,
        format=exported.format,
        weights=exported.weights,
        download_url=url,
        artifact=TrainingRunArtifactOut.model_validate(exported.artifact) if exported.artifact else None,
    )


@router.get("/{run_id}/export/download")
def download_training_run_export(
    run_id: str,
    format: str = Query("pt", description="pt | onnx"),
    weights: str = Query("best", description="best | last"),
    include_report: bool = Query(False, description="package model export with DOCX training report"),
    db: Session = Depends(get_db),
):
    download = download_export(
        db,
        run_id,
        format=format,
        weights=weights,
        include_report=bool(include_report),
    )
    if download.content is not None:
        filename = str(download.filename or "training_export.zip")
        quoted = quote(filename)
        return Response(
            content=download.content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}; filename*=UTF-8''{quoted}",
            },
        )

    return FileResponse(
        path=str(download.path),
        filename=download.path.name if download.path else None,
        media_type="application/octet-stream",
    )


@router.post("/compare", response_model=TrainingRunCompareResponse)
def compare_training_runs(payload: TrainingRunCompareRequest, db: Session = Depends(get_db)):
    try:
        return compare_runs(db, payload.run_ids)
    except FrameworkCompareConflict as e:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(e),
                "framework_groups": e.framework_groups,
            },
        ) from e


@router.post("/benchmark-inference-time", response_model=TrainingRunBenchmarkInferenceResponse)
def benchmark_training_runs_inference_time(
    payload: TrainingRunBenchmarkInferenceRequest,
    db: Session = Depends(get_db),
):
    return TrainingRunBenchmarkService().benchmark_inference_times(
        db,
        run_ids=payload.run_ids,
        force=bool(payload.force),
    )


@router.get("/{run_id}/meta", response_model=TrainingRunMetaOut)
def get_training_run_meta(run_id: str, db: Session = Depends(get_db)):
    return get_meta(db, run_id)


@router.patch("/{run_id}/meta", response_model=TrainingRunMetaOut)
def update_training_run_meta(run_id: str, payload: TrainingRunMetaUpdate, db: Session = Depends(get_db)):
    return update_meta(db, run_id, patch=payload.model_dump(exclude_unset=True))


@router.get("/{run_id}/logs/tail", response_model=TrainingRunLogTailOut)
def tail_training_run_logs(
    run_id: str,
    which: str = Query("stdout", description="stdout|stderr"),
    lines: int = Query(200, ge=1, le=20000),
    db: Session = Depends(get_db),
):
    which_norm = str(which or "").strip().lower()
    text = tail_logs(db, run_id, which=which_norm, lines=lines)
    return TrainingRunLogTailOut(run_id=str(run_id), which=which_norm, lines=int(lines), text=text)


@router.websocket("/{run_id}/logs/stream")
async def stream_training_run_logs(websocket: WebSocket, run_id: str):
    """
    WebSocket: stream worker stdout/stderr logs by tailing the log files.

    Path: /api/v3/training-runs/{run_id}/logs/stream?which=stdout|stderr|both&tail=200
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
                if want_stdout:
                    text = tail_logs(db, str(run_id), which="stdout", lines=int(tail_lines))
                    await _send_lines("stdout", "tail", (text or "").splitlines())
                if want_stderr:
                    text = tail_logs(db, str(run_id), which="stderr", lines=int(tail_lines))
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

    Path: /api/v3/training-runs/{run_id}/metrics/stream
    """
    await websocket.accept()

    def _parse_cursor(raw: str | None) -> int:
        if raw is None:
            return 0
        try:
            return max(int(raw), 0)
        except Exception:
            return 0

    last_metric_id = _parse_cursor(websocket.query_params.get("from_metric_id"))
    last_event_id = _parse_cursor(websocket.query_params.get("from_event_id"))
    last_status = None
    last_progress = None
    last_epoch = None
    cursor_sent = False
    loop = asyncio.get_running_loop()
    next_ping_at = loop.time() + 15.0

    try:
        while True:
            with SessionLocal() as db:
                run = db.query(TrainingRun).filter(TrainingRun.run_id == str(run_id)).first()
                if not run:
                    await websocket.send_json({"type": "error", "data": {"message": "run not found"}})
                    await websocket.close(code=1008)
                    return

                if not cursor_sent:
                    latest_metric_id = (
                        db.query(func.max(TrainingRunEpochMetric.metric_id))
                        .filter(TrainingRunEpochMetric.run_id == str(run_id))
                        .scalar()
                        or 0
                    )
                    latest_event_id = (
                        db.query(func.max(TrainingRunEvent.event_id))
                        .filter(TrainingRunEvent.run_id == str(run_id))
                        .scalar()
                        or 0
                    )
                    await websocket.send_json(
                        {
                            "type": "cursor",
                            "data": {
                                "last_metric_id": int(last_metric_id),
                                "last_event_id": int(last_event_id),
                                "latest_metric_id": int(latest_metric_id),
                                "latest_event_id": int(latest_event_id),
                            },
                        }
                    )
                    cursor_sent = True

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
                                "metric_id": int(m.metric_id),
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
                                    "metric_id": int(m.metric_id),
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

            now = loop.time()
            if now >= next_ping_at:
                await websocket.send_json(
                    {
                        "type": "ping",
                        "data": {
                            "last_metric_id": int(last_metric_id),
                            "last_event_id": int(last_event_id),
                        },
                    }
                )
                next_ping_at = now + 15.0

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
