from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, Query
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.db.session import SessionLocal
from train_platform.models.dataset import DatasetVersion
from train_platform.models.enums import TrainingRunStatus
from train_platform.models.training_run import TrainingRun, TrainingRunEpochMetric, TrainingRunEvent
from train_platform.schemas.v2.common import DeleteResponse, Page, PageMeta
from train_platform.schemas.v2.training_runs import (
    TrainingRunArtifactOut,
    TrainingRunCompareRequest,
    TrainingRunCompareResponse,
    TrainingRunEpochMetricOut,
    TrainingRunEventOut,
    TrainingRunLogTailOut,
    TrainingRunCreate,
    TrainingRunMetaOut,
    TrainingRunMetaUpdate,
    TrainingRunOut,
    TrainingRunUpdate,
)
from train_platform.services.training_run_service import TrainingRunService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/training-runs", tags=["training-runs"])


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
def delete_training_run(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().request_delete(db, run_id)


@router.get("/{run_id}/events", response_model=list[TrainingRunEventOut])
def list_training_run_events(run_id: str, limit: int = Query(200, ge=1, le=5000), db: Session = Depends(get_db)):
    return TrainingRunService().list_events(db, run_id, limit=limit)


@router.get("/{run_id}/metrics/epochs", response_model=list[TrainingRunEpochMetricOut])
def list_training_run_epoch_metrics(run_id: str, limit: int = Query(5000, ge=1, le=100000), db: Session = Depends(get_db)):
    return TrainingRunService().list_epoch_metrics(db, run_id, limit=limit)


@router.get("/{run_id}/artifacts", response_model=list[TrainingRunArtifactOut])
def list_training_run_artifacts(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().list_artifacts(db, run_id)


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
    lines: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    which_norm = str(which or "").strip().lower()
    text = TrainingRunService().tail_logs(db, run_id, which=which_norm, lines=lines)
    return TrainingRunLogTailOut(run_id=str(run_id), which=which_norm, lines=int(lines), text=text)


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
