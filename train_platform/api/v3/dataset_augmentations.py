from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.schemas.v3.dataset_augmentations import (
    DatasetAugmentationCancelOut,
    DatasetAugmentationCreate,
    DatasetAugmentationJobOut,
    DatasetAugmentationPreviewOut,
    DatasetAugmentationPreviewRequest,
    DatasetAugmentationPublishIn,
    DatasetAugmentationPublishOut,
)
from train_platform.services.v3.dataset_augmentation_service import DatasetAugmentationService


router = APIRouter(prefix="/standard-datasets/{standard_dataset_id}/augmentations", tags=["dataset-augmentations"])
_svc = DatasetAugmentationService()


def _parse_cursor(raw: Any) -> int:
    try:
        if raw is None:
            return 0
        return max(0, int(str(raw).strip()))
    except Exception:
        return 0


@router.post("/preview", response_model=DatasetAugmentationPreviewOut)
def preview_dataset_augmentation(
    standard_dataset_id: int,
    payload: DatasetAugmentationPreviewRequest,
    db: Session = Depends(get_db),
):
    return _svc.preview(db, int(standard_dataset_id), payload)


@router.post("", response_model=DatasetAugmentationJobOut, status_code=201)
def create_dataset_augmentation(
    standard_dataset_id: int,
    payload: DatasetAugmentationCreate,
    db: Session = Depends(get_db),
):
    return _svc.create_job(db, int(standard_dataset_id), payload)


@router.get("/{job_id}", response_model=DatasetAugmentationJobOut)
def get_dataset_augmentation(standard_dataset_id: int, job_id: str):
    return _svc.get_job(int(standard_dataset_id), str(job_id))


@router.post("/{job_id}/cancel", response_model=DatasetAugmentationCancelOut)
def cancel_dataset_augmentation(standard_dataset_id: int, job_id: str):
    row = _svc.cancel_job(int(standard_dataset_id), str(job_id))
    return DatasetAugmentationCancelOut(
        job_id=str(row.job_id),
        status=row.status,
        cancel_requested=bool(row.cancel_requested),
    )


@router.post("/{job_id}/publish", response_model=DatasetAugmentationPublishOut)
def publish_dataset_augmentation(
    standard_dataset_id: int,
    job_id: str,
    payload: DatasetAugmentationPublishIn,
    db: Session = Depends(get_db),
):
    return _svc.publish_job(db, int(standard_dataset_id), str(job_id), payload)


@router.websocket("/{job_id}/stream")
async def stream_dataset_augmentation(websocket: WebSocket, standard_dataset_id: int, job_id: str):
    """
    Stream augmentation status/result updates via status/result file polling.

    Message types:
      - snapshot
      - progress
      - item
      - done
      - error
      - ping
    """
    await websocket.accept()

    did = int(standard_dataset_id)
    jid = str(job_id)
    last_seq = _parse_cursor(websocket.query_params.get("from_seq"))
    last_result_id = _parse_cursor(websocket.query_params.get("from_result_id"))
    ping_every_s = 15.0
    last_ping = time.monotonic()

    try:
        try:
            snap = _svc.get_job(did, jid)
        except Exception as e:
            await websocket.send_json({"type": "error", "data": {"message": f"{type(e).__name__}: {e}"}})
            await websocket.close(code=1008)
            return

        snap_payload = snap.model_dump(mode="json")
        await websocket.send_json({"type": "snapshot", "data": snap_payload})
        if last_seq <= 0:
            last_seq = int(snap_payload.get("seq") or 0)

        while True:
            try:
                snap = _svc.get_job(did, jid)
            except Exception as e:
                await websocket.send_json({"type": "error", "data": {"message": f"{type(e).__name__}: {e}"}})
                await websocket.close(code=1011)
                return

            payload = snap.model_dump(mode="json")
            seq = int(payload.get("seq") or 0)
            if seq > last_seq:
                await websocket.send_json(
                    {
                        "type": "progress",
                        "data": {
                            "job_id": payload.get("job_id"),
                            "standard_dataset_id": payload.get("standard_dataset_id"),
                            "status": payload.get("status"),
                            "phase": payload.get("phase"),
                            "progress": payload.get("progress"),
                            "processed": payload.get("processed"),
                            "total": payload.get("total"),
                            "seq": seq,
                            "last_result_id": payload.get("last_result_id"),
                            "error_message": payload.get("error_message"),
                            "result": payload.get("result"),
                        },
                    }
                )
                last_seq = seq

            rows = _svc.read_results_since(did, jid, after_result_id=last_result_id)
            for row in rows:
                rid = int(row.get("result_id") or 0)
                if rid <= last_result_id:
                    continue
                await websocket.send_json({"type": "item", "data": row})
                last_result_id = rid

            status = str(payload.get("status") or "").strip().lower()
            if status in {"completed", "failed", "cancelled"}:
                await websocket.send_json(
                    {
                        "type": "done",
                        "data": {
                            "status": payload.get("status"),
                            "phase": payload.get("phase"),
                            "error_message": payload.get("error_message"),
                            "result": payload.get("result"),
                        },
                    }
                )
                await websocket.close()
                return

            now = time.monotonic()
            if (now - last_ping) >= ping_every_s:
                await websocket.send_json({"type": "ping", "data": {}})
                last_ping = now

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
