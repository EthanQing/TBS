from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.schemas.v3.inference_jobs import InferenceJobCreate, InferenceJobOut, InferenceModelCandidate
from train_platform.services.v3.inference_job_service import InferenceJobService


router = APIRouter(prefix="/inference-jobs", tags=["inference-jobs"])
_svc = InferenceJobService()


def _parse_cursor(raw: Any) -> int:
    try:
        if raw is None:
            return 0
        v = int(str(raw).strip())
        return max(0, v)
    except Exception:
        return 0


@router.get("/models", response_model=list[InferenceModelCandidate])
def list_inference_models(project_id: int | None = Query(None), db: Session = Depends(get_db)):
    return _svc.list_inferable_models(db, project_id=project_id)


@router.post("", response_model=InferenceJobOut, status_code=201)
def create_inference_job(payload: InferenceJobCreate, db: Session = Depends(get_db)):
    return _svc.create_job(db, payload)


@router.get("/{job_id}", response_model=InferenceJobOut)
def get_inference_job(job_id: str, include_items: bool = Query(True)):
    return _svc.get_job(job_id, include_items=bool(include_items))


@router.post("/{job_id}/cancel", response_model=InferenceJobOut)
def cancel_inference_job(job_id: str):
    return _svc.cancel_job(job_id)


@router.websocket("/{job_id}/stream")
async def stream_inference_job(websocket: WebSocket, job_id: str):
    """
    Realtime-ish stream via status/result file polling.

    Message types:
      - snapshot: current job summary
      - progress: status/progress updates
      - item: per-image incremental result
      - done: terminal status
      - error: stream/runtime error
      - ping: heartbeat
    """
    await websocket.accept()

    last_seq = _parse_cursor(websocket.query_params.get("from_seq"))
    last_result_id = _parse_cursor(websocket.query_params.get("from_result_id"))
    ping_every_s = 15.0
    last_ping = time.monotonic()

    try:
        try:
            snap = _svc.get_job(job_id, include_items=False)
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
                snap = _svc.get_job(job_id, include_items=False)
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
                            "status": payload.get("status"),
                            "phase": payload.get("phase"),
                            "mode": payload.get("mode"),
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

            rows = _svc.read_results_since(job_id, after_result_id=last_result_id)
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
