from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.db.session import SessionLocal
from train_platform.schemas.v2.deployments import (
    DeploymentExecuteCreate,
    DeploymentRunOut,
    DeploymentRunRetryOut,
    DeploymentRunWsMessage,
)
from train_platform.services.deployment_runtime_service import DeploymentRuntimeService


router = APIRouter(prefix="/deployment-runs", tags=["deployment-runs"])
_svc = DeploymentRuntimeService()


def _parse_cursor(raw: Any) -> int:
    try:
        if raw is None:
            return 0
        v = int(str(raw).strip())
        return max(0, v)
    except Exception:
        return 0


@router.get("/{run_id}", response_model=DeploymentRunOut)
def get_deployment_run(run_id: str, db: Session = Depends(get_db)):
    return _svc.get_run(db, run_id)


@router.post("/{run_id}/retry", response_model=DeploymentRunRetryOut)
def retry_deployment_run(
    run_id: str,
    payload: DeploymentExecuteCreate | None = None,
    db: Session = Depends(get_db),
):
    body = payload.model_dump() if payload else {}
    return _svc.retry_run(db, run_id, payload=body)


@router.post("/{run_id}/cancel", response_model=DeploymentRunOut)
def cancel_deployment_run(run_id: str, db: Session = Depends(get_db)):
    return _svc.cancel_run(db, run_id)


@router.get("/{run_id}/logs", response_model=list[dict[str, Any]])
def list_deployment_run_logs(
    run_id: str,
    from_seq: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    return _svc.list_logs_since(db, run_id, after_seq=int(from_seq), limit=int(limit))


@router.websocket("/{run_id}/stream")
async def stream_deployment_run(websocket: WebSocket, run_id: str):
    await websocket.accept()
    last_seq = _parse_cursor(websocket.query_params.get("from_seq"))
    ping_every_s = 15.0
    last_ping = time.monotonic()
    last_state = None

    try:
        with SessionLocal() as db:
            run = _svc.get_run(db, run_id)
            payload = DeploymentRunOut.model_validate(run).model_dump(mode="json")
            await websocket.send_json(DeploymentRunWsMessage(type="snapshot", data=payload).model_dump(mode="json"))

        while True:
            with SessionLocal() as db:
                run = _svc.get_run(db, run_id)
                payload = DeploymentRunOut.model_validate(run).model_dump(mode="json")

                state = (
                    payload.get("status"),
                    payload.get("phase"),
                    payload.get("progress"),
                    payload.get("current_step"),
                    payload.get("error_message"),
                    payload.get("finished_at"),
                    payload.get("updated_at"),
                )
                if state != last_state:
                    await websocket.send_json(
                        DeploymentRunWsMessage(
                            type="progress",
                            data={
                                "run_id": payload.get("run_id"),
                                "status": payload.get("status"),
                                "phase": payload.get("phase"),
                                "progress": payload.get("progress"),
                                "current_step": payload.get("current_step"),
                                "cancel_requested": payload.get("cancel_requested"),
                                "error_message": payload.get("error_message"),
                                "snapshot": payload.get("snapshot"),
                            },
                        ).model_dump(mode="json")
                    )
                    last_state = state

                rows = _svc.list_logs_since(db, run_id, after_seq=last_seq, limit=1000)
                for row in rows:
                    seq = int(row.get("seq") or 0)
                    if seq <= last_seq:
                        continue
                    await websocket.send_json(DeploymentRunWsMessage(type="log", data=row).model_dump(mode="json"))
                    last_seq = seq

                if str(payload.get("status") or "").lower() in {"completed", "failed", "cancelled"}:
                    await websocket.send_json(
                        DeploymentRunWsMessage(
                            type="done",
                            data={
                                "status": payload.get("status"),
                                "phase": payload.get("phase"),
                                "progress": payload.get("progress"),
                                "error_message": payload.get("error_message"),
                            },
                        ).model_dump(mode="json")
                    )
                    await websocket.close()
                    return

            now = time.monotonic()
            if now - last_ping >= ping_every_s:
                await websocket.send_json(DeploymentRunWsMessage(type="ping", data={}).model_dump(mode="json"))
                last_ping = now
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json(
                DeploymentRunWsMessage(type="error", data={"message": f"{type(e).__name__}: {e}"}).model_dump(mode="json")
            )
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
