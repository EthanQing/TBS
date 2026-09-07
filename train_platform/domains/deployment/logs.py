from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy.orm import Session

from train_platform.models.v3.deployment import Deployment, DeploymentLog
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.models.v3.enums import LogLevel
from train_platform.utils.exceptions import NotFoundError


def append_deployment_log(
    db: Session,
    deployment_id: int,
    *,
    level: LogLevel,
    message: str,
    data: dict[str, Any] | None = None,
) -> DeploymentLog:
    deployment = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
    if not deployment:
        raise NotFoundError("Deployment not found")
    row = DeploymentLog(
        deployment_id=int(deployment_id),
        level=level,
        message=str(message),
        data=data,
    )
    db.add(row)
    db.flush()
    return row


def append_run_log(
    db: Session,
    run_id: str,
    *,
    level: LogLevel,
    message: str,
    step_key: str | None = None,
    action: str,
    detail: dict[str, Any] | None = None,
) -> DeploymentLog:
    run = (
        db.query(DeploymentRun)
        .filter(DeploymentRun.run_id == str(run_id))
        .with_for_update()
        .first()
    )
    if not run:
        raise NotFoundError("Deployment run not found")
    snapshot = deepcopy(run.snapshot) if isinstance(run.snapshot, dict) else {}
    sequence = int(snapshot.get("last_seq") or 0) + 1
    snapshot["last_seq"] = sequence
    run.snapshot = snapshot
    row = DeploymentLog(
        deployment_id=int(run.deployment_id),
        level=level,
        message=str(message),
        data={
            "run_id": str(run.run_id),
            "seq": sequence,
            "step_key": str(step_key or ""),
            "action": str(action),
            "detail": detail or {},
        },
    )
    db.add(row)
    db.flush()
    return row


def list_deployment_logs(db: Session, deployment_id: int, *, limit: int = 200) -> list[DeploymentLog]:
    deployment = db.query(Deployment).filter(Deployment.deployment_id == int(deployment_id)).first()
    if not deployment:
        raise NotFoundError("Deployment not found")
    return (
        db.query(DeploymentLog)
        .filter(DeploymentLog.deployment_id == int(deployment_id))
        .order_by(DeploymentLog.created_at.desc(), DeploymentLog.log_id.desc())
        .limit(max(1, min(int(limit), 5000)))
        .all()
    )


def list_run_logs(
    db: Session,
    run_id: str,
    *,
    after_seq: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    run = (
        db.query(DeploymentRun)
        .filter(DeploymentRun.run_id == str(run_id))
        .first()
    )
    if not run:
        raise NotFoundError("Deployment run not found")
    rows = (
        db.query(DeploymentLog)
        .filter(
            DeploymentLog.deployment_id == int(run.deployment_id),
            DeploymentLog.data["run_id"].as_string() == str(run.run_id),
            DeploymentLog.data["seq"].as_integer() > int(after_seq),
        )
        .order_by(DeploymentLog.log_id.asc())
        .limit(max(1, min(int(limit), 5000)))
        .all()
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        data = row.data if isinstance(row.data, dict) else {}
        sequence = int(data.get("seq") or 0)
        if sequence <= int(after_seq):
            continue
        output.append(
            {
                "seq": sequence,
                "log_id": int(row.log_id),
                "level": row.level.value if hasattr(row.level, "value") else row.level,
                "message": row.message,
                "created_at": row.created_at,
                "step_key": data.get("step_key"),
                "action": data.get("action"),
                "detail": data.get("detail"),
            }
        )
        if len(output) >= int(limit):
            break
    return output


def rollback_log_data(
    *,
    project_id: int,
    from_model_version_id: int,
    to_model_version_id: int,
    from_version: str,
    to_version: str,
    reason: str,
    operator: str,
    at: str,
) -> dict[str, Any]:
    return {
        "action": "rollback",
        "project_id": int(project_id),
        "from_model_version_id": int(from_model_version_id),
        "to_model_version_id": int(to_model_version_id),
        "from_version": str(from_version),
        "to_version": str(to_version),
        "reason": str(reason),
        "operator": str(operator),
        "stage_sync": True,
        "at": str(at),
    }


def map_rollback_log(row: DeploymentLog) -> dict[str, Any] | None:
    data = row.data if isinstance(row.data, dict) else {}
    if str(data.get("action") or "").strip().lower() != "rollback":
        return None
    return {
        "log_id": int(row.log_id),
        "deployment_id": int(row.deployment_id),
        "created_at": row.created_at,
        "operator": data.get("operator"),
        "reason": data.get("reason"),
        "from_model_version_id": data.get("from_model_version_id"),
        "to_model_version_id": data.get("to_model_version_id"),
        "from_version": data.get("from_version"),
        "to_version": data.get("to_version"),
    }


__all__ = [
    "append_deployment_log",
    "append_run_log",
    "list_deployment_logs",
    "list_run_logs",
    "map_rollback_log",
    "rollback_log_data",
]
