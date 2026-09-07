from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from train_platform.api.deps import get_db
from train_platform.domains.deployment.credentials import verify_api_key
from train_platform.domains.deployment.service import assert_serving_ready, get_serving_deployment
from train_platform.domains.inference.input import save_uploaded_file
from train_platform.domains.inference.service import InferenceService
from train_platform.models.v3.deployment import Deployment
from train_platform.schemas.v3.deployments import (
    ServingInferResponse,
    ServingJobCreate,
    ServingJobOut,
)
from train_platform.utils.exceptions import ConflictError, ValidationError


router = APIRouter(prefix="/serving", tags=["serving"])
_infer = InferenceService()


def _extract_api_key(request: Request) -> str:
    header_key = str(request.headers.get("X-Deployment-Key") or "").strip()
    if header_key:
        return header_key
    auth = str(request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _assert_serving_ready(dep) -> None:
    try:
        assert_serving_ready(dep)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _assert_key(dep: Deployment, request: Request) -> None:
    raw = _extract_api_key(request)
    if not raw:
        raise HTTPException(status_code=401, detail="Missing deployment API key")
    if not verify_api_key(raw, str(dep.api_key_hash or "")):
        raise HTTPException(status_code=401, detail="Invalid deployment API key")


def _normalize_output(raw: dict[str, Any] | None) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    output = raw if isinstance(raw, dict) else {}
    preds = output.get("predictions")
    names = output.get("names")
    pred_list = preds if isinstance(preds, list) else []
    name_map = names if isinstance(names, dict) else None
    return pred_list, name_map


@router.get("/deployments/{deployment_id}/health")
def serving_health(
    deployment_id: int,
    request: Request,
    auth_required: bool = True,
    db: Session = Depends(get_db),
):
    dep = get_serving_deployment(db, int(deployment_id))
    _assert_serving_ready(dep)
    if auth_required:
        _assert_key(dep, request)
    return {
        "ok": True,
        "deployment_id": int(dep.deployment_id),
        "model_version_id": int(dep.model_version_id),
        "endpoint_url": dep.endpoint_url,
        "status": dep.status.value if hasattr(dep.status, "value") else dep.status,
    }


@router.post("/deployments/{deployment_id}/infer", response_model=ServingInferResponse)
async def serving_infer(
    deployment_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    dep = get_serving_deployment(db, int(deployment_id))
    _assert_serving_ready(dep)
    _assert_key(dep, request)

    content_type = str(request.headers.get("content-type") or "").lower()
    image_url = None
    conf = 0.5
    iou = 0.45
    show_labels = True
    show_confidence = True
    input_path = None
    file = None

    if content_type.startswith("application/json"):
        body = await request.json()
        if not isinstance(body, dict):
            raise ValidationError("JSON body must be an object")
        image_url = str(body.get("image_url") or "").strip() or None
        conf = float(body.get("conf", conf))
        iou = float(body.get("iou", iou))
        show_labels = bool(body.get("show_labels", show_labels))
        show_confidence = bool(body.get("show_confidence", show_confidence))
    else:
        form = await request.form()
        maybe_url = str(form.get("image_url") or "").strip()
        image_url = maybe_url or None
        try:
            conf = float(form.get("conf", conf))
            iou = float(form.get("iou", iou))
        except Exception:
            raise ValidationError("conf/iou must be numeric")
        if "show_labels" in form:
            show_labels = str(form.get("show_labels")).strip().lower() in {"1", "true", "yes", "on"}
        if "show_confidence" in form:
            show_confidence = str(form.get("show_confidence")).strip().lower() in {"1", "true", "yes", "on"}
        up = form.get("file")
        if isinstance(up, UploadFile):
            file = up

    if file is not None:
        input_path = save_uploaded_file(
            file.filename or "",
            file.file,
            subdir="serving_uploads",
            validate_suffix=False,
        )

    if bool(input_path) == bool(image_url):
        raise ValidationError("Provide exactly one of file or image_url")

    out = _infer.run_inference_output(
        db,
        model_version_id=int(dep.model_version_id),
        input_path=input_path,
        image_url=image_url,
        conf=float(conf),
        iou=float(iou),
    )

    err = str(out.get("error_message") or "").strip()
    if err:
        raise HTTPException(status_code=500, detail=err)

    payload = out.get("output") if isinstance(out.get("output"), dict) else {}
    predictions, names = _normalize_output(payload)
    input_meta = dict(out.get("input_meta") or {})
    input_meta["show_labels"] = bool(show_labels)
    input_meta["show_confidence"] = bool(show_confidence)

    return ServingInferResponse(
        deployment_id=int(dep.deployment_id),
        model_version_id=int(dep.model_version_id),
        engine=str(out.get("engine") or "") or None,
        family=str(out.get("family") or "") or None,
        variant=str(out.get("variant") or "") or None,
        run_id=str(out.get("run_id") or "") or None,
        input_path=str(out.get("input_path") or "") or None,
        input_meta=input_meta,
        inference_time_ms=float(out.get("inference_time_ms") or 0.0),
        names=names,
        predictions=predictions,
        raw_output=payload,
    )


@router.post("/deployments/{deployment_id}/jobs", response_model=ServingJobOut)
def create_serving_job(deployment_id: int, payload: ServingJobCreate, db: Session = Depends(get_db)):
    _ = get_serving_deployment(db, int(deployment_id))
    return ServingJobOut(
        enabled=False,
        status="not_enabled",
        message="mode not enabled in V3",
        mode=str(payload.mode),
        job_id=None,
    )


@router.get("/jobs/{job_id}", response_model=ServingJobOut)
def get_serving_job(job_id: str):
    return ServingJobOut(
        enabled=False,
        status="not_enabled",
        message="mode not enabled in V3",
        mode=None,
        job_id=str(job_id),
    )
