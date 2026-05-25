from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from train_platform.core.config import settings
from train_platform.schemas.v3.model_conversions import ModelConversionOut
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.model_conversion_jobs import _append_log, _job_dir, _read_status, _utcnow, _write_status


router = APIRouter(prefix="/model-conversions", tags=["model-conversions"])


@router.post("", response_model=ModelConversionOut, status_code=201)
async def create_model_conversion(
    file: UploadFile = File(...),
    source_format: str = Form("pt"),
    target_format: str = Form("onnx"),
    opset: int | None = Form(None),
    dynamic: bool = Form(True),
):
    """
    Convert a YOLOv8 PyTorch weight file (.pt/.pth) to ONNX.

    This endpoint is async from the client's POV:
    - returns a job_id immediately
    - client polls GET /model-conversions/{job_id}
    """
    sf = str(source_format or "").strip().lower()
    tf = str(target_format or "").strip().lower()
    if sf not in ("pt", "pth"):
        raise ValidationError("Only pt/pth is supported for now (YOLOv8)")
    if tf != "onnx":
        raise ValidationError("Only onnx is supported for now (YOLOv8)")

    filename = file.filename or "model.pt"
    suffix = Path(filename).suffix.lower() or ".pt"
    if suffix not in (".pt", ".pth"):
        raise ValidationError("Unsupported source model file type")

    job_id = uuid.uuid4().hex
    root = _job_dir(job_id)

    input_path = (root / "input.pt").resolve(strict=False)
    if settings.temp_dir.resolve() not in input_path.parents:
        raise ValidationError("Unsafe upload path")

    # Persist upload
    try:
        with open(input_path, "wb") as f:
            import shutil

            shutil.copyfileobj(file.file, f)
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    created_at = _utcnow().isoformat()
    status: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "logs": [f"已接收文件: {filename}", f"source_format={sf} target_format={tf}"],
        "source_format": sf,
        "target_format": tf,
        "opset": int(opset) if opset is not None else None,
        "dynamic": bool(dynamic),
        "worker_id": None,
        "output_url": None,
        "output_filename": None,
        "error_message": None,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _append_log(status, "已加入 YOLO worker 转换队列")
    _write_status(job_id, status)

    return ModelConversionOut.model_validate(status)


@router.get("/{job_id}", response_model=ModelConversionOut)
def get_model_conversion(job_id: str):
    data = _read_status(job_id)
    return ModelConversionOut.model_validate(data)


@router.get("/{job_id}/download")
def download_model_conversion(job_id: str):
    data = _read_status(job_id)
    if str(data.get("status") or "").strip().lower() != "completed":
        raise ValidationError("Conversion is not completed")

    root = _job_dir(job_id)
    filename = str(data.get("output_filename") or "output.onnx").strip() or "output.onnx"
    out_path = (root / filename).resolve(strict=False)
    if settings.temp_dir.resolve() not in out_path.parents:
        raise ValidationError("Unsafe conversion output path")
    if not out_path.exists() or not out_path.is_file():
        raise ValidationError("Conversion output file not found")
    return FileResponse(
        path=str(out_path),
        filename=filename,
        media_type="application/octet-stream",
    )
