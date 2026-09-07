from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from train_platform.core.license import assert_valid_license
from train_platform.domains.model_assets.conversion.jobs import (
    create_job,
    read_job,
    resolve_download_path,
)
from train_platform.schemas.v3.model_conversions import ModelConversionOut


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
    assert_valid_license()
    try:
        status = create_job(
            file.file,
            filename=file.filename,
            source_format=source_format,
            target_format=target_format,
            opset=opset,
            dynamic=dynamic,
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
    return ModelConversionOut.model_validate(status)


@router.get("/{job_id}", response_model=ModelConversionOut)
def get_model_conversion(job_id: str):
    data = read_job(job_id)
    data = _response_payload(data, job_id)
    return ModelConversionOut.model_validate(data)


@router.get("/{job_id}/download")
def download_model_conversion(job_id: str):
    out_path, filename = resolve_download_path(job_id)
    return FileResponse(
        path=str(out_path),
        filename=filename,
        media_type="application/octet-stream",
    )


def _response_payload(data: dict, job_id: str) -> dict:
    payload = dict(data)
    if str(payload.get("status") or "").strip().lower() == "completed":
        payload["output_url"] = f"/api/v3/model-conversions/{job_id}/download"
    else:
        payload["output_url"] = None
    return payload
