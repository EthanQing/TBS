from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session
from urllib.parse import quote

from train_platform.api.deps import get_db
from train_platform.schemas.v3.training_reports import TrainingRunReportOut
from train_platform.services.v3.training_run_service import TrainingRunService
from train_platform.utils.training_report_docx import (
    DOCX_MEDIA_TYPE,
    build_training_report_docx,
    build_training_report_filename,
)


router = APIRouter(prefix="/training-runs", tags=["training-reports"])


@router.get("/{run_id}/report", response_model=TrainingRunReportOut)
def get_training_report(run_id: str, db: Session = Depends(get_db)):
    return TrainingRunService().build_report(db, run_id)


@router.get("/{run_id}/report.docx")
def download_training_report_docx(run_id: str, db: Session = Depends(get_db)):
    report = TrainingRunService().build_report(db, run_id)
    filename = build_training_report_filename(report)
    content = build_training_report_docx(report)
    quoted = quote(filename)
    return Response(
        content=content,
        media_type=DOCX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f"attachment; filename=training_report.docx; filename*=UTF-8''{quoted}",
        },
    )
