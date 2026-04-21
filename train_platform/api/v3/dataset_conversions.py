from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import FileResponse

from train_platform.core.config import settings
from train_platform.schemas.v3.dataset_conversions import DatasetConversionOut
from train_platform.services.v3.dataset_conversion_service import DatasetConversionService
from train_platform.services.v3.file_service import FileService
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/dataset-conversions", tags=["dataset-conversions"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _job_dir(job_id: str) -> Path:
    settings.ensure_dirs()
    root = settings.temp_dir / "dataset_conversions" / str(job_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _read_status(job_id: str) -> Dict[str, Any]:
    path = _status_path(job_id)
    if not path.exists():
        raise ValidationError("Job not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("status.json is not a dict")
        return data
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to read job status: {type(e).__name__}: {e}") from e


def _write_status(job_id: str, data: Dict[str, Any]) -> None:
    path = _status_path(job_id)
    tmp = path.with_suffix(".json.tmp")
    data = dict(data or {})
    data["updated_at"] = _utcnow().isoformat()
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(path)
    except Exception:
        # Best-effort; status updates should never crash the API.
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _append_log(data: Dict[str, Any], msg: str) -> None:
    logs = data.get("logs")
    if not isinstance(logs, list):
        logs = []
    logs.append(str(msg))
    # Keep last ~400 lines to avoid huge payloads.
    if len(logs) > 400:
        logs = logs[-400:]
    data["logs"] = logs


def _zip_dir(src_dir: Path, out_zip: Path) -> None:
    src_dir = Path(src_dir)
    out_zip = Path(out_zip)
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValidationError("Output directory not found")

    tmp = out_zip.with_suffix(out_zip.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src_dir.rglob("*"):
            if not p.is_file():
                continue
            arc = p.relative_to(src_dir).as_posix()
            zf.write(p, arcname=arc)

    tmp.replace(out_zip)


def _run_conversion_job(job_id: str) -> None:
    data = _read_status(job_id)
    data["status"] = "running"
    data["stage"] = "extracting"
    data["progress"] = 1
    target_format = data.get("target_format", "yolo")
    _append_log(data, f"Extracting zip (target: {target_format})...")
    _write_status(job_id, data)

    job_root = _job_dir(job_id)
    input_path = job_root / "input.zip"
    if not input_path.exists():
        data["status"] = "failed"
        data["progress"] = 100
        data["error_message"] = "input.zip not found"
        _append_log(data, data["error_message"])
        _write_status(job_id, data)
        return

    extract_dir = (job_root / "extract").resolve(strict=False)
    out_dir = (job_root / "output").resolve(strict=False)
    images_dir = (out_dir / "images").resolve(strict=False)
    labels_dir = (out_dir / "labels").resolve(strict=False)

    try:
        # (1) Extract to temp dir1
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as zf:
            FileService()._safe_extract_zip(zf, extract_dir)

        extracted = [p for p in extract_dir.iterdir()]
        src_dir = extracted[0] if len(extracted) == 1 and extracted[0].is_dir() else extract_dir

        json_files = sorted([p for p in src_dir.rglob("*.json") if p.is_file()])
        if not json_files:
            raise ValidationError("No .json annotation files found in zip")

        # Image files in temp dir1 (best-effort; we only copy known image extensions)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = sorted([p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts])
        if not image_files:
            raise ValidationError("No image files found in zip")

        # (2) Create temp dir2 (dataset dir) with images/labels
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # (3) Convert json -> yolo labels
        data = _read_status(job_id)
        data["stage"] = "converting_labels"
        data["total"] = int(len(json_files))
        data["processed"] = 0
        data["progress"] = max(int(data.get("progress") or 0), 10)
        _append_log(data, f"Converting {len(json_files)} json files to YOLO labels...")
        _write_status(job_id, data)

        def _on_progress(processed: int, total: int, current: str) -> None:
            st = _read_status(job_id)
            st["stage"] = "converting_labels"
            st["processed"] = int(processed)
            st["total"] = int(total)
            # Map conversion to 10..70
            pct = 10 + int(60 * (float(processed) / float(total))) if total > 0 else 70
            st["progress"] = max(int(st.get("progress") or 0), min(70, int(pct)))
            # Do not spam logs; only log occasionally.
            if processed == 1 or processed == total or processed % 25 == 0:
                _append_log(st, f"[{processed}/{total}] {current}")
            _write_status(job_id, st)

        DatasetConversionService().conversion_json_2_yolo(
            str(src_dir),
            output_labels_dir=labels_dir,
            class_names_path=out_dir / "class_names.txt",
            write_files=True,
            on_progress=_on_progress,
        )

        # (4) Copy images to temp dir2/images
        data = _read_status(job_id)
        data["stage"] = "copying_images"
        data["progress"] = max(int(data.get("progress") or 0), 72)
        _append_log(data, f"Copying {len(image_files)} images...")
        _write_status(job_id, data)

        seen_names_cf: set[str] = set()
        total_images = int(len(image_files))
        for i, src_img in enumerate(image_files, start=1):
            name = src_img.name
            key = name.casefold()
            if key in seen_names_cf:
                raise ValidationError(f"Duplicate image filename in zip: {name}")
            seen_names_cf.add(key)

            dst = (images_dir / name).resolve(strict=False)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst)

            if total_images > 0:
                st = _read_status(job_id)
                st["stage"] = "copying_images"
                # Map copy to 72..90
                pct = 72 + int(18 * (float(i) / float(total_images)))
                st["progress"] = max(int(st.get("progress") or 0), min(90, int(pct)))
                if i == 1 or i == total_images or i % 100 == 0:
                    _append_log(st, f"[{i}/{total_images}] {name}")
                _write_status(job_id, st)

        if target_format == "coco":
            # (5) Convert YOLO -> COCO
            data = _read_status(job_id)
            data["stage"] = "converting_coco"
            data["progress"] = max(int(data.get("progress") or 0), 92)
            _append_log(data, "Converting to COCO JSON...")
            _write_status(job_id, data)

            DatasetConversionService().conversion_yolo_2_coco(
                images_dir=images_dir,
                labels_dir=labels_dir,
                class_names_path=out_dir / "class_names.txt",
                output_json_path=out_dir / "annotations" / "instances_default.json",
            )
            
            # Zip only images and annotations
            # We need to filter what we zip. _zip_dir currently zips everything in src_dir.
            # Ideally we only keep images and annotations folder.
            # Let's clean up labels and auxiliary files from out_dir before zipping if COCO.
            try:
                shutil.rmtree(labels_dir, ignore_errors=True)
                (out_dir / "class_names.txt").unlink(missing_ok=True)
                (out_dir / "data.yaml").unlink(missing_ok=True) # in case it was created
            except Exception:
                pass

        else:
            # (5) Create data.yaml (YOLO)
            data = _read_status(job_id)
            data["stage"] = "writing_data_yaml"
            data["progress"] = max(int(data.get("progress") or 0), 92)
            _append_log(data, "Creating data.yaml...")
            _write_status(job_id, data)

            FileService()._create_yolo_data_yaml(out_dir, out_dir / "data.yaml")

        # Zip result
        data = _read_status(job_id)
        data["stage"] = "zipping"
        data["progress"] = max(int(data.get("progress") or 0), 96)
        _append_log(data, "Packaging zip...")
        _write_status(job_id, data)

        out_zip = (job_root / "output.zip").resolve(strict=False)
        _zip_dir(out_dir, out_zip)

        data = _read_status(job_id)
        data["status"] = "completed"
        data["stage"] = "done"
        data["progress"] = 100
        data["output_filename"] = f"dataset_{target_format}.zip"
        # Expose via static mount for temp
        data["output_url"] = f"/static/temp/dataset_conversions/{job_id}/output.zip"
        data["error_message"] = None
        _append_log(data, "Done.")
        _write_status(job_id, data)
    except Exception as e:
        data = _read_status(job_id)
        data["status"] = "failed"
        data["stage"] = "failed"
        data["progress"] = 100
        data["error_message"] = f"{type(e).__name__}: {e}"
        _append_log(data, data["error_message"])
        _write_status(job_id, data)


@router.post("", response_model=DatasetConversionOut, status_code=201)
async def create_dataset_conversion(
    file: UploadFile = File(...),
    target_format: str = Query("yolo", pattern="^(yolo|coco)$"),
):
    """
    Convert a zip containing images + json annotations into a YOLO-style dataset zip.

    This endpoint is async from the client's POV:
    - returns a job_id immediately
    - client polls GET /dataset-conversions/{job_id}
    """
    filename = file.filename or "dataset.zip"
    if not str(filename).lower().endswith(".zip"):
        raise ValidationError("Only .zip archives are supported")

    job_id = uuid.uuid4().hex
    root = _job_dir(job_id)

    input_path = (root / "input.zip").resolve(strict=False)
    if settings.temp_dir.resolve() not in input_path.parents:
        raise ValidationError("Unsafe upload path")

    # Persist upload
    try:
        with open(input_path, "wb") as f:
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
        "stage": "queued",
        "progress": 0,
        "processed": 0,
        "total": None,
        "target_format": target_format,
        "logs": [f"Received: {filename}", f"Target format: {target_format}"],
        "output_url": None,
        "output_filename": None,
        "error_message": None,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _write_status(job_id, status)

    # Run conversion in background (in-process).
    t = threading.Thread(target=_run_conversion_job, args=(job_id,), daemon=True)
    t.start()

    return DatasetConversionOut.model_validate(status)


@router.get("/{job_id}", response_model=DatasetConversionOut)
def get_dataset_conversion(job_id: str):
    data = _read_status(job_id)
    return DatasetConversionOut.model_validate(data)


@router.get("/{job_id}/download")
def download_dataset_conversion(job_id: str):
    data = _read_status(job_id)
    url = data.get("output_url")
    if not url:
        raise ValidationError("Output not ready")

    job_root = _job_dir(job_id)
    out_zip = (job_root / "output.zip").resolve(strict=False)
    if not out_zip.exists() or not out_zip.is_file():
        raise ValidationError("Output file not found")

    filename = str(data.get("output_filename") or "dataset_yolo.zip")
    return FileResponse(path=str(out_zip), filename=filename, media_type="application/zip")
