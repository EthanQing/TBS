from __future__ import annotations

import uuid
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.deployment import Deployment
from train_platform.models.inference import InferenceRun
from train_platform.models.model_registry import ModelVersion
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_temp_path, resolve_training_path


class InferenceService:
    def create_inference_run(self, db: Session, *, obj: dict) -> InferenceRun:
        mv_id = int(obj["model_version_id"])
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == mv_id).first()
        if not mv:
            raise NotFoundError("Model version not found")

        dep_id = obj.get("deployment_id")
        if dep_id is not None:
            dep = db.query(Deployment).filter(Deployment.deployment_id == int(dep_id)).first()
            if not dep:
                raise NotFoundError("Deployment not found")

        row = InferenceRun(
            model_version_id=mv_id,
            deployment_id=int(dep_id) if dep_id is not None else None,
            input_path=str(obj["input_path"]),
            input_meta=obj.get("input_meta"),
            output=None,
            error_message=None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def run_inference(
        self,
        db: Session,
        *,
        model_version_id: int,
        input_path: Optional[str] = None,
        image_url: Optional[str] = None,
        input_meta: Optional[Dict[str, Any]] = None,
        deployment_id: Optional[int] = None,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> InferenceRun:
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if not mv:
            raise NotFoundError("Model version not found")

        dep_id = None
        if deployment_id is not None:
            dep_id = int(deployment_id)
            dep = db.query(Deployment).filter(Deployment.deployment_id == dep_id).first()
            if not dep:
                raise NotFoundError("Deployment not found")

        if not input_path and not image_url:
            raise ValidationError("Either input_path or image_url is required")
        if input_path and image_url:
            raise ValidationError("Provide only one of input_path and image_url")

        local_path, stored_token, derived_meta = self._materialize_input(input_path=input_path, image_url=image_url)

        weights = mv.weights_path
        if not weights:
            raise ValidationError("Model version has no weights_path; register from a completed training run first")

        weights_path = resolve_training_path(weights)
        if not weights_path.exists():
            raise NotFoundError(f"Weights file not found: {weights_path}")

        row = InferenceRun(
            model_version_id=int(mv.model_version_id),
            deployment_id=dep_id,
            input_path=stored_token,
            input_meta={**(input_meta or {}), **(derived_meta or {}), "conf": float(conf), "iou": float(iou)},
            output=None,
            error_message=None,
        )
        db.add(row)
        db.flush()

        try:
            output = self._run_ultralytics_yolo(weights_path, local_path, conf=float(conf), iou=float(iou))
            row.output = output
            row.error_message = None
        except Exception as e:
            row.output = None
            row.error_message = f"{type(e).__name__}: {e}"

        db.commit()
        db.refresh(row)
        return row

    def _materialize_input(self, *, input_path: Optional[str], image_url: Optional[str]) -> tuple[Path, str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        if image_url:
            url = str(image_url).strip()
            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValidationError("image_url must be http(s)")
            local_path, token = self._download_to_temp(url)
            meta["image_url"] = url
            return local_path, token, meta

        assert input_path is not None
        raw = str(input_path).strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            local_path, token = self._download_to_temp(raw)
            meta["image_url"] = raw
            return local_path, token, meta

        # If client passes a /static/temp/... URL or a relative token, resolve to BASE_TEMP_DIR.
        p = resolve_temp_path(raw)
        if p.exists() and p.is_file():
            token = p.relative_to(settings.temp_dir.resolve()).as_posix() if settings.temp_dir.resolve() in p.parents else str(p)
            return p, token, meta

        # As a fallback, accept absolute file paths on the server.
        p2 = Path(raw)
        if p2.is_absolute() and p2.exists() and p2.is_file():
            return p2, str(p2), meta

        raise NotFoundError(f"Input file not found: {raw}")

    def _download_to_temp(self, url: str) -> tuple[Path, str]:
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        out_dir = settings.temp_dir / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(url.split("?", 1)[0]).suffix or ".jpg"
        name = f"{uuid.uuid4().hex}{suffix}"
        out_path = out_dir / name

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        except Exception as e:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise ValidationError(f"Failed to download image_url: {e}") from e

        token = out_path.relative_to(settings.temp_dir.resolve()).as_posix()
        return out_path, token

    def _run_ultralytics_yolo(self, weights_path: Path, image_path: Path, *, conf: float, iou: float) -> Dict[str, Any]:
        worker_url = os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")
        timeout = float(os.getenv("INFERENCE_WORKER_TIMEOUT", "120"))
        fallback_local = str(os.getenv("INFERENCE_FALLBACK_LOCAL", "1")).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
        payload = {
            "weights_path": str(weights_path),
            "image_path": str(image_path),
            "conf": float(conf),
            "iou": float(iou),
        }

        worker_error: str | None = None
        try:
            resp = requests.post(f"{worker_url}/internal/inference/yolo", json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Inference worker error {resp.status_code}: {resp.text}")

            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Inference worker returned non-JSON response: {e}") from e

            err = data.get("error")
            if err:
                raise RuntimeError(str(err))

            output = data.get("output")
            if output is None:
                raise RuntimeError("Inference worker response missing output")

            return output
        except Exception as e:
            worker_error = f"{type(e).__name__}: {e}"

        if not fallback_local:
            raise RuntimeError(worker_error or "Inference worker request failed")

        try:
            from train_platform.workers.inference_worker import _run_ultralytics_yolo as _local_infer

            return _local_infer(weights_path, image_path, conf=float(conf), iou=float(iou))
        except Exception as e:
            fallback_error = f"{type(e).__name__}: {e}"
            if worker_error:
                raise RuntimeError(
                    f"Inference worker failed ({worker_error}); local fallback failed ({fallback_error})"
                ) from e
            raise RuntimeError(f"Local inference fallback failed: {fallback_error}") from e

        raise RuntimeError(worker_error or "Inference failed")
