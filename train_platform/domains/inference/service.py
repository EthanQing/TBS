from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from train_platform.domains.model_assets.runtime import ModelRuntimeSpec, resolve_model_runtime
from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.inference import InferenceRun
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.utils.exceptions import NotFoundError, ValidationError

from .input import materialize_input, resolve_temp_token


class InferenceService:
    """Application service for synchronous inference capabilities."""

    def __init__(self, worker_client: ModelWorkerClient | None = None) -> None:
        self._worker = worker_client or ModelWorkerClient()

    def run_inference_output(
        self,
        db: Session,
        *,
        model_version_id: int,
        input_path: Optional[str] = None,
        image_url: Optional[str] = None,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        model = resolve_model_runtime(db, model_version_id=int(model_version_id))
        local_path, stored_token, derived_meta = materialize_input(
            input_path=input_path,
            image_url=image_url,
        )
        return self.run_inference_output_for_model(
            model,
            input_path=str(local_path),
            stored_token=stored_token,
            derived_meta=derived_meta,
            conf=float(conf),
            iou=float(iou),
        )

    def run_inference_output_for_model(
        self,
        model: ModelRuntimeSpec,
        *,
        input_path: str,
        stored_token: Optional[str] = None,
        derived_meta: Optional[Dict[str, Any]] = None,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        output = None
        error_message = None
        try:
            output = self._worker.execute_model(
                engine=model.engine,
                weights_path=model.weights_path,
                image_path=Path(input_path),
                conf=float(conf),
                iou=float(iou),
                config_path=model.config_path,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        return {
            "model_version_id": int(model.model_version_id),
            "input_path": str(stored_token or input_path),
            "input_meta": {**(derived_meta or {}), "conf": float(conf), "iou": float(iou)},
            "output": output,
            "error_message": error_message,
            "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "engine": model.engine,
            "family": model.family,
            "variant": model.variant,
            "run_id": model.run_id,
        }

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
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_version_id == int(model_version_id))
            .first()
        )
        if not model_version:
            raise NotFoundError("Model version not found")

        deployment_value = None
        if deployment_id is not None:
            deployment_value = int(deployment_id)
            deployment = (
                db.query(Deployment)
                .filter(Deployment.deployment_id == deployment_value)
                .first()
            )
            if not deployment:
                raise NotFoundError("Deployment not found")

        model = resolve_model_runtime(db, model_version=model_version)
        local_path, stored_token, derived_meta = materialize_input(
            input_path=input_path,
            image_url=image_url,
        )
        result = self.run_inference_output_for_model(
            model,
            input_path=str(local_path),
            stored_token=stored_token,
            derived_meta=derived_meta,
            conf=float(conf),
            iou=float(iou),
        )

        row = InferenceRun(
            model_version_id=int(model_version.model_version_id),
            deployment_id=deployment_value,
            input_path=str(result.get("input_path") or ""),
            input_meta={**(input_meta or {}), **(result.get("input_meta") or {})},
            output=result.get("output"),
            error_message=result.get("error_message"),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def run_batch_inference(
        self,
        db: Session,
        *,
        model_version_id: int,
        input_tokens: list[str],
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        success_count = 0
        for token in input_tokens:
            item_started = time.perf_counter()
            item: dict[str, Any] = {
                "token": token,
                "filename": Path(token).name,
                "output": None,
                "error_message": None,
                "inference_time_ms": 0,
            }
            try:
                run = self.run_inference(
                    db,
                    model_version_id=int(model_version_id),
                    input_path=token,
                    conf=float(conf),
                    iou=float(iou),
                )
                item["output"] = run.output
                item["error_message"] = run.error_message
                if run.output and not run.error_message:
                    success_count += 1
            except Exception as exc:
                item["error_message"] = f"{type(exc).__name__}: {exc}"
            item["inference_time_ms"] = round((time.perf_counter() - item_started) * 1000.0, 1)
            results.append(item)
        return {
            "results": results,
            "total": len(results),
            "success_count": success_count,
            "total_time_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    def run_video_inference(
        self,
        db: Session,
        *,
        model_version_id: int,
        video_token: str,
        frame_interval: int,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> dict[str, Any]:
        model = resolve_model_runtime(db, model_version_id=int(model_version_id))
        token = resolve_temp_token(video_token)
        try:
            return self._worker.infer_video_frames(
                engine=model.engine,
                weights_path=model.weights_path,
                video_token=token,
                frame_interval=int(frame_interval),
                conf=float(conf),
                iou=float(iou),
                config_path=model.config_path,
            )
        except Exception as exc:
            raise ValidationError(f"Failed to call inference worker: {type(exc).__name__}: {exc}") from exc


__all__ = ["InferenceService"]
