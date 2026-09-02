from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import requests

from train_platform.core.config import settings


class ModelWorkerError(RuntimeError):
    """A model worker could not accept or complete a runtime request."""


class ModelWorkerClient:
    """Small HTTP client for the model inference worker processes."""

    INTERNAL_TOKEN_HEADER = "X-Internal-Token"

    def __init__(self) -> None:
        self._internal_token = str(settings.internal_api_token or "").strip()

    def execute_model(
        self,
        *,
        engine: str,
        weights_path: str | Path,
        image_path: str | Path,
        conf: float,
        iou: float,
        config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        normalized_engine = self._normalize_engine(engine)
        payload: dict[str, Any] = {
            "weights_path": str(weights_path),
            "image_path": str(image_path),
            "conf": float(conf),
            "iou": float(iou),
        }
        if normalized_engine == "paddle-det":
            payload["config_path"] = str(config_path or "")

        endpoint = (
            "/internal/inference/paddle-det"
            if normalized_engine == "paddle-det"
            else "/internal/inference/yolo"
        )
        data = self._post_json(
            self._endpoint(normalized_engine, endpoint),
            payload,
            timeout=self._timeout(
                "PADDLE_INFERENCE_WORKER_TIMEOUT"
                if normalized_engine == "paddle-det"
                else "INFERENCE_WORKER_TIMEOUT",
                240.0 if normalized_engine == "paddle-det" else 120.0,
            ),
        )
        output = data.get("output")
        if output is None:
            raise ModelWorkerError("Model worker response missing output")
        if not isinstance(output, dict):
            raise ModelWorkerError("Model worker response output must be an object")
        return output

    def validate_ultralytics_yolo(
        self,
        *,
        weights_path: str | Path,
        data_yaml: str | Path,
        conf: float,
        iou: float,
    ) -> dict[str, Any]:
        data = self._post_json(
            self._endpoint("ultralytics-yolo", "/internal/model-evaluations/yolo-val"),
            {
                "weights_path": str(weights_path),
                "data_yaml": str(data_yaml),
                "conf": float(conf),
                "iou": float(iou),
            },
            timeout=self._timeout("INFERENCE_WORKER_TIMEOUT", 7200.0),
        )
        output = data.get("output")
        if output is None:
            raise ModelWorkerError("Model worker validation response missing output")
        if not isinstance(output, dict):
            raise ModelWorkerError("Model worker validation output must be an object")
        return output

    def infer_video_frames(
        self,
        *,
        engine: str,
        weights_path: str | Path,
        video_token: str,
        frame_interval: int,
        conf: float,
        iou: float,
        config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        normalized_engine = self._normalize_engine(engine)
        payload: dict[str, Any] = {
            "weights_path": str(weights_path),
            "video_token": str(video_token),
            "frame_interval": int(frame_interval),
            "conf": float(conf),
            "iou": float(iou),
        }
        if normalized_engine == "paddle-det":
            payload["config_path"] = str(config_path or "")
        return self._post_json(
            self._endpoint(normalized_engine, "/internal/inference/video-frames"),
            payload,
            timeout=self._timeout(
                "PADDLE_INFERENCE_WORKER_TIMEOUT"
                if normalized_engine == "paddle-det"
                else "INFERENCE_WORKER_TIMEOUT",
                240.0 if normalized_engine == "paddle-det" else 120.0,
            ),
        )

    def dispatch_inference_job(
        self,
        *,
        engine: str,
        job_id: str,
        mode: str,
        weights_path: str | Path,
        input_tokens: list[str],
        video_token: str | None,
        conf: float,
        iou: float,
        show_labels: bool,
        show_confidence: bool,
        config_path: str | Path | None = None,
    ) -> None:
        normalized_engine = self._normalize_engine(engine)
        payload: dict[str, Any] = {
            "job_id": str(job_id),
            "mode": str(mode),
            "weights_path": str(weights_path),
            "input_tokens": list(input_tokens),
            "video_token": video_token,
            "conf": float(conf),
            "iou": float(iou),
            "show_labels": bool(show_labels),
            "show_confidence": bool(show_confidence),
        }
        if normalized_engine == "paddle-det":
            payload["config_path"] = str(config_path or "")

        data = self._post_json(
            self._endpoint(normalized_engine, "/internal/inference-jobs/run"),
            payload,
            timeout=self._timeout("INFERENCE_JOB_WORKER_DISPATCH_TIMEOUT", 10.0),
        )
        status = str(data.get("status") or "").strip().lower()
        if status not in {"started", "ok"}:
            raise ModelWorkerError(
                str(data.get("error") or f"Model worker returned status={status or 'unknown'}")
            )

    def get_yolo_model_stats(
        self,
        *,
        weights_path: str | Path,
        image_path: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        warmup: int,
        iters: int,
    ) -> dict[str, Any]:
        return self._post_json(
            self._endpoint("ultralytics-yolo", "/internal/model-stats/yolo"),
            {
                "weights_path": str(weights_path),
                "image_path": str(image_path),
                "imgsz": int(imgsz),
                "conf": float(conf),
                "iou": float(iou),
                "warmup": int(warmup),
                "iters": int(iters),
            },
            timeout=self._timeout("INFERENCE_WORKER_TIMEOUT", 120.0),
        )

    @staticmethod
    def _normalize_engine(engine: str) -> str:
        return str(engine or "").strip().lower() or "ultralytics-yolo"

    @staticmethod
    def _timeout(env_name: str, default: float) -> float:
        raw = os.getenv(env_name)
        try:
            return max(0.1, float(raw if raw is not None else default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _worker_url(engine: str) -> str:
        if engine == "paddle-det":
            return os.getenv("PADDLE_INFERENCE_WORKER_URL", "http://127.0.0.1:18003").rstrip("/")
        return os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")

    def _endpoint(self, engine: str, path: str) -> str:
        return f"{self._worker_url(engine)}{path}"

    def _headers(self) -> dict[str, str]:
        if not self._internal_token:
            return {}
        return {self.INTERNAL_TOKEN_HEADER: self._internal_token}

    def _post_json(self, url: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            response = requests.post(url, json=dict(payload), timeout=timeout, headers=self._headers())
        except requests.RequestException as exc:
            raise ModelWorkerError(f"Model worker request failed: {type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise ModelWorkerError(f"Model worker request failed: {type(exc).__name__}: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise ModelWorkerError(
                f"Model worker returned non-JSON response (HTTP {response.status_code}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ModelWorkerError("Model worker response must be a JSON object")
        if response.status_code != 200:
            detail = str(data.get("error") or data.get("detail") or "").strip()
            raise ModelWorkerError(detail or f"Model worker returned HTTP {response.status_code}")
        error = str(data.get("error") or "").strip()
        if error:
            raise ModelWorkerError(error)
        return data


__all__ = ["ModelWorkerClient", "ModelWorkerError"]
