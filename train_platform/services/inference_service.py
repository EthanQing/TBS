from __future__ import annotations

import os
import time
import uuid
import ipaddress
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.architecture import ModelArchitecture
from train_platform.models.deployment import Deployment
from train_platform.models.inference import InferenceRun
from train_platform.models.model_registry import ModelVersion
from train_platform.models.training_run import TrainingRun
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.path_utils import resolve_temp_path, resolve_training_path


class InferenceService:
    INTERNAL_TOKEN_HEADER = "X-Internal-Token"

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

    def resolve_model_context(self, db: Session, *, model_version_id: int) -> Dict[str, Any]:
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if not mv:
            raise NotFoundError("Model version not found")

        weights = str(mv.weights_path or "").strip()
        if not weights:
            raise ValidationError("Model version has no weights_path; register from a completed training run first")

        weights_path = resolve_training_path(weights)
        if not weights_path.exists() or not weights_path.is_file():
            raise NotFoundError(f"Weights file not found: {weights_path}")

        run = None
        if mv.run_id:
            run = db.query(TrainingRun).filter(TrainingRun.run_id == str(mv.run_id)).first()

        arch = None
        if run:
            arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()

        engine = str(getattr(arch, "engine", "") or "ultralytics-yolo").strip().lower()
        family = str(getattr(arch, "family", "") or "").strip() or None
        variant = str(getattr(arch, "variant", "") or "").strip() or None

        config_path = None
        if engine == "paddle-det":
            config_path = self._resolve_paddle_config_path(arch)
            if not config_path:
                raise ValidationError("Paddle model missing valid config_path in architecture.default_params")

        return {
            "model_version_id": int(mv.model_version_id),
            "run_id": str(mv.run_id or ""),
            "project_id": int(mv.project_id),
            "engine": engine,
            "family": family,
            "variant": variant,
            "weights_path": str(weights_path),
            "config_path": str(config_path) if config_path else None,
        }

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
        if not input_path and not image_url:
            raise ValidationError("Either input_path or image_url is required")
        if input_path and image_url:
            raise ValidationError("Provide only one of input_path and image_url")

        local_path, stored_token, derived_meta = self._materialize_input(input_path=input_path, image_url=image_url)
        ctx = self.resolve_model_context(db, model_version_id=int(model_version_id))

        t0 = time.perf_counter()
        output = None
        err_msg = None
        try:
            output = self._run_by_engine(
                engine=str(ctx.get("engine") or "ultralytics-yolo"),
                weights_path=Path(str(ctx["weights_path"])),
                image_path=local_path,
                conf=float(conf),
                iou=float(iou),
                config_path=Path(str(ctx["config_path"])) if ctx.get("config_path") else None,
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "model_version_id": int(model_version_id),
            "input_path": stored_token,
            "input_meta": {**(derived_meta or {}), "conf": float(conf), "iou": float(iou)},
            "output": output,
            "error_message": err_msg,
            "inference_time_ms": elapsed_ms,
            "engine": ctx.get("engine"),
            "family": ctx.get("family"),
            "variant": ctx.get("variant"),
            "run_id": ctx.get("run_id"),
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
        mv = db.query(ModelVersion).filter(ModelVersion.model_version_id == int(model_version_id)).first()
        if not mv:
            raise NotFoundError("Model version not found")

        dep_id = None
        if deployment_id is not None:
            dep_id = int(deployment_id)
            dep = db.query(Deployment).filter(Deployment.deployment_id == dep_id).first()
            if not dep:
                raise NotFoundError("Deployment not found")

        result = self.run_inference_output(
            db,
            model_version_id=int(model_version_id),
            input_path=input_path,
            image_url=image_url,
            conf=float(conf),
            iou=float(iou),
        )

        row = InferenceRun(
            model_version_id=int(mv.model_version_id),
            deployment_id=dep_id,
            input_path=str(result.get("input_path") or ""),
            input_meta={**(input_meta or {}), **(result.get("input_meta") or {})},
            output=result.get("output"),
            error_message=result.get("error_message"),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def _resolve_paddle_config_path(self, arch: ModelArchitecture | None) -> Optional[Path]:
        if not arch:
            return None
        params = arch.default_params if isinstance(arch.default_params, dict) else {}
        raw = params.get("config_path")
        if raw is None:
            return None
        txt = str(raw).strip().replace("\\", "/")
        if not txt:
            return None

        p = Path(txt)
        if p.is_absolute() and p.exists():
            return p.resolve(strict=False)

        candidates = [
            (settings.paddle_det_dir / txt).resolve(strict=False),
            (settings.home_dir / txt).resolve(strict=False),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

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

        p = resolve_temp_path(raw)
        if p.exists() and p.is_file():
            token = p.relative_to(settings.temp_dir.resolve()).as_posix()
            return p, token, meta

        raise NotFoundError(f"Input file not found: {raw}")

    def _download_to_temp(self, url: str) -> tuple[Path, str]:
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        out_dir = settings.temp_dir / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(url.split("?", 1)[0]).suffix or ".jpg"
        name = f"{uuid.uuid4().hex}{suffix}"
        out_path = out_dir / name

        try:
            self._validate_remote_url(url)
            max_bytes = max(1, int(settings.inference_max_download_bytes))
            timeout = max(1.0, float(settings.inference_download_timeout_sec))

            written = 0
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValidationError(
                                f"image_url exceeds max allowed size ({max_bytes} bytes)"
                            )
                        f.write(chunk)
        except Exception as e:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise ValidationError(f"Failed to download image_url: {e}") from e

        token = out_path.relative_to(settings.temp_dir.resolve()).as_posix()
        return out_path, token

    def _validate_remote_url(self, url: str) -> None:
        parsed = urlparse(url)
        scheme = str(parsed.scheme or "").strip().lower()
        if not scheme:
            raise ValidationError("image_url scheme is required")

        allowed_schemes = {str(s).strip().lower() for s in settings.inference_allowed_schemes if str(s).strip()}
        if not allowed_schemes:
            allowed_schemes = {"http", "https"}
        if scheme not in allowed_schemes:
            raise ValidationError(f"image_url scheme not allowed: {scheme}")

        host = str(parsed.hostname or "").strip().lower()
        if not host:
            raise ValidationError("image_url host is required")

        allowed_hosts = {str(h).strip().lower() for h in settings.inference_allowed_hosts if str(h).strip()}
        if allowed_hosts and host not in allowed_hosts:
            raise ValidationError(f"image_url host not allowed: {host}")

        # When host allowlist is absent, block local/private addresses by default.
        if not allowed_hosts:
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                return
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ValidationError("image_url host resolves to a disallowed private address")

    def _internal_request_headers(self) -> Dict[str, str]:
        token = str(settings.internal_api_token or "").strip()
        if not token:
            return {}
        return {self.INTERNAL_TOKEN_HEADER: token}

    def _run_by_engine(
        self,
        *,
        engine: str,
        weights_path: Path,
        image_path: Path,
        conf: float,
        iou: float,
        config_path: Path | None = None,
    ) -> Dict[str, Any]:
        e = str(engine or "").strip().lower()
        if e == "paddle-det":
            return self._run_paddle_det(weights_path, image_path, config_path=config_path, conf=conf, iou=iou)
        return self._run_ultralytics_yolo(weights_path, image_path, conf=conf, iou=iou)

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
            resp = requests.post(
                f"{worker_url}/internal/inference/yolo",
                json=payload,
                timeout=timeout,
                headers=self._internal_request_headers(),
            )
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

    def _run_paddle_det(
        self,
        weights_path: Path,
        image_path: Path,
        *,
        config_path: Path | None,
        conf: float,
        iou: float,
    ) -> Dict[str, Any]:
        if config_path is None:
            raise ValidationError("Paddle inference requires config_path")
        if not config_path.exists():
            raise NotFoundError(f"Paddle config not found: {config_path}")

        worker_url = os.getenv("PADDLE_INFERENCE_WORKER_URL", "http://127.0.0.1:18003").rstrip("/")
        timeout = float(os.getenv("PADDLE_INFERENCE_WORKER_TIMEOUT", "240"))
        fallback_local = str(os.getenv("INFERENCE_FALLBACK_LOCAL", "1")).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
        payload = {
            "config_path": str(config_path),
            "weights_path": str(weights_path),
            "image_path": str(image_path),
            "conf": float(conf),
            "iou": float(iou),
        }

        worker_error: str | None = None
        try:
            resp = requests.post(
                f"{worker_url}/internal/inference/paddle-det",
                json=payload,
                timeout=timeout,
                headers=self._internal_request_headers(),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Paddle inference worker error {resp.status_code}: {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Paddle inference worker returned non-JSON response: {e}") from e
            err = data.get("error")
            if err:
                raise RuntimeError(str(err))
            output = data.get("output")
            if output is None:
                raise RuntimeError("Paddle inference worker response missing output")
            return output
        except Exception as e:
            worker_error = f"{type(e).__name__}: {e}"

        if not fallback_local:
            raise RuntimeError(worker_error or "Paddle inference worker request failed")

        try:
            from train_platform.workers.paddle_inference_worker import _run_paddle_det as _local_paddle_infer

            return _local_paddle_infer(config_path, weights_path, image_path, conf=float(conf), iou=float(iou))
        except Exception as e:
            fallback_error = f"{type(e).__name__}: {e}"
            if worker_error:
                raise RuntimeError(
                    f"Paddle worker failed ({worker_error}); local fallback failed ({fallback_error})"
                ) from e
            raise RuntimeError(f"Local paddle inference fallback failed: {fallback_error}") from e
