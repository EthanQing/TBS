from __future__ import annotations

import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from train_platform.core.config import settings

app = FastAPI(title="Inference Worker", version="1.0")


class InferenceRequest(BaseModel):
    weights_path: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    conf: float = 0.5
    iou: float = 0.45


class InferenceResponse(BaseModel):
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    inference_time_ms: Optional[float] = None


class ModelConversionRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    opset: Optional[int] = None
    dynamic: bool = True


class ExportOnnxRequest(BaseModel):
    src_pt: str = Field(..., min_length=1)
    out_onnx: str = Field(..., min_length=1)
    dynamic: bool = True
    opset: Optional[int] = None
    imgsz: Optional[int] = None


class ModelStatsRequest(BaseModel):
    weights_path: str = Field(..., min_length=1)
    image_path: Optional[str] = None
    imgsz: int = Field(640, ge=1)
    conf: float = 0.25
    iou: float = 0.45
    warmup: int = Field(1, ge=0)
    iters: int = Field(5, ge=1)


class InferenceJobRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    mode: str = Field("image", min_length=1)
    weights_path: str = Field(..., min_length=1)
    input_tokens: list[str] = Field(default_factory=list)
    video_token: Optional[str] = None
    conf: float = 0.5
    iou: float = 0.45
    show_labels: bool = True
    show_confidence: bool = True


class VideoFrameSamplingRequest(BaseModel):
    weights_path: str = Field(..., min_length=1)
    video_token: str = Field(..., min_length=1)
    frame_interval: int = Field(1, ge=1)
    conf: float = 0.5
    iou: float = 0.45


class WorkerStatusResponse(BaseModel):
    status: str
    error: Optional[str] = None


class ModelStatsResponse(BaseModel):
    flops: Optional[int] = None
    inference_time_ms: Optional[float] = None
    device: Optional[str] = None
    error: Optional[str] = None


def _verify_internal_auth(x_internal_token: Optional[str] = Header(default=None, alias="X-Internal-Token")) -> None:
    expected = str(settings.internal_api_token or "").strip()
    if not expected:
        return
    provided = str(x_internal_token or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized internal request")


def _resolve_training_path(raw: str, *, label: str, must_exist: bool = True) -> Path:
    base = settings.training_dir.resolve()
    path = Path(str(raw)).resolve(strict=False)
    try:
        path.relative_to(base)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{label} must be under training directory") from e
    if must_exist and (not path.exists() or not path.is_file()):
        raise HTTPException(status_code=404, detail=f"{label} not found: {path}")
    return path


def _apply_ultralytics_safe_load_patches() -> None:
    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass


def _select_ultralytics_device() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


def _torch_device(device: str):
    import torch

    raw = str(device or "").strip().lower()
    if raw and raw != "cpu":
        return torch.device("cuda:0")
    return torch.device("cpu")


def _sync_device(device: str) -> None:
    try:
        import torch

        if str(device or "").strip().lower() != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _load_ultralytics_yolo(weights_path: Path):
    from ultralytics import YOLO

    _apply_ultralytics_safe_load_patches()
    return YOLO(str(weights_path))


def _ensure_benchmark_image() -> Path:
    out_dir = (settings.temp_dir / "benchmark_inputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_640.jpg"
    if not out_path.exists():
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError(f"Pillow is required for benchmark image generation: {e}") from e
        Image.new("RGB", (640, 640), color=(0, 0, 0)).save(out_path, format="JPEG", quality=95)
    return out_path


def _run_ultralytics_yolo(weights_path: Path, image_path: Path, *, conf: float, iou: float) -> Dict[str, Any]:
    model = _load_ultralytics_yolo(weights_path)
    device = _select_ultralytics_device()
    results = model.predict(source=str(image_path), conf=float(conf), iou=float(iou), device=device, verbose=False)
    if not results:
        return {"predictions": [], "names": {}}

    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(model, "names", None) or {}
    out: Dict[str, Any] = {"predictions": [], "names": names}

    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return out

    for b in boxes:
        try:
            cls_id = int(getattr(b, "cls")[0])
        except Exception:
            try:
                cls_id = int(b.cls)
            except Exception:
                cls_id = -1
        try:
            conf_v = float(getattr(b, "conf")[0])
        except Exception:
            try:
                conf_v = float(b.conf)
            except Exception:
                conf_v = 0.0
        try:
            xyxy = getattr(b, "xyxy")[0].tolist()
            xyxy = [float(x) for x in xyxy]
        except Exception:
            xyxy = None

        out["predictions"].append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id) if isinstance(names, dict) else None,
                "confidence": conf_v,
                "xyxy": xyxy,
            }
        )

    return out


def _compute_ultralytics_model_flops(model, *, imgsz: int = 640, device: str = "cpu") -> Optional[int]:
    try:
        from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler
    except Exception:
        return None

    try:
        net = getattr(model, "model", None)
        if net is None:
            return None
        net.to(_torch_device(device))
        net.eval()
        size = max(1, int(imgsz or 640))
        gflops = float(get_flops(net, imgsz=size) or 0.0)
        if gflops <= 0:
            gflops = float(get_flops_with_torch_profiler(net, imgsz=size) or 0.0)
        return int(round(gflops * 1e9)) if gflops > 0 else None
    except Exception:
        return None


def _benchmark_ultralytics_yolo(
    model,
    image_path: Path,
    *,
    conf: float,
    iou: float,
    warmup: int,
    iters: int,
    device: str,
) -> float:
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))

    for _ in range(warmup):
        model.predict(source=str(image_path), conf=float(conf), iou=float(iou), device=device, verbose=False)
    _sync_device(device)

    timings: list[float] = []
    for _ in range(iters):
        _sync_device(device)
        t0 = time.perf_counter()
        model.predict(source=str(image_path), conf=float(conf), iou=float(iou), device=device, verbose=False)
        _sync_device(device)
        timings.append((time.perf_counter() - t0) * 1000.0)

    return round(float(statistics.median(timings)), 4)


@app.post("/internal/inference/yolo", response_model=InferenceResponse)
def run_inference(
    req: InferenceRequest,
    _: None = Depends(_verify_internal_auth),
) -> InferenceResponse:
    weights_path = Path(req.weights_path)
    if not weights_path.exists() or not weights_path.is_file():
        raise HTTPException(status_code=404, detail=f"Weights not found: {weights_path}")

    image_path = Path(req.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    t0 = time.perf_counter()
    try:
        output = _run_ultralytics_yolo(weights_path, image_path, conf=req.conf, iou=req.iou)
    except Exception as e:
        return InferenceResponse(error=f"{type(e).__name__}: {e}")
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return InferenceResponse(output=output, inference_time_ms=dt_ms)


@app.post("/internal/model-stats/yolo", response_model=ModelStatsResponse)
def model_stats(
    req: ModelStatsRequest,
    _: None = Depends(_verify_internal_auth),
) -> ModelStatsResponse:
    weights_path = _resolve_training_path(req.weights_path, label="weights", must_exist=True)
    if req.image_path:
        image_path = Path(req.image_path)
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    else:
        image_path = _ensure_benchmark_image()

    device = _select_ultralytics_device()
    try:
        model = _load_ultralytics_yolo(weights_path)
        flops = _compute_ultralytics_model_flops(model, imgsz=int(req.imgsz or 640), device=device)
        latency = _benchmark_ultralytics_yolo(
            model,
            image_path,
            conf=float(req.conf),
            iou=float(req.iou),
            warmup=int(req.warmup),
            iters=int(req.iters),
            device=device,
        )
    except Exception as e:
        return ModelStatsResponse(device=device, error=f"{type(e).__name__}: {e}")
    if flops is None and latency is None:
        return ModelStatsResponse(device=device, error="Unable to compute model stats")
    return ModelStatsResponse(
        flops=int(flops) if flops is not None else None,
        inference_time_ms=latency,
        device=device,
    )


@app.post("/internal/inference-jobs/run", response_model=WorkerStatusResponse)
def start_inference_job(
    req: InferenceJobRequest,
    _: None = Depends(_verify_internal_auth),
) -> WorkerStatusResponse:
    weights_path = _resolve_training_path(req.weights_path, label="weights", must_exist=True)
    status_path = settings.temp_dir / "inference_jobs" / str(req.job_id) / "status.json"
    if not status_path.exists():
        return WorkerStatusResponse(
            status="error",
            error=f"Job status is not visible to inference worker: {status_path}",
        )

    try:
        from train_platform.workers.inference_job_task import run_inference_job
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"Failed to import inference job runner: {type(e).__name__}: {e}")

    def _infer_image(image_path: Path) -> Dict[str, Any]:
        return _run_ultralytics_yolo(
            weights_path,
            image_path,
            conf=float(req.conf),
            iou=float(req.iou),
        )

    def _runner() -> None:
        run_inference_job(
            req.job_id,
            mode=req.mode,
            input_tokens=list(req.input_tokens or []),
            video_token=req.video_token,
            infer_image=_infer_image,
            show_labels=bool(req.show_labels),
            show_confidence=bool(req.show_confidence),
        )

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return WorkerStatusResponse(status="started")


@app.post("/internal/inference/video-frames")
def run_video_frame_sampling(
    req: VideoFrameSamplingRequest,
    _: None = Depends(_verify_internal_auth),
) -> Dict[str, Any]:
    weights_path = _resolve_training_path(req.weights_path, label="weights", must_exist=True)

    from train_platform.workers.inference_job_task import run_video_frame_sampling as _run_sampling

    def _infer_image(image_path: Path) -> Dict[str, Any]:
        return _run_ultralytics_yolo(
            weights_path,
            image_path,
            conf=float(req.conf),
            iou=float(req.iou),
        )

    return _run_sampling(
        video_token=req.video_token,
        frame_interval=int(req.frame_interval),
        infer_image=_infer_image,
    )


@app.post("/internal/model-conversions/pt-to-onnx", response_model=WorkerStatusResponse)
def start_model_conversion(
    req: ModelConversionRequest,
    _: None = Depends(_verify_internal_auth),
) -> WorkerStatusResponse:
    job_root = settings.temp_dir / "model_conversions" / str(req.job_id)
    status_path = job_root / "status.json"
    input_path = job_root / "input.pt"
    missing = []
    if not status_path.exists():
        missing.append("status.json")
    if not input_path.exists():
        missing.append("input.pt")
    if missing:
        return WorkerStatusResponse(
            status="error",
            error=(
                "Job artifacts are not visible to inference worker: "
                + ", ".join(missing)
                + f" under {job_root}"
            ),
        )

    try:
        from train_platform.utils.model_conversion_jobs import _append_log, _read_status, _write_status
        from train_platform.workers.model_conversion_task import _run_pt_to_onnx
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"Failed to import conversion worker: {type(e).__name__}: {e}")

    def _runner() -> None:
        try:
            _run_pt_to_onnx(req.job_id, opset=req.opset, dynamic=req.dynamic)
        except Exception as e:
            try:
                data = _read_status(req.job_id)
                data["status"] = "failed"
                data["progress"] = 100
                data["error_message"] = f"{type(e).__name__}: {e}"
                _append_log(data, data["error_message"])
                _write_status(req.job_id, data)
            except Exception:
                pass

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return WorkerStatusResponse(status="started")


@app.post("/internal/training-runs/export-onnx", response_model=WorkerStatusResponse)
def export_training_onnx(
    req: ExportOnnxRequest,
    _: None = Depends(_verify_internal_auth),
) -> WorkerStatusResponse:
    src_pt = _resolve_training_path(req.src_pt, label="weights", must_exist=True)
    out_onnx = _resolve_training_path(req.out_onnx, label="output", must_exist=False)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    try:
        model = _load_ultralytics_yolo(src_pt)
        device = _select_ultralytics_device()

        export_kwargs: Dict[str, Any] = {"dynamic": bool(req.dynamic)}
        if req.opset is not None:
            export_kwargs["opset"] = int(req.opset)
        if req.imgsz is not None:
            export_kwargs["imgsz"] = int(req.imgsz)
        export_kwargs["device"] = 0 if device != "cpu" else "cpu"

        exported = model.export(format="onnx", **export_kwargs)

        if not out_onnx.exists():
            exported_path: Optional[Path] = None
            try:
                if exported:
                    exported_path = Path(str(exported)).resolve(strict=False)
            except Exception:
                exported_path = None

            if exported_path and exported_path.exists():
                try:
                    exported_path.replace(out_onnx)
                except Exception:
                    pass
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"{type(e).__name__}: {e}")

    if not out_onnx.exists():
        return WorkerStatusResponse(status="error", error=f"ONNX export did not produce {out_onnx.name}")

    return WorkerStatusResponse(status="ok")


if __name__ == "__main__":
    import uvicorn

    host = (
        str(settings.worker_bind_host).strip()
        or os.getenv("INFERENCE_WORKER_HOST")
        or "0.0.0.0"
    )
    port = int(os.getenv("INFERENCE_WORKER_PORT", "18002"))
    uvicorn.run("train_platform.workers.inference_worker:app", host=host, port=port)
