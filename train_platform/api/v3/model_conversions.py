from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, UploadFile
import requests

from train_platform.core.config import settings
from train_platform.schemas.v3.model_conversions import ModelConversionOut
from train_platform.utils.exceptions import ValidationError


router = APIRouter(prefix="/model-conversions", tags=["model-conversions"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _job_dir(job_id: str) -> Path:
    settings.ensure_dirs()
    root = settings.temp_dir / "model_conversions" / str(job_id)
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


def _bytes_to_mb(n: int | float | None) -> float | None:
    try:
        if n is None:
            return None
        v = float(n)
        if v < 0:
            return None
        return round(v / (1024 * 1024), 2)
    except Exception:
        return None


def _file_size_mb(path: Path) -> float | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        return _bytes_to_mb(int(path.stat().st_size))
    except Exception:
        return None


def _select_device() -> tuple[str, list[str]]:
    """
    Pick a device/provider that works for both PyTorch and ONNX Runtime.

    Returns: (device, onnx_providers)
    - device: "cuda" | "cpu"
    - onnx_providers: list passed into onnxruntime.InferenceSession
    """
    device = "cpu"
    providers = ["CPUExecutionProvider"]
    try:
        import torch

        has_cuda_torch = bool(torch.cuda.is_available())
    except Exception:
        has_cuda_torch = False

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers() or [])
        has_cuda_ort = "CUDAExecutionProvider" in available
    except Exception:
        has_cuda_ort = False

    if has_cuda_torch and has_cuda_ort:
        device = "cuda"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return device, providers


def _bench_torch_yolo(weights_path: Path, *, imgsz: int, device: str) -> tuple[float, float]:
    """
    Return (latency_ms, throughput_img_s) using a simple forward benchmark.
    """
    import torch
    from ultralytics import YOLO

    # Ensure safe torch load on Windows for Ultralytics weights.
    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass

    model = YOLO(str(weights_path))
    net = getattr(model, "model", None)
    if net is None:
        raise RuntimeError("Ultralytics YOLO model is missing .model")

    dev = torch.device("cuda:0" if device == "cuda" else "cpu")
    net.to(dev)
    net.eval()

    x = torch.randn(1, 3, int(imgsz), int(imgsz), device=dev)

    warmup = 5
    iters = 20

    with torch.no_grad():
        for _ in range(warmup):
            _ = net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    avg_s = (t1 - t0) / float(iters)
    latency_ms = float(avg_s * 1000.0)
    throughput = float(1.0 / avg_s) if avg_s > 0 else 0.0
    return round(latency_ms, 2), round(throughput, 2)


def _bench_onnx(onnx_path: Path, *, imgsz: int, providers: list[str]) -> tuple[float, float]:
    """
    Return (latency_ms, throughput_img_s) using ONNX Runtime inference.
    """
    import numpy as np
    import onnxruntime as ort

    # Session options: keep defaults; this is a best-effort quick benchmark.
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=list(providers))
    except Exception:
        # Fallback to CPU if CUDA provider fails to initialize.
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    in0 = sess.get_inputs()[0]
    name = in0.name

    x = np.random.randn(1, 3, int(imgsz), int(imgsz)).astype(np.float32)

    warmup = 5
    iters = 50 if "CUDAExecutionProvider" in (sess.get_providers() or []) else 20

    for _ in range(warmup):
        _ = sess.run(None, {name: x})

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {name: x})
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / float(iters)
    latency_ms = float(avg_s * 1000.0)
    throughput = float(1.0 / avg_s) if avg_s > 0 else 0.0
    return round(latency_ms, 2), round(throughput, 2)


def _run_pt_to_onnx(job_id: str, *, opset: int | None, dynamic: bool) -> None:
    data = _read_status(job_id)
    data["status"] = "running"
    data["progress"] = 5
    _append_log(data, "加载模型...")
    _write_status(job_id, data)

    job_root = _job_dir(job_id)
    input_path = job_root / "input.pt"
    if not input_path.exists():
        data["status"] = "failed"
        data["error_message"] = "input.pt not found"
        _append_log(data, data["error_message"])
        _write_status(job_id, data)
        return

    try:
        from ultralytics import YOLO
    except Exception as e:
        data["status"] = "failed"
        data["error_message"] = f"Ultralytics not installed: {type(e).__name__}: {e}"
        _append_log(data, data["error_message"])
        _write_status(job_id, data)
        return

    # Ensure safe torch load on Windows for Ultralytics weights.
    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass

    try:
        model = YOLO(str(input_path))
        data["progress"] = 25
        _append_log(data, "开始导出 ONNX...")
        _write_status(job_id, data)

        export_kwargs: Dict[str, Any] = {"dynamic": bool(dynamic)}
        if opset is not None:
            export_kwargs["opset"] = int(opset)

        exported = model.export(format="onnx", **export_kwargs)
        data["progress"] = 85
        _append_log(data, "写入输出文件...")
        _write_status(job_id, data)

        exported_path: Path | None = None
        try:
            if exported:
                exported_path = Path(str(exported)).resolve(strict=False)
        except Exception:
            exported_path = None

        out = job_root / "output.onnx"
        if not out.exists():
            newest: Path | None = None
            try:
                for cand in job_root.glob("*.onnx"):
                    if newest is None or cand.stat().st_mtime > newest.stat().st_mtime:
                        newest = cand
            except Exception:
                newest = None

            if exported_path and exported_path.exists():
                newest = exported_path

            if newest and newest.exists() and newest != out:
                try:
                    newest.replace(out)
                except Exception:
                    import shutil

                    shutil.copy2(newest, out)

        if not out.exists():
            raise RuntimeError("ONNX export failed: output file not found")

        # Compute basic stats/performance (best-effort). Never fail the conversion for benchmark issues.
        try:
            data["progress"] = 92
            _append_log(data, "计算模型大小与性能指标...")
            _write_status(job_id, data)

            imgsz = 640
            device, providers = _select_device()

            perf: Dict[str, Any] = {
                "device": device,
                "imgsz": int(imgsz),
                "pt": {"size_mb": _file_size_mb(input_path)},
                "onnx": {"size_mb": _file_size_mb(out)},
            }

            # Latency/throughput benchmarks (can be slow; keep it small).
            try:
                pt_lat, pt_thr = _bench_torch_yolo(input_path, imgsz=imgsz, device=device)
                perf["pt"]["latency_ms"] = pt_lat
                perf["pt"]["throughput_img_s"] = pt_thr
            except Exception as e:
                _append_log(data, f"PT 性能测试失败: {type(e).__name__}: {e}")

            try:
                onnx_lat, onnx_thr = _bench_onnx(out, imgsz=imgsz, providers=providers)
                perf["onnx"]["latency_ms"] = onnx_lat
                perf["onnx"]["throughput_img_s"] = onnx_thr
            except Exception as e:
                _append_log(data, f"ONNX 性能测试失败: {type(e).__name__}: {e}")

            data["performance"] = perf
        except Exception as e:
            _append_log(data, f"性能统计失败(已忽略): {type(e).__name__}: {e}")

        token = f"model_conversions/{job_id}/{out.name}"
        data["status"] = "completed"
        data["progress"] = 100
        data["output_url"] = f"/static/temp/{token}"
        data["output_filename"] = out.name
        data["error_message"] = None
        _append_log(data, "转换完成")
        _write_status(job_id, data)
    except Exception as e:
        data["status"] = "failed"
        data["progress"] = 100
        data["error_message"] = f"{type(e).__name__}: {e}"
        _append_log(data, data["error_message"])
        _write_status(job_id, data)


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
        "output_url": None,
        "output_filename": None,
        "error_message": None,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _write_status(job_id, status)

    worker_url = os.getenv("INFERENCE_WORKER_URL", "http://127.0.0.1:18002").rstrip("/")
    headers = {}
    token = str(settings.internal_api_token or "").strip()
    if token:
        headers["X-Internal-Token"] = token
    try:
        resp = requests.post(
            f"{worker_url}/internal/model-conversions/pt-to-onnx",
            json={"job_id": job_id, "opset": opset, "dynamic": dynamic},
            timeout=10,
            headers=headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        _append_log(status, "dispatched to worker")
        _write_status(job_id, status)
    except Exception as e:
        status["status"] = "failed"
        status["progress"] = 100
        status["error_message"] = f"Failed to dispatch to worker: {type(e).__name__}: {e}"
        _append_log(status, status["error_message"])
        _write_status(job_id, status)

    return ModelConversionOut.model_validate(status)


@router.get("/{job_id}", response_model=ModelConversionOut)
def get_model_conversion(job_id: str):
    data = _read_status(job_id)
    return ModelConversionOut.model_validate(data)
