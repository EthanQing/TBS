from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from train_platform.utils.model_conversion_jobs import (
    _append_log,
    _file_size_mb,
    _job_dir,
    _read_status,
    _write_status,
)


def _env_flag_enabled(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _windows_cuda12_runtime_available() -> bool:
    if os.name != "nt":
        return True
    if shutil.which("cublasLt64_12.dll"):
        return True
    for raw_dir in os.getenv("PATH", "").split(os.pathsep):
        if not raw_dir:
            continue
        try:
            if (Path(raw_dir) / "cublasLt64_12.dll").exists():
                return True
        except Exception:
            continue
    return False


def _allow_onnx_cuda_provider() -> bool:
    forced = _env_flag_enabled("MODEL_CONVERSION_ONNX_CUDA")
    if forced is not None:
        return bool(forced)
    return _windows_cuda12_runtime_available()


def _select_device() -> tuple[str, list[str]]:
    device = "cpu"
    providers = ["CPUExecutionProvider"]

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "cuda"
    except Exception:
        pass

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers() or [])
        if "CUDAExecutionProvider" in available and _allow_onnx_cuda_provider():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    return device, providers


def _bench_torch_yolo(weights_path: Path, *, imgsz: int, device: str) -> tuple[float, float]:
    import torch
    from ultralytics import YOLO

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
    import numpy as np
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=list(providers))
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

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

    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass

    try:
        device, providers = _select_device()
        model = YOLO(str(input_path))
        data["progress"] = 25
        _append_log(data, "开始导出 ONNX...")
        _write_status(job_id, data)

        export_kwargs: Dict[str, Any] = {"dynamic": bool(dynamic)}
        if opset is not None:
            export_kwargs["opset"] = int(opset)
        export_kwargs["device"] = 0 if device == "cuda" else "cpu"

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
                    shutil.copy2(newest, out)

        if not out.exists():
            raise RuntimeError("ONNX export failed: output file not found")

        try:
            data["progress"] = 92
            _append_log(data, "计算模型大小与性能指标...")
            _write_status(job_id, data)

            imgsz = 640
            perf: Dict[str, Any] = {
                "device": device,
                "onnx_provider": providers[0] if providers else "CPUExecutionProvider",
                "imgsz": int(imgsz),
                "pt": {"size_mb": _file_size_mb(input_path)},
                "onnx": {"size_mb": _file_size_mb(out)},
            }
            if device == "cuda" and "CUDAExecutionProvider" not in providers:
                _append_log(data, "ONNX Runtime CUDA provider 不可用，ONNX 性能测试使用 CPU")

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

        data["status"] = "completed"
        data["progress"] = 100
        data["output_url"] = f"/api/v3/model-conversions/{job_id}/download"
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
