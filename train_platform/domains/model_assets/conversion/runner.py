from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from train_platform.platform.jobs import JobStatus
from train_platform.platform.runtime.ultralytics import apply_torch_safe_load_patches

from .jobs import input_path, read_job, update_job


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
        except OSError:
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


def _bytes_to_mb(value: int | float | None) -> float | None:
    try:
        if value is None or float(value) < 0:
            return None
        return round(float(value) / (1024 * 1024), 2)
    except (TypeError, ValueError):
        return None


def _file_size_mb(path: Path) -> float | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        return _bytes_to_mb(path.stat().st_size)
    except OSError:
        return None


def _bench_torch_yolo(weights_path: Path, *, imgsz: int, device: str) -> tuple[float, float]:
    import torch
    from ultralytics import YOLO

    apply_torch_safe_load_patches()
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

    avg_seconds = (t1 - t0) / float(iters)
    return round(avg_seconds * 1000.0, 2), round(1.0 / avg_seconds, 2) if avg_seconds > 0 else 0.0


def _bench_onnx(onnx_path: Path, *, imgsz: int, providers: list[str]) -> tuple[float, float]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=list(providers))
    except Exception:
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    sample = np.random.randn(1, 3, int(imgsz), int(imgsz)).astype(np.float32)
    warmup = 5
    iters = 50 if "CUDAExecutionProvider" in (session.get_providers() or []) else 20
    for _ in range(warmup):
        _ = session.run(None, {input_name: sample})

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = session.run(None, {input_name: sample})
    t1 = time.perf_counter()
    avg_seconds = (t1 - t0) / float(iters)
    return round(avg_seconds * 1000.0, 2), round(1.0 / avg_seconds, 2) if avg_seconds > 0 else 0.0


def record_failure(job_id: str, error: BaseException | str) -> Dict[str, Any] | None:
    """Record an adapter/runner failure without allowing status handling to escape."""

    message = str(error)
    if not isinstance(error, str):
        message = f"{type(error).__name__}: {error}"
    try:
        return update_job(
            job_id,
            {
                "status": "failed",
                "progress": 100,
                "error_message": message,
            },
            log=message,
        )
    except Exception:
        return None


def _discover_output(job_root: Path, exported: Any) -> Path | None:
    exported_path: Path | None = None
    try:
        if exported:
            exported_path = Path(str(exported)).resolve(strict=False)
    except (OSError, TypeError, ValueError):
        exported_path = None

    candidates: list[Path] = []
    try:
        candidates.extend(path for path in job_root.glob("*.onnx") if path.is_file())
    except OSError:
        pass
    if exported_path and exported_path.exists() and exported_path.is_file():
        candidates.append(exported_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run(job_id: str, *, opset: int | None, dynamic: bool) -> None:
    data = read_job(job_id)
    if str(data.get("status") or "").strip().lower() in {
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    }:
        return
    update_job(job_id, {"status": JobStatus.RUNNING, "progress": 5}, log="加载模型...")

    model_input = input_path(job_id)
    if not model_input.exists() or not model_input.is_file():
        raise FileNotFoundError("input.pt not found")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics not installed: {type(exc).__name__}: {exc}") from exc

    apply_torch_safe_load_patches()
    device, providers = _select_device()
    model = YOLO(str(model_input))
    update_job(job_id, {"progress": 25}, log="开始导出 ONNX...")

    export_kwargs: Dict[str, Any] = {"dynamic": bool(dynamic), "device": 0 if device == "cuda" else "cpu"}
    if opset is not None:
        export_kwargs["opset"] = int(opset)
    exported = model.export(format="onnx", **export_kwargs)
    update_job(job_id, {"progress": 85}, log="写入输出文件...")

    job_root = model_input.parent
    output = job_root / "output.onnx"
    if not output.exists():
        discovered = _discover_output(job_root, exported)
        if discovered and discovered.resolve(strict=False) != output.resolve(strict=False):
            shutil.copy2(discovered, output)
    if not output.exists() or not output.is_file():
        raise RuntimeError("ONNX export failed: output file not found")

    performance: Dict[str, Any] = {}
    try:
        update_job(job_id, {"progress": 92}, log="计算模型大小与性能指标...")
        imgsz = 640
        performance = {
            "device": device,
            "onnx_provider": providers[0] if providers else "CPUExecutionProvider",
            "imgsz": imgsz,
            "pt": {"size_mb": _file_size_mb(model_input)},
            "onnx": {"size_mb": _file_size_mb(output)},
        }
        if device == "cuda" and "CUDAExecutionProvider" not in providers:
            update_job(job_id, {}, log="ONNX Runtime CUDA provider 不可用，ONNX 性能测试使用 CPU")
        try:
            latency, throughput = _bench_torch_yolo(model_input, imgsz=imgsz, device=device)
            performance["pt"].update({"latency_ms": latency, "throughput_img_s": throughput})
        except Exception as exc:
            update_job(job_id, {}, log=f"PT 性能测试失败: {type(exc).__name__}: {exc}")
        try:
            latency, throughput = _bench_onnx(output, imgsz=imgsz, providers=providers)
            performance["onnx"].update({"latency_ms": latency, "throughput_img_s": throughput})
        except Exception as exc:
            update_job(job_id, {}, log=f"ONNX 性能测试失败: {type(exc).__name__}: {exc}")
    except Exception as exc:
        update_job(job_id, {}, log=f"性能统计失败(已忽略): {type(exc).__name__}: {exc}")

    update_job(
        job_id,
        {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "performance": performance or None,
            "output_filename": output.name,
            "error_message": None,
        },
        log="转换完成",
    )


def run_job(job_id: str, *, opset: int | None = None, dynamic: bool | None = None) -> None:
    """Execute one conversion and isolate all expected failures in its job state."""

    try:
        data = read_job(job_id)
        persisted_opset = data.get("opset")
        persisted_dynamic = data.get("dynamic", True)
        effective_opset = (
            int(opset)
            if opset is not None
            else (int(persisted_opset) if persisted_opset is not None else None)
        )
        effective_dynamic = bool(dynamic) if dynamic is not None else bool(persisted_dynamic)
        _run(job_id, opset=effective_opset, dynamic=effective_dynamic)
    except Exception as exc:
        record_failure(job_id, exc)


__all__ = ["record_failure", "run_job"]
