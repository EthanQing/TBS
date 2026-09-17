from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone


def cuda_environment_fingerprint(env: dict[str, str] | None = None) -> str:
    source = env or os.environ
    payload = {
        "CUDA_DEVICE_ORDER": source.get("CUDA_DEVICE_ORDER"),
        "CUDA_VISIBLE_DEVICES": source.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": source.get("NVIDIA_VISIBLE_DEVICES"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CudaDeviceBinding:
    ordinal: int
    gpu_uuid: str | None
    pci_bus_id: str | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class CudaBindingResult:
    status: str
    sampled_at: datetime
    environment_fingerprint: str
    bindings: list[CudaDeviceBinding] = field(default_factory=list)
    error: str | None = None
    complete: bool = False


def probe_cuda_devices(*, timeout_seconds: int = 10, env: dict[str, str] | None = None) -> CudaBindingResult:
    child_env = dict(os.environ if env is None else env)
    fingerprint = cuda_environment_fingerprint(child_env)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "train_platform.platform.runtime.cuda_probe"],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        payload = json.loads(process.stdout)
        bindings = [CudaDeviceBinding(**item) for item in payload.get("bindings", [])]
        sampled_at = datetime.fromisoformat(payload["sampled_at"])
        return CudaBindingResult(
            status=payload["status"],
            sampled_at=sampled_at,
            environment_fingerprint=fingerprint,
            bindings=bindings,
            error=payload.get("error") or ((process.stderr or "").strip() or None),
            complete=bool(payload.get("complete")),
        )
    except Exception as exc:
        return CudaBindingResult("failed", datetime.now(timezone.utc), fingerprint, error=str(exc))


def enumerate_current_environment(*, timeout_seconds: int = 10) -> CudaBindingResult:
    """Enumerate the caller's inherited CUDA environment without initializing CUDA here."""
    return probe_cuda_devices(timeout_seconds=timeout_seconds)


def validate_assigned_devices(assigned_gpu_uuids: list[str], *, timeout_seconds: int = 10) -> CudaBindingResult:
    expected = [str(value).lower() for value in assigned_gpu_uuids]
    visible = [part.strip().lower() for part in (os.getenv("CUDA_VISIBLE_DEVICES") or "").split(",") if part.strip()]
    if visible != expected:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not exactly match the allocation UUID order")
    result = enumerate_current_environment(timeout_seconds=timeout_seconds)
    actual = [
        str(item.gpu_uuid).lower()
        for item in sorted(result.bindings, key=lambda item: item.ordinal)
        if item.status in {"success", "partial"} and item.gpu_uuid
    ]
    ordinals = [item.ordinal for item in sorted(result.bindings, key=lambda item: item.ordinal)]
    if result.status != "success" or not result.complete or ordinals != list(range(len(expected))) or actual != expected:
        raise RuntimeError(result.error or "CUDA devices do not match the active allocation")
    return result
