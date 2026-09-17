import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from train_platform.platform.runtime import cuda_devices


GPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_probe_inherits_gpu_environment_without_framework_import(monkeypatch):
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "success", "sampled_at": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "bindings": [{"ordinal": 0, "gpu_uuid": GPU, "pci_bus_id": "0000:01:00.0", "status": "success"}],
        }))

    monkeypatch.setattr(cuda_devices.subprocess, "run", run)
    env = {"CUDA_VISIBLE_DEVICES": GPU, "NVIDIA_VISIBLE_DEVICES": "all"}
    result = cuda_devices.probe_cuda_devices(env=env)
    assert result.bindings[0].gpu_uuid == GPU
    assert seen["env"] == env
    assert "train_platform.platform.runtime.cuda_probe" in seen["command"]
    assert result.environment_fingerprint == cuda_devices.cuda_environment_fingerprint(env)
    assert seen["timeout"] == 10 and seen["check"] is False


def test_binding_environment_fingerprint_changes_with_mask():
    assert cuda_devices.cuda_environment_fingerprint({"CUDA_VISIBLE_DEVICES": "0"}) != cuda_devices.cuda_environment_fingerprint({"CUDA_VISIBLE_DEVICES": GPU})


def test_assigned_uuid_order_is_verified(monkeypatch):
    second = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU},{second}")
    result = cuda_devices.CudaBindingResult("success", datetime.now(timezone.utc), "env", [
        cuda_devices.CudaDeviceBinding(0, second, None, "success"),
        cuda_devices.CudaDeviceBinding(1, GPU, None, "success"),
    ], complete=True)
    monkeypatch.setattr(cuda_devices, "probe_cuda_devices", lambda **kwargs: result)
    with pytest.raises(RuntimeError):
        cuda_devices.validate_assigned_devices([GPU, second])


def test_failed_binding_cannot_validate_assignment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(cuda_devices, "probe_cuda_devices", lambda **kwargs: cuda_devices.CudaBindingResult(
        "failed", datetime.now(timezone.utc), "env", error="driver failed"))
    with pytest.raises(RuntimeError):
        cuda_devices.validate_assigned_devices([GPU])


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_probe_preserves_failure_protocol(monkeypatch, failure):
    def run(command, **kwargs):
        assert kwargs["timeout"] == 3
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 3)
        return SimpleNamespace(returncode=1, stdout="bad JSON", stderr="driver error")

    monkeypatch.setattr(cuda_devices.subprocess, "run", run)
    result = cuda_devices.probe_cuda_devices(timeout_seconds=3)
    assert result.status == "failed" and not result.complete
    assert result.bindings == [] and result.error


@pytest.mark.parametrize("protected", [False, True])
def test_probe_module_runs_with_only_json_stdout(tmp_path, protected):
    root = Path(__file__).resolve().parents[1]
    runtime = root
    if protected:
        runtime = tmp_path / "runtime"
        build = subprocess.run([
            sys.executable, "-m", "train_platform.core.build_protected_runtime",
            "--output", str(runtime),
        ], cwd=root, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        module = runtime / "train_platform/platform/runtime/cuda_probe_impl"
        assert module.with_suffix(".pyc").exists() and not module.with_suffix(".py").exists()
    result = subprocess.run([sys.executable, "-m", "train_platform.platform.runtime.cuda_probe"],
        cwd=runtime, env={**os.environ, "PYTHONPATH": str(runtime)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"status", "sampled_at", "bindings", "complete", "error"}
    assert payload["status"] in {"success", "empty", "unavailable", "failed"}
    assert isinstance(payload["bindings"], list) and isinstance(payload["complete"], bool)
