from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from train_platform.platform.runtime import gpu_probe as probe


GPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
MIB = 1024 * 1024


def nvml(**overrides):
    values = dict(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda _: "handle",
        nvmlDeviceGetName=lambda _: b"Test GPU", nvmlDeviceGetUUID=lambda _: GPU.encode(),
        nvmlDeviceGetPciInfo=lambda _: SimpleNamespace(busId=b"0000:01:00.0"),
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(total=10 * MIB + 99, used=MIB + 1, free=8 * MIB + 99),
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=12),
        nvmlDeviceGetComputeMode=lambda _: 0,
        nvmlDeviceGetMigMode=lambda _: (0, 0),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def fail(*args, **kwargs):
    raise RuntimeError("permission denied")


def test_nvml_rounds_memory_and_reads_actual_free(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml())
    result = probe.probe_gpus()
    assert result.status == "success" and result.complete
    device, = result.devices
    assert device.gpu_uuid == GPU
    assert (device.memory_total_mib, device.memory_used_mib, device.memory_free_mib) == (10, 2, 8)
    assert device.pci_bus_id == "0000:01:00.0"


def test_empty_nvml_inventory_does_not_fallback(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetCount=lambda: 0))
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected smi"))
    result = probe.probe_gpus()
    assert result.status == "empty" and result.complete and result.devices == []


def test_memory_failure_is_null_not_zero(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetMemoryInfo=fail))
    result = probe.probe_gpus()
    device, = result.devices
    assert device.memory_total_mib is device.memory_used_mib is device.memory_free_mib is None
    assert "permission denied" in device.error


def test_optional_field_failure_does_not_make_enumeration_incomplete(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetUtilizationRates=fail))
    result = probe.probe_gpus()
    assert result.complete
    assert result.devices[0].utilization_percent is None
    assert result.devices[0].error


@pytest.mark.parametrize("uuid", [None, "GPU-short", "MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"])
def test_missing_physical_uuid_cannot_be_registered(monkeypatch, uuid):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetUUID=lambda _: uuid))
    result = probe.probe_gpus()
    assert not result.complete
    assert result.devices[0].gpu_uuid is None
    assert result.devices[0].error


def test_mig_parent_is_single_device(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetMigMode=lambda _: (1, 1)))
    result = probe.probe_gpus()
    assert len(result.devices) == 1
    assert result.devices[0].gpu_uuid == GPU
    assert result.devices[0].mig_mode == "enabled"


def test_smi_fallback_parses_units_and_unknowns(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", None)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr="", stdout=f'0, "Test, GPU", {GPU}, 0000:01:00.0, 2 GiB, 1.1 MiB, 1025 KiB, N/A, Default, Disabled\n'))
    result = probe.probe_gpus()
    assert result.source == "nvidia-smi"
    device, = result.devices
    assert device.name == "Test, GPU"
    assert (device.memory_total_mib, device.memory_used_mib, device.memory_free_mib) == (2048, 2, 1)
    assert device.utilization_percent is None


def test_probe_unavailable_distinct_from_failure(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", None)
    monkeypatch.setattr(probe.shutil, "which", lambda _: None)
    assert probe.probe_gpus().status == "unavailable"
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", fail)
    result = probe.probe_gpus()
    assert result.status == "failed" and not result.complete
    assert "permission denied" in result.error


def test_malformed_smi_output_is_not_empty_success(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", None)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stderr="", stdout="bad,row"))
    result = probe.probe_gpus()
    assert not result.complete and result.error
    assert result.status != "empty"


def test_monitoring_retains_existing_fields(monkeypatch):
    from train_platform.domains.monitoring.metrics import collector

    monkeypatch.setattr(probe, "pynvml", nvml())
    metric, = collector.get_gpu_device_metrics()
    assert set(metric) == {"gpu_index", "name", "uuid", "utilization_percent", "memory_used_mb", "memory_total_mb", "memory_percent"}
    assert metric["uuid"] == GPU
    assert metric["memory_used_mb"] == 2


def test_monitoring_can_display_gpu_without_registerable_uuid(monkeypatch):
    from train_platform.domains.monitoring.metrics import collector

    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetUUID=fail))
    metric, = collector.get_gpu_device_metrics()
    assert metric["gpu_index"] == 0
    assert metric["uuid"] is None
    assert metric["memory_total_mb"] == 10


def test_all_nvml_handles_failed_is_failed_not_empty_or_success(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(
        nvmlDeviceGetCount=lambda: 2, nvmlDeviceGetHandleByIndex=fail))
    result = probe._nvml_probe(datetime.now(timezone.utc))
    assert result.status == "failed" and not result.complete
    assert "permission denied" in result.error
    assert all(device.status == "failed" for device in result.devices)


@pytest.mark.parametrize("fallback_ok", [True, False])
def test_all_handle_failures_attempt_smi_fallback(monkeypatch, fallback_ok):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetHandleByIndex=fail))
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    calls = []

    def smi(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(
            returncode=0 if fallback_ok else 1,
            stderr="" if fallback_ok else "smi failed",
            stdout=f"0, Test GPU, {GPU}, 0000:01:00.0, 10 MiB, 2 MiB, 8 MiB, 12, Default, Disabled",
        )

    monkeypatch.setattr(probe.subprocess, "run", smi)
    result = probe.probe_gpus()
    assert calls == [1]
    if fallback_ok:
        assert result.status == "success" and result.source == "nvidia-smi"
        assert result.devices[0].gpu_uuid == GPU
    else:
        assert result.status == "failed" and not result.complete
        assert "permission denied" in result.error and "smi failed" in result.error
        assert not any(device.status in {"success", "partial"} for device in result.devices)


def test_mixed_nvml_handles_keep_good_card_and_diagnostics(monkeypatch):
    from train_platform.domains.monitoring.metrics import collector

    def handle(index):
        if index == 1:
            raise RuntimeError("GPU 1 inaccessible")
        return "handle"

    monkeypatch.setattr(probe, "pynvml", nvml(
        nvmlDeviceGetCount=lambda: 2, nvmlDeviceGetHandleByIndex=handle))
    monkeypatch.setattr(probe, "_smi_probe", lambda _: pytest.fail("partial NVML must be retained"))
    result = probe.probe_gpus()
    assert result.status == "success" and not result.complete
    assert "GPU 1 inaccessible" in result.error
    assert [device.gpu_uuid for device in result.devices if device.status == "success"] == [GPU]
    monkeypatch.setattr(collector, "probe_gpus", lambda: result)
    snapshot = collector.collect_system_snapshot()
    assert snapshot["gpu_count"] == 1 and snapshot["gpu_available"] is True


def test_monitoring_excludes_handle_failure_placeholders(monkeypatch):
    from train_platform.domains.monitoring.metrics import collector

    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetHandleByIndex=fail))
    result = probe._nvml_probe(datetime.now(timezone.utc))
    monkeypatch.setattr(collector, "probe_gpus", lambda: result)
    assert collector.get_gpu_device_metrics() == []
    snapshot = collector.collect_system_snapshot()
    assert snapshot["gpu_count"] == 0 and snapshot["gpu_available"] is False


@pytest.mark.parametrize("uuid", ["N/A", "GPU-short"])
def test_smi_missing_uuid_is_displayable_partial_observation(monkeypatch, uuid):
    from train_platform.domains.monitoring.metrics import collector

    monkeypatch.setattr(probe, "pynvml", None)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr="", stdout=f"0, Test GPU, {uuid}, N/A, 10 MiB, N/A, 8 MiB, N/A, Default, N/A"))
    result = probe.probe_gpus()
    assert result.status == "success" and not result.complete
    device, = result.devices
    assert device.status == "partial" and device.gpu_uuid is None
    assert device.error and device.memory_used_mib is None
    monkeypatch.setattr(collector, "probe_gpus", lambda: result)
    metric, = collector.get_gpu_device_metrics()
    assert metric["uuid"] is None and metric["memory_total_mb"] == 10
    assert metric["memory_used_mb"] is None


def test_smi_index_alone_is_not_an_observation(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", None)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr="", stdout="0, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A"))
    result = probe.probe_gpus()
    assert result.status == "failed" and not result.complete and result.error
    assert not any(device.status in {"success", "partial"} for device in result.devices)


def test_handle_without_readable_fields_is_failed(monkeypatch):
    monkeypatch.setattr(probe, "pynvml", nvml(
        nvmlDeviceGetName=fail, nvmlDeviceGetUUID=lambda _: "GPU-short",
        nvmlDeviceGetMemoryInfo=fail, nvmlDeviceGetPciInfo=fail,
        nvmlDeviceGetUtilizationRates=fail, nvmlDeviceGetComputeMode=fail,
        nvmlDeviceGetMigMode=fail,
    ))
    result = probe._nvml_probe(datetime.now(timezone.utc))
    assert result.status == "failed" and not result.complete
    assert result.devices[0].status == "failed"
    assert result.devices[0].gpu_uuid is None


@pytest.mark.parametrize("missing_name", [None, "N/A"])
def test_nvml_missing_name_is_null_partial_but_other_fields_survive(monkeypatch, missing_name):
    monkeypatch.setattr(probe, "pynvml", nvml(nvmlDeviceGetName=lambda _: missing_name))
    result = probe.probe_gpus()
    assert result.status == "success" and result.complete
    device, = result.devices
    assert device.name is None and device.status == "partial"
    assert "name" in device.error
    assert device.gpu_uuid == GPU and device.memory_total_mib == 10
