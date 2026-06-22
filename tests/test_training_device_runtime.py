import pytest


def test_build_device_runtime_maps_host_gpu_to_container_local_index():
    from train_platform.utils.training_params import build_device_runtime

    runtime = build_device_runtime("1", visible_host_gpu_ids=[1])

    assert runtime == {
        "requested": "1",
        "runtime_device": "0",
        "cuda_visible_devices": "0",
    }


def test_build_device_runtime_maps_multi_gpu_visible_subset():
    from train_platform.utils.training_params import build_device_runtime

    runtime = build_device_runtime("1,3", visible_host_gpu_ids=[1, 3])

    assert runtime["requested"] == "1,3"
    assert runtime["runtime_device"] == "0,1"
    assert runtime["cuda_visible_devices"] == "0,1"


def test_build_device_runtime_preserves_requested_gpu_order_after_mapping():
    from train_platform.utils.training_params import build_device_runtime

    runtime = build_device_runtime("3,1", visible_host_gpu_ids=[1, 3])

    assert runtime["requested"] == "3,1"
    assert runtime["runtime_device"] == "0,1"
    assert runtime["cuda_visible_devices"] == "1,0"


def test_build_device_runtime_rejects_gpu_not_visible_to_worker():
    from train_platform.utils.training_params import build_device_runtime

    with pytest.raises(ValueError, match="not visible"):
        build_device_runtime("1", visible_host_gpu_ids=[0])


def test_worker_can_run_device_matches_explicit_gpu_only():
    from train_platform.utils.training_params import worker_can_run_device

    assert worker_can_run_device("1", [1]) is True
    assert worker_can_run_device("0", [1]) is False
    assert worker_can_run_device("auto", [1]) is True
    assert worker_can_run_device("cpu", [1]) is True


def test_parse_visible_host_gpu_ids_ignores_non_numeric_runtime_values(monkeypatch):
    from train_platform.utils.training_params import parse_visible_host_gpu_ids

    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-abc123")

    assert parse_visible_host_gpu_ids() is None


def test_parse_visible_host_gpu_ids_treats_none_as_empty(monkeypatch):
    from train_platform.utils.training_params import parse_visible_host_gpu_ids

    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")

    assert parse_visible_host_gpu_ids() == []
