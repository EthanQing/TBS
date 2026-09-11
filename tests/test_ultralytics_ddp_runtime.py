import json
from types import SimpleNamespace

import pytest

from train_platform.domains.training.frameworks.ultralytics_yolo import _collect_metrics, _register_epoch_callbacks
from train_platform.platform.runtime.ultralytics_ddp import MetricsJSONLReader


def test_metrics_reader_keeps_partial_line_and_filters_attempt(tmp_path):
    path = tmp_path / "metrics.jsonl"
    event = {"type": "epoch_metrics", "run_id": "run", "attempt_id": "attempt", "epoch": 2, "metrics": {"score": 0.5}}
    encoded = json.dumps(event).encode()
    path.write_bytes(encoded[:20])
    received = []
    reader = MetricsJSONLReader(path, run_id="run", attempt_id="attempt")
    reader.read(lambda epoch, metrics: received.append((epoch, metrics)))
    assert received == []
    with path.open("ab") as file:
        file.write(encoded[20:] + b"\n")
        file.write(json.dumps({**event, "attempt_id": "old"}).encode() + b"\n")
        file.write(b"[]\n")
        file.write(json.dumps({**event, "epoch": -1}).encode() + b"\n")
        file.write(json.dumps({**event, "epoch": "3"}).encode() + b"\n")
    reader.read(lambda epoch, metrics: received.append((epoch, metrics)))
    assert received == [(2, {"score": 0.5})]


def test_epoch_callback_reports_once_and_skips_final_eval():
    callbacks = {}
    class Model:
        def add_callback(self, name, callback):
            callbacks[name] = callback
    trainer = SimpleNamespace(epoch=0, metrics={"map": 0.4}, tloss=[1.2], lr={"lr/pg0": 0.01})
    trainer.label_loss_items = lambda loss, prefix: {f"{prefix}/loss": loss[0]}
    received = []
    _register_epoch_callbacks(Model(), lambda epoch, metrics: received.append((epoch, metrics)))
    callbacks["on_train_epoch_start"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    callbacks["on_fit_epoch_end"](trainer)
    assert received == [(0, _collect_metrics(trainer))]
    assert received[0][1] == {"map": 0.4, "train/loss": 1.2, "lr/pg0": 0.01}


@pytest.mark.parametrize("mask", ["2,5", "0,1"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("model_type", ["yolo", "rtdetr"])
def test_rank_uses_frozen_mask_and_only_rank_zero_emits(tmp_path, monkeypatch, mask, rank, model_type):
    import os
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer
    from train_platform.workers.training import ultralytics_ddp_entry_impl as entry
    from train_platform.domains.training.frameworks import ultralytics_yolo as adapter

    metrics = tmp_path / "metrics.jsonl"
    metrics.touch()
    context = {
        "run_id": "run", "attempt_id": "attempt", "execution_owner": {},
        "world_size": 2, "cuda_visible_devices": mask, "requested_device": "2,5",
        "pin_memory": False, "model_type": model_type, "model_path": "prepared.pt",
        "processes_dir": str(tmp_path / "processes"), "metrics_path": str(metrics),
        "train_args": {"device": mask, "batch": 8, "amp": False, "resume": True},
    }
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    for key, value in {"RANK": rank, "LOCAL_RANK": rank, "WORLD_SIZE": 2}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    bound = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda local: bound.append((local, os.environ["CUDA_VISIBLE_DEVICES"])))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    destroyed = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: destroyed.append(True))
    monkeypatch.setattr(adapter, "_patch_ultralytics_dataloader_pin_memory", lambda enabled: None)
    from train_platform.platform.runtime import ultralytics as compatibility
    monkeypatch.setattr(compatibility, "apply_torch_safe_load_patches", lambda: None)
    settings_updates = []
    monkeypatch.setattr(ultralytics, "settings", SimpleNamespace(update=settings_updates.append))

    class Model:
        def __init__(self, model_path):
            assert model_path == "prepared.pt"
            assert bound == [(rank, mask)]
            self.callbacks = {}

        def _smart_load(self, key):
            return DetectionTrainer

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **kwargs):
            assert kwargs["device"] == mask
            assert kwargs["batch"] == 8
            assert kwargs["amp"] is False
            assert kwargs["resume"] is True
            trainer = SimpleNamespace(epoch=0, metrics={"map": 0.4}, tloss=None, lr={"lr/pg0": 0.01})
            self.callbacks["on_train_epoch_start"](trainer)
            self.callbacks["on_fit_epoch_end"](trainer)
            self.callbacks["on_fit_epoch_end"](trainer)

    monkeypatch.setattr(ultralytics, "YOLO" if model_type == "yolo" else "RTDETR", Model)
    assert entry.main(["--context", str(path)]) == 0
    assert destroyed == [True]
    assert settings_updates == [{"mlflow": False}]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == mask
    events = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(events) == (1 if rank == 0 else 0)
    if events:
        assert events[0]["epoch"] == 0
        assert events[0]["attempt_id"] == "attempt"
