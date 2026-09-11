from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from train_platform.domains.training.execution_paths import prepare_ultralytics_execution
from train_platform.domains.training.frameworks import ultralytics_yolo
from train_platform.domains.training.frameworks.contract import TrainingCallbacks, TrainingExecutionSpec


@pytest.mark.parametrize(
    "resume_mode,recorded,device",
    [
        ("fresh", False, "0"),
        ("fresh", False, "0,1"),
        ("same", False, "cpu"),
        ("same", True, "cpu"),
        ("cross", False, "cpu"),
        ("cross", True, "0,1"),
    ],
)
def test_adapter_keeps_current_execution_paths(tmp_path, monkeypatch, resume_mode, recorded, device):
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer

    from train_platform.domains.training.frameworks.ultralytics_trainers import PlatformDetectionTrainer

    training_dir = tmp_path / "training"
    run_root = training_dir / "current"
    source_root = training_dir / "source" if resume_mode == "cross" else run_root
    checkpoint = None
    if resume_mode != "fresh":
        source_output = prepare_ultralytics_execution(source_root).output_dir if recorded else source_root
        checkpoint = source_output / "weights" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"original checkpoint")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source_yaml = dataset / "data.yaml"
    source_yaml.write_text("path: old-dataset\ntrain: images/train\nval: images/val\nnames: [object]\n", encoding="utf-8")
    original_yaml = source_yaml.read_bytes()
    calls = {}

    class FakeYOLO:
        def __init__(self, model_path):
            calls["model_path"] = model_path

        def _smart_load(self, key):
            assert key == "trainer"
            return DetectionTrainer

        def add_callback(self, *args):
            pass

        def train(self, **kwargs):
            calls["train_args"] = kwargs

        def val(self):
            pass

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(ultralytics, "settings", SimpleNamespace(update=lambda values: None))
    monkeypatch.setattr(ultralytics_yolo, "settings", SimpleNamespace(training_dir=training_dir))
    monkeypatch.setattr(ultralytics_yolo, "apply_torch_safe_load_patches", lambda: None)
    monkeypatch.setattr(ultralytics_yolo, "_ensure_amp_check_weight", lambda: True)
    monkeypatch.setattr(ultralytics_yolo, "_patch_ultralytics_dataloader_pin_memory", lambda value: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    spec = TrainingExecutionSpec(
        run_id="current", dataset_path=dataset, dataset_name="dataset", run_dir=run_root,
        engine="ultralytics-yolo", family="YOLOv8", variant="yolov8n", epochs=10,
        batch_size=2, image_size=64, learning_rate=0.001, lr_scheduler="cosine", patience=5,
        requested_device=device, runtime_device=device, workers=1, optimizer="SGD",
        use_pretrained=False, resume_training=resume_mode != "fresh",
        resume_job_id="source" if resume_mode == "cross" else None,
    )
    callbacks = TrainingCallbacks(
        cancel_requested=lambda: False, upsert_epoch_metrics=lambda *args: None,
        report_artifact=lambda *args: None,
    )

    ultralytics_yolo.UltralyticsYOLOTrainer().run(spec, callbacks)

    args = calls["train_args"]
    assert args["trainer"] is PlatformDetectionTrainer
    assert Path(args["data"]) == run_root / "runtime" / "data.runtime.yaml"
    assert Path(args["data"]).is_absolute()
    assert args["project"] == str(run_root)
    assert args["name"] == "output"
    assert args["save_dir"] == str(run_root / "output")
    assert args["exist_ok"] is True
    assert (run_root / "runtime" / "layout.json").is_file()
    assert (run_root / "logs").is_dir()
    assert not (run_root / "output" / "data.runtime.yaml").exists()
    runtime_data = yaml.safe_load(Path(args["data"]).read_text(encoding="utf-8"))
    assert runtime_data["path"] == str(dataset.resolve())
    assert runtime_data["train"] == "images/train"
    assert source_yaml.read_bytes() == original_yaml
    if checkpoint is not None:
        assert calls["model_path"] == str(checkpoint)
        assert args["resume"] is True
        assert checkpoint.read_bytes() == b"original checkpoint"
    else:
        assert "resume" not in args
