from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_platform.domains.training.execution_paths import (
    prepare_ultralytics_execution,
    resolve_ultralytics_resume_checkpoint,
)
from train_platform.domains.training.frameworks.contract import TrainingCallbacks, TrainingExecutionSpec
from train_platform.domains.training.frameworks import ultralytics_yolo as adapter


@pytest.fixture
def execution(tmp_path, monkeypatch):
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer
    from train_platform.platform.runtime import ultralytics_ddp

    training_dir = tmp_path / "training"
    run_root = training_dir / "run"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [object]\n")
    monkeypatch.setattr(adapter, "settings", SimpleNamespace(training_dir=training_dir, temp_dir=tmp_path / "temp"))
    monkeypatch.setattr(adapter, "apply_torch_safe_load_patches", lambda: None)
    monkeypatch.setattr(adapter, "_ensure_amp_check_weight", lambda: True)
    monkeypatch.setattr(adapter, "_patch_ultralytics_dataloader_pin_memory", lambda enabled: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(ultralytics, "settings", SimpleNamespace(update=lambda values: None))
    state = SimpleNamespace(loads=[], trains=[], validate=lambda path, args: None)

    class Model:
        def __init__(self, model_path):
            self.path = str(model_path)
            state.loads.append(self.path)
            self.ckpt_path = self.path if Path(self.path).is_file() else None
            self.ckpt = {"epoch": 1, "train_results": {"epoch": [1, 2], "metric": [0.1, 0.2]}}

        def _smart_load(self, name):
            return DetectionTrainer

        def add_callback(self, *args):
            pass

        def train(self, **kwargs):
            state.validate(self.path, kwargs)
            state.trains.append((self.path, kwargs))

        def val(self, **kwargs):
            pass

    def ddp(context, **kwargs):
        state.validate(context["model_path"], context["train_args"])
        state.trains.append((context["model_path"], context["train_args"]))

    monkeypatch.setattr(ultralytics, "YOLO", Model)
    monkeypatch.setattr(ultralytics_ddp, "run_ultralytics_ddp", ddp)
    spec = TrainingExecutionSpec(
        run_id="run", dataset_path=dataset, dataset_name="dataset", run_dir=run_root,
        engine="ultralytics-yolo", family="yolo", variant="yolov8n", epochs=4,
        batch_size=2, image_size=64, learning_rate=0.001, lr_scheduler="cosine", patience=1,
        requested_device="cpu", runtime_device="cpu", workers=1, optimizer="SGD", use_pretrained=False,
    )
    callbacks = TrainingCallbacks(cancel_requested=lambda: False, upsert_epoch_metrics=lambda *args: None,
                                 report_artifact=lambda *args: None)
    return spec, callbacks, state


@pytest.mark.parametrize("device", ["cpu", "0,1"])
@pytest.mark.parametrize("use_output_weight", [False, True])
def test_fresh_start_resets_all_output_and_stages_its_model_input(execution, device, use_output_weight):
    spec, callbacks, state = execution
    paths = prepare_ultralytics_execution(spec.run_dir)
    old_files = ["weights/best.pt", "weights/last.pt", "weights/epoch2.pt", "weights/best.onnx",
                 "exports/custom.onnx", "results.csv", "args.yaml", "results.png", "other/nested.log"]
    for name in old_files:
        path = paths.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old framework output")
    (paths.runtime_dir / "keep").write_text("runtime")
    (paths.logs_dir / "keep").write_text("logs")
    legacy = spec.run_dir / "weights/last.pt"
    legacy.parent.mkdir()
    legacy.write_bytes(b"older legacy checkpoint")
    spec = replace(spec, requested_device=device, runtime_device=device,
                   use_pretrained=use_output_weight,
                   pretrained_model_path=str(paths.weights_dir / "best.pt") if use_output_weight else None)

    def validate(model_path, args):
        assert list(paths.output_dir.iterdir()) == []
        assert (paths.runtime_dir / "keep").read_text() == "runtime"
        assert (paths.logs_dir / "keep").read_text() == "logs"
        if use_output_weight:
            saved_input = Path(model_path)
            assert paths.runtime_dir in saved_input.parents
            assert saved_input.read_bytes() == b"old framework output"
            assert Path(args["pretrained"]) == saved_input

    state.validate = validate
    adapter.UltralyticsYOLOTrainer().run(spec, callbacks)
    assert len(state.trains) == 1
    assert resolve_ultralytics_resume_checkpoint(spec.run_dir) is None
    assert legacy.read_bytes() == b"older legacy checkpoint"
    # Even before this fresh execution emits a checkpoint, API/adapter resume must not restart old training.
    with pytest.raises(ValueError, match="resume|checkpoint"):
        adapter.UltralyticsYOLOTrainer().run(replace(spec, resume_training=True), callbacks)
    assert len(state.trains) == 1
    current = paths.weights_dir / "last.pt"
    current.parent.mkdir()
    current.write_bytes(b"new checkpoint")
    assert resolve_ultralytics_resume_checkpoint(spec.run_dir) == current


@pytest.mark.parametrize("cross_task", [False, True])
def test_explicit_legacy_resume_source_survives_layout_migration(execution, cross_task):
    spec, callbacks, state = execution
    source_root = spec.run_dir.parent / "source" if cross_task else spec.run_dir
    last = source_root / "weights/last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"selected checkpoint")
    best = last.with_name("best.pt")
    best.write_bytes(b"historical best")
    spec = replace(spec, resume_training=True, resume_job_id="source" if cross_task else None)
    adapter.UltralyticsYOLOTrainer().run(spec, callbacks)
    output = spec.run_dir / "output"
    assert (output / "weights/best.pt").read_bytes() == b"historical best"
    assert (output / "results.csv").is_file()
    assert resolve_ultralytics_resume_checkpoint(spec.run_dir) == last
    # Layout preparation must not erase the explicitly selected resume source.
    prepare_ultralytics_execution(spec.run_dir)
    assert resolve_ultralytics_resume_checkpoint(spec.run_dir) == last
    current = output / "weights/last.pt"
    current.write_bytes(b"continued checkpoint")
    assert resolve_ultralytics_resume_checkpoint(spec.run_dir) == current
    assert last.read_bytes() == b"selected checkpoint"


def test_reset_rejects_output_redirect_to_logs(tmp_path):
    from train_platform.domains.training.execution_paths import reset_ultralytics_output

    logs = tmp_path / "logs"
    logs.mkdir()
    marker = logs / "train.stdout.log"
    marker.write_text("keep")
    try:
        (tmp_path / "output").symlink_to(logs, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink privilege unavailable")
    with pytest.raises(ValueError, match="redirected"):
        reset_ultralytics_output(tmp_path)
    assert marker.read_text() == "keep"


def test_failed_reset_blocks_old_checkpoint_fallback(tmp_path, monkeypatch):
    from train_platform.domains.training import execution_paths as paths

    output = paths.prepare_ultralytics_execution(tmp_path).output_dir
    (output / "weights").mkdir()
    (output / "weights" / "last.pt").write_bytes(b"old")
    monkeypatch.setattr(paths.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(OSError, match="busy"):
        paths.reset_ultralytics_output(tmp_path)
    assert paths.resolve_ultralytics_resume_checkpoint(tmp_path) is None
    with pytest.raises(ValueError, match="incomplete"):
        paths.read_ultralytics_paths(tmp_path)


def test_corrupt_layout_does_not_discard_fresh_boundary(tmp_path):
    from train_platform.domains.training import execution_paths as paths

    manifest = paths.prepare_ultralytics_execution(tmp_path).layout_manifest
    manifest.write_text("{")
    with pytest.raises(ValueError):
        paths.prepare_ultralytics_execution(tmp_path)
    with pytest.raises(ValueError):
        paths.resolve_ultralytics_resume_checkpoint(tmp_path)
