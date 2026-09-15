from types import SimpleNamespace

import pytest

from train_platform.domains.training.execution_paths import prepare_ultralytics_execution
from train_platform.domains.training.runs import reports


@pytest.mark.parametrize("index_fails", [False, True])
def test_report_does_not_return_old_root_weight_projection(tmp_path, monkeypatch, index_fails):
    training_dir = tmp_path / "training"
    run_root = training_dir / "run"
    prepare_ultralytics_execution(run_root)
    legacy = run_root / "weights/best.pt"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")
    result = SimpleNamespace(best_weights_path="run/weights/best.pt", last_weights_path="run/weights/last.pt",
                             model_size_mb=1, flops=1, inference_time_ms=1)
    run = SimpleNamespace(run_id="run", result=result, architecture=SimpleNamespace(engine="ultralytics-yolo"))
    db = SimpleNamespace(commit=lambda: None, rollback=lambda: None, refresh=lambda value: None)
    monkeypatch.setattr(reports, "settings", SimpleNamespace(training_dir=training_dir))
    monkeypatch.setattr(reports, "TrainingRunBenchmarkService", lambda: SimpleNamespace())
    indexed = []

    def index(db, run_id):
        indexed.append(run_id)
        if index_fails:
            raise OSError("index unavailable")
        result.best_weights_path = result.last_weights_path = None

    monkeypatch.setattr(reports, "index_completion_artifacts", index)
    if index_fails:
        with pytest.raises(reports.ValidationError, match="Current training output"):
            reports._ensure_report_artifacts(db, run)
    else:
        actual = reports._ensure_report_artifacts(db, run)
        assert actual.best_weights_path is None
        assert actual.last_weights_path is None
    assert indexed == ["run"]
    assert legacy.read_bytes() == b"legacy"
