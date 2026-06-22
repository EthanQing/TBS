from types import SimpleNamespace

import pytest


def test_yolo_worker_uses_worker_id_env(monkeypatch):
    from train_platform.workers import yolo_worker_impl

    seen = {}

    class FakeTrainingWorker:
        poll_interval = 0

        def __init__(self, *, worker_id=None, allowed_engines=None):
            self.worker_id = worker_id
            self.allowed_engines = allowed_engines
            seen["worker_id"] = worker_id
            seen["allowed_engines"] = allowed_engines

        def tick(self):
            raise KeyboardInterrupt

    class FakeConversionWorker:
        def __init__(self, *, worker_id):
            seen["conversion_worker_id"] = worker_id

        def tick(self):
            pass

    monkeypatch.setenv("WORKER_ID", "custom-yolo-worker-a")
    monkeypatch.setattr(yolo_worker_impl, "DbQueueWorker", FakeTrainingWorker)
    monkeypatch.setattr(yolo_worker_impl, "ModelConversionQueueWorker", FakeConversionWorker)
    monkeypatch.setattr(yolo_worker_impl, "settings", SimpleNamespace(ensure_dirs=lambda: None))
    monkeypatch.setattr(yolo_worker_impl, "_start_inference_worker_if_needed", lambda: None)

    with pytest.raises(KeyboardInterrupt):
        yolo_worker_impl.main()

    assert seen["worker_id"] == "custom-yolo-worker-a"
    assert seen["conversion_worker_id"] == "custom-yolo-worker-a"
    assert seen["allowed_engines"] == {"ultralytics-yolo"}
