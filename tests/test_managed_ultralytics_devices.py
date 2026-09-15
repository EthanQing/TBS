import os
from types import SimpleNamespace


def test_pinned_select_device_preserves_uuid_mask(monkeypatch):
    import torch
    from ultralytics.utils.torch_utils import select_device

    mask = "GPU-11111111-1111-1111-1111-111111111111"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert select_device(torch.device("cuda", 0)) == torch.device("cuda", 0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == mask


def test_managed_rank_keeps_world_size_with_torch_device(monkeypatch):
    from train_platform.domains.training.frameworks.ultralytics_trainers import _ExecutionPathsMixin

    class Base:
        def __init__(self):
            self.world_size = 1
            self.ddp = False

    class Trainer(_ExecutionPathsMixin, Base):
        pass

    monkeypatch.setenv("TRAIN_PLATFORM_ALLOCATION_ID", "allocation")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("LOCAL_RANK", "1")
    trainer = Trainer()
    assert trainer.world_size == 2
    assert not trainer.ddp


def test_resume_keeps_local_device_and_amp(monkeypatch):
    import torch
    from train_platform.domains.training.frameworks.ultralytics_trainers import _ExecutionPathsMixin

    class Base:
        def check_resume(self, overrides):
            self.args = SimpleNamespace(device="7,8", amp=True)

    class Trainer(_ExecutionPathsMixin, Base):
        pass

    trainer = Trainer.__new__(Trainer)
    trainer.check_resume({"device": torch.device("cuda", 1), "amp": False})
    assert trainer.args.device == torch.device("cuda", 1)
    assert not trainer.args.amp
