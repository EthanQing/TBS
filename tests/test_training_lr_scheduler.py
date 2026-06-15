import pytest
from pydantic import ValidationError as PydanticValidationError


def test_training_parameters_lr_scheduler_defaults_to_linear():
    from train_platform.schemas.v3.training_runs import TrainingRunParametersIn

    params = TrainingRunParametersIn()

    assert params.lr_scheduler == "linear"


@pytest.mark.parametrize("scheduler", ["linear", "cosine"])
def test_training_parameters_accept_supported_lr_schedulers(scheduler):
    from train_platform.schemas.v3.training_runs import TrainingRunParametersIn

    params = TrainingRunParametersIn(lr_scheduler=scheduler)

    assert params.lr_scheduler == scheduler


def test_training_parameters_reject_unknown_lr_scheduler():
    from train_platform.schemas.v3.training_runs import TrainingRunParametersIn

    with pytest.raises(PydanticValidationError):
        TrainingRunParametersIn(lr_scheduler="step")


def test_validate_training_params_normalizes_lr_scheduler_default():
    from train_platform.utils.training_params import validate_training_params_for_engine

    params = validate_training_params_for_engine("ultralytics-yolo", {"batch_size": 16, "device": "auto"})

    assert params["lr_scheduler"] == "linear"


@pytest.mark.parametrize(
    ("scheduler", "expected"),
    [
        ("linear", False),
        ("cosine", True),
    ],
)
def test_ultralytics_lr_scheduler_mapping(scheduler, expected):
    from train_platform.training.plugins.ultralytics_yolo import _lr_scheduler_to_ultralytics_args

    assert _lr_scheduler_to_ultralytics_args(scheduler) == {"cos_lr": expected}


def test_ultralytics_pin_memory_wrapper_overrides_existing_value():
    from train_platform.training.plugins.ultralytics_yolo import _wrap_build_dataloader_pin_memory

    seen = {}

    def fake_build_dataloader(*args, **kwargs):
        seen["pin_memory"] = kwargs.get("pin_memory")
        return "loader"

    wrapped = _wrap_build_dataloader_pin_memory(fake_build_dataloader, False)

    assert wrapped(pin_memory=True) == "loader"
    assert seen["pin_memory"] is False


def test_paddle_linear_scheduler_keeps_existing_schedulers():
    from train_platform.training.plugins.paddle_det import _apply_lr_scheduler_to_cfg

    cfg = {
        "LearningRate": {
            "base_lr": 0.01,
            "schedulers": [
                {"name": "PiecewiseDecay", "milestones": [30, 60]},
                {"name": "LinearWarmup", "epochs": 1},
            ],
        }
    }

    changed = _apply_lr_scheduler_to_cfg(cfg, "linear", epochs=100)

    assert changed is False
    assert cfg["LearningRate"]["schedulers"][0]["name"] == "PiecewiseDecay"


def test_paddle_cosine_scheduler_preserves_warmup_and_replaces_main_scheduler():
    from train_platform.training.plugins.paddle_det import _apply_lr_scheduler_to_cfg

    cfg = {
        "LearningRate": {
            "base_lr": 0.01,
            "schedulers": [
                {"name": "PiecewiseDecay", "milestones": [30, 60]},
                {"name": "LinearWarmup", "epochs": 1},
            ],
        }
    }

    changed = _apply_lr_scheduler_to_cfg(cfg, "cosine", epochs=100)

    assert changed is True
    assert cfg["LearningRate"]["schedulers"] == [
        {"name": "CosineDecay", "max_epochs": 100},
        {"name": "LinearWarmup", "epochs": 1},
    ]
