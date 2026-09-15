from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_context(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("distributed context must be an object")
    mask = str(value.get("cuda_visible_devices") or "").strip()
    if not mask:
        raise ValueError("distributed context is missing CUDA_VISIBLE_DEVICES")
    from train_platform.platform.runtime import process_scope

    owner = value.get("execution_owner")
    expected_scope = owner.get("process_scope") if isinstance(owner, dict) else None
    scope_status = process_scope.compare_process_scope(expected_scope, process_scope.get_process_scope())
    if scope_status != "same":
        raise RuntimeError(f"distributed rank process scope is {scope_status}")
    os.environ["CUDA_VISIBLE_DEVICES"] = mask
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args(argv)
    context = _load_context(args.context.resolve(strict=True))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if (
        world_size < 2
        or world_size != int(context["world_size"])
        or rank != local_rank
        or not 0 <= local_rank < world_size
    ):
        raise RuntimeError("torchrun rank environment does not match distributed context")

    from train_platform.platform.runtime.ultralytics_ddp import process_identity, register_process

    identity = process_identity(
        os.getpid(), role="rank", rank=rank, local_rank=local_rank,
        run_id=context["run_id"], attempt_id=context["attempt_id"],
        execution_owner=context.get("execution_owner", {}),
    )
    register_process(Path(context["processes_dir"]), f"rank-{rank}", **identity)
    print(
        "[ultralytics-ddp] "
        f"run_id={context['run_id']} rank={rank} local_rank={local_rank} "
        f"requested_device={context['requested_device']} "
        f"cuda_visible_devices={os.environ['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in distributed rank")
    torch.cuda.set_device(local_rank)

    from train_platform.domains.training.frameworks.ultralytics_trainers import platform_trainer_for
    from train_platform.domains.training.frameworks.ultralytics_yolo import (
        _collect_metrics,
        _patch_ultralytics_dataloader_pin_memory,
        _register_epoch_callbacks,
    )
    from train_platform.platform.runtime.ultralytics import apply_torch_safe_load_patches

    apply_torch_safe_load_patches()
    _patch_ultralytics_dataloader_pin_memory(bool(context["pin_memory"]))
    try:
        from ultralytics import RTDETR, YOLO, settings as ultralytics_settings

        ultralytics_settings.update({"mlflow": False})
        loaders = {"yolo": YOLO, "rtdetr": RTDETR}
        try:
            loader = loaders[context["model_type"]]
        except KeyError as exc:
            raise ValueError(f"unsupported model type: {context['model_type']!r}") from exc
        model = loader(context["model_path"])
        trainer_class = platform_trainer_for(model._smart_load("trainer"))
        metrics_file = None
        if rank == 0:
            metrics_file = Path(context["metrics_path"]).open("a", encoding="utf-8", buffering=1)

        def emit(epoch: int, metrics: dict[str, float]) -> None:
            if metrics_file is None:
                return
            metrics_file.write(json.dumps({
                "type": "epoch_metrics", "run_id": context["run_id"],
                "attempt_id": context["attempt_id"], "epoch": epoch, "metrics": metrics,
            }, ensure_ascii=False) + "\n")
            metrics_file.flush()

        _register_epoch_callbacks(model, emit, collect=_collect_metrics)
        train_args = dict(context["train_args"])
        train_args["device"] = str(context["cuda_visible_devices"])
        try:
            model.train(trainer=trainer_class, **train_args)
            return 0
        finally:
            if metrics_file is not None:
                metrics_file.close()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


__all__ = ["main"]
