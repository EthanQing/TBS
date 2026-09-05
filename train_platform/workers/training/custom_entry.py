from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from train_platform.training_sdk import TrainingContext


def _load_context(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        context = json.load(handle)
    if not isinstance(context, dict):
        raise ValueError("Custom training context must be a JSON object")
    return context


def _required(context: dict[str, Any], name: str) -> Any:
    if name not in context:
        raise ValueError(f"Custom training context is missing '{name}'")
    return context[name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)

    try:
        context = _load_context(Path(args.context))
        source_root = Path(_required(context, "source_root")).resolve()
        if not source_root.is_dir():
            raise ValueError(f"Custom source root does not exist: {source_root}")

        manifest = _required(context, "manifest")
        if not isinstance(manifest, dict):
            raise ValueError("Custom training manifest must be a JSON object")
        entrypoint = manifest.get("entrypoint")
        if not isinstance(entrypoint, dict):
            raise ValueError("Custom training manifest is missing entrypoint")
        module_name = str(entrypoint.get("module") or "").strip()
        class_name = str(entrypoint.get("class") or "").strip()
        if not module_name or not class_name:
            raise ValueError("Custom training manifest entrypoint is incomplete")

        sys.path.insert(0, str(source_root))
        module = importlib.import_module(module_name)
        trainer_class = getattr(module, class_name)
        trainer = trainer_class()
        train = getattr(trainer, "train", None)
        if not callable(train):
            raise TypeError(f"Custom trainer '{module_name}.{class_name}' has no callable train(ctx)")

        ctx = TrainingContext(
            run_id=str(_required(context, "run_id")),
            dataset_path=Path(_required(context, "dataset_path")),
            output_dir=Path(_required(context, "output_dir")),
            epochs=int(_required(context, "epochs")),
            batch_size=int(_required(context, "batch_size")),
            image_size=int(_required(context, "image_size")),
            learning_rate=float(_required(context, "learning_rate")),
            optimizer=str(_required(context, "optimizer")),
            workers=int(_required(context, "workers")),
            device=str(_required(context, "device")),
            custom_args=dict(context.get("custom_args") or {}),
            _cancel_marker_path=Path(_required(context, "cancel_marker_path")),
            _event_path=Path(_required(context, "event_path")),
        )

        train(ctx)
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
