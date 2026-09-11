from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


LAYOUT_VERSION = 1
ULTRALYTICS_ENGINE = "ultralytics-yolo"
ULTRALYTICS_OUTPUT_DIR = "output"


@dataclass(frozen=True)
class TrainingExecutionPaths:
    run_root: Path
    runtime_dir: Path
    logs_dir: Path
    output_dir: Path

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    @property
    def data_runtime_yaml(self) -> Path:
        return self.runtime_dir / "data.runtime.yaml"

    @property
    def layout_manifest(self) -> Path:
        return self.runtime_dir / "layout.json"

    @property
    def stdout_log(self) -> Path:
        return self.logs_dir / "train.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.logs_dir / "train.stderr.log"

    def output_file(self, relative_path: str) -> Path:
        candidate = (self.output_dir / relative_path).resolve(strict=False)
        output_root = self.output_dir.resolve(strict=False)
        if candidate != output_root and output_root not in candidate.parents:
            raise ValueError("output path escapes the framework output directory")
        return candidate


def _base_paths(run_root: Path, output_dir: Path) -> TrainingExecutionPaths:
    root = Path(run_root).resolve(strict=False)
    return TrainingExecutionPaths(
        run_root=root,
        runtime_dir=root / "runtime",
        logs_dir=root / "logs",
        output_dir=output_dir,
    )


def new_ultralytics_paths(run_root: Path) -> TrainingExecutionPaths:
    root = Path(run_root).resolve(strict=False)
    output = (root / ULTRALYTICS_OUTPUT_DIR).resolve(strict=False)
    if root not in output.parents:
        raise ValueError("Ultralytics output directory escapes the run root")
    return _base_paths(root, output)


def _layout_output_dir(run_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("training layout output_dir must be a non-empty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    relative = Path(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise ValueError("training layout output_dir must be confined to the run root")
    root = run_root.resolve(strict=False)
    output = (root / relative).resolve(strict=False)
    if output != root and root not in output.parents:
        raise ValueError("training layout output_dir escapes the run root")
    return output


def read_ultralytics_paths(run_root: Path) -> TrainingExecutionPaths:
    """Resolve recorded output layout, or the legacy run-root layout when unrecorded."""

    root = Path(run_root).resolve(strict=False)
    manifest = root / "runtime" / "layout.json"
    if not manifest.exists():
        return _base_paths(root, root)
    try:
        layout = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid training layout manifest: {manifest}") from exc
    if not isinstance(layout, dict):
        raise ValueError(f"invalid training layout manifest: {manifest}")
    if layout.get("version") != LAYOUT_VERSION:
        raise ValueError(f"unsupported training layout version: {layout.get('version')!r}")
    if layout.get("engine") != ULTRALYTICS_ENGINE:
        raise ValueError(f"unexpected training layout engine: {layout.get('engine')!r}")
    return _base_paths(root, _layout_output_dir(root, layout.get("output_dir")))


def prepare_ultralytics_execution(run_root: Path) -> TrainingExecutionPaths:
    paths = new_ultralytics_paths(run_root)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths.layout_manifest
    temporary = paths.runtime_dir / "layout.json.tmp"
    temporary.write_text(
        json.dumps(
            {"version": LAYOUT_VERSION, "engine": ULTRALYTICS_ENGINE, "output_dir": ULTRALYTICS_OUTPUT_DIR},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return paths


def resolve_ultralytics_checkpoint(run_root: Path, kind: str = "last") -> Path | None:
    if kind not in {"best", "last"}:
        raise ValueError("checkpoint kind must be 'best' or 'last'")
    candidate = read_ultralytics_paths(run_root).weights_dir / f"{kind}.pt"
    return candidate if candidate.is_file() else None


def resolve_ultralytics_resume_checkpoint(run_root: Path) -> Path | None:
    paths = read_ultralytics_paths(run_root)
    candidates = [paths.weights_dir / "last.pt"]
    legacy = paths.run_root / "weights" / "last.pt"
    if legacy != candidates[0]:
        candidates.append(legacy)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


__all__ = [
    "TrainingExecutionPaths",
    "new_ultralytics_paths",
    "prepare_ultralytics_execution",
    "read_ultralytics_paths",
    "resolve_ultralytics_checkpoint",
    "resolve_ultralytics_resume_checkpoint",
]
