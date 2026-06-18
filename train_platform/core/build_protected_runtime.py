from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PROTECTED_PATHS = ("train_platform/services", "train_platform/workers")
CYTHON_ENTRY_STUBS = (
    "train_platform/workers/worker.py",
    "train_platform/workers/yolo_worker.py",
    "train_platform/workers/paddle_worker.py",
    "train_platform/workers/inference_worker.py",
    "train_platform/workers/paddle_inference_worker.py",
    "train_platform/workers/training/train_entry.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble Cython-protected Train Platform runtime sources.")
    parser.add_argument("--source", type=Path, default=Path("train_platform"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect", action="append", default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"source package not found: {source}")

    project_root = source.parent
    setup_py = project_root / "setup.py"
    if not setup_py.exists():
        raise SystemExit(f"setup.py not found next to source package: {setup_py}")

    protected_paths = [_resolve_protected_path(source, item) for item in (args.protect or DEFAULT_PROTECTED_PATHS)]
    include_globs = _cython_include_globs(source, protected_paths)
    exclude_globs = _cython_exclude_globs()

    runtime = _build_cython_runtime(project_root, include_globs=include_globs, exclude_globs=exclude_globs)
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(runtime, output)
    _remove_build_helper(output / source.name)
    return 0


def _resolve_protected_path(source: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        candidate = (source.parent / raw).resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise SystemExit(f"protected path not found: {raw}")

    try:
        candidate.relative_to(source)
    except ValueError as e:
        raise SystemExit(f"protected path must be inside {source}: {candidate}") from e
    return candidate


def _cython_include_globs(source: Path, protected_paths: list[Path]) -> str:
    project_root = source.parent
    globs: list[str] = []
    for path in protected_paths:
        rel = path.relative_to(project_root).as_posix()
        globs.append(f"{rel}/*.py")
        globs.append(f"{rel}/**/*.py")
    return ",".join(globs)


def _cython_exclude_globs() -> str:
    existing = os.environ.get("CYTHON_EXCLUDE_GLOBS", "").strip()
    entries = [*CYTHON_ENTRY_STUBS]
    if existing:
        entries.append(existing)
    return ",".join(entries)


def _build_cython_runtime(project_root: Path, *, include_globs: str, exclude_globs: str) -> Path:
    env = os.environ.copy()
    env["CYTHON_INCLUDE_GLOBS"] = include_globs
    env["CYTHON_EXCLUDE_GLOBS"] = exclude_globs
    env.setdefault("CYTHON_NTHREADS", "0")

    cmd = [sys.executable, "setup.py", "build_ext"]
    print("+ " + " ".join(cmd), flush=True)
    print(f"CYTHON_INCLUDE_GLOBS={include_globs}", flush=True)
    print(f"CYTHON_EXCLUDE_GLOBS={exclude_globs}", flush=True)
    try:
        subprocess.run(cmd, cwd=project_root, env=env, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit("Cython protection build failed.") from e

    runtime = project_root / "build" / "runtime"
    if not runtime.exists():
        raise SystemExit(f"Cython runtime output not found: {runtime}")
    return runtime


def _remove_build_helper(package_root: Path) -> None:
    helper = package_root / "core" / "build_protected_runtime.py"
    if helper.exists():
        helper.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
