from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


DEFAULT_PROTECTED_PATHS = ("train_platform/services",)
DEFAULT_PROTECTED_FILES = (
    "train_platform/workers/worker_impl.py",
    "train_platform/workers/yolo_worker_impl.py",
    "train_platform/workers/paddle_worker_impl.py",
    "train_platform/workers/training/train_entry_impl.py",
    "train_platform/workers/training/vdl_bridge.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble pyc-protected Train Platform runtime sources.")
    parser.add_argument("--source", type=Path, default=Path("train_platform"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect", action="append", default=None)
    parser.add_argument("--protect-file", action="append", default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"source package not found: {source}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    protected_paths = [_resolve_protected_path(source, item) for item in (args.protect or DEFAULT_PROTECTED_PATHS)]
    protected_files = [_resolve_protected_file(source, item) for item in (args.protect_file or DEFAULT_PROTECTED_FILES)]

    package_root = output / source.name
    _copy_package(source, package_root)
    _strip_bom_in_tree(package_root)
    _remove_build_helper(package_root)
    _compile_protected_sources(output, source, protected_paths, protected_files)
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


def _resolve_protected_file(source: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        candidate = (source.parent / raw).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise SystemExit(f"protected file not found: {raw}")

    try:
        candidate.relative_to(source)
    except ValueError as e:
        raise SystemExit(f"protected file must be inside {source}: {candidate}") from e
    return candidate


def _copy_package(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )


def _strip_bom_in_tree(root: Path) -> None:
    for py_file in root.rglob("*.py"):
        data = py_file.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            py_file.write_bytes(data[3:])


def _remove_build_helper(package_root: Path) -> None:
    helper = package_root / "core" / "build_protected_runtime.py"
    if helper.exists():
        helper.unlink()


def _compile_protected_sources(
    output: Path,
    source: Path,
    protected_paths: list[Path],
    protected_files: list[Path],
) -> None:
    package_root = output / source.name
    seen: set[Path] = set()
    for path in protected_paths:
        rel = path.relative_to(source)
        for py_file in sorted((package_root / rel).rglob("*.py")):
            _compile_pyc_file(py_file, seen=seen)
    for path in protected_files:
        rel = path.relative_to(source)
        _compile_pyc_file(package_root / rel, seen=seen)


def _compile_pyc_file(py_file: Path, *, seen: set[Path]) -> None:
    py_file = py_file.resolve()
    if py_file in seen or py_file.name == "__init__.py":
        return
    if not py_file.exists():
        raise SystemExit(f"protected source missing in output: {py_file}")
    pyc_file = py_file.with_suffix(".pyc")
    py_compile.compile(str(py_file), cfile=str(pyc_file), doraise=True, optimize=2)
    py_file.unlink()
    seen.add(py_file)


if __name__ == "__main__":
    raise SystemExit(main())
