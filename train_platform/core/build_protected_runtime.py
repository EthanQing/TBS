from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PROTECTED_PATHS = ("train_platform/services", "train_platform/workers")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble protected Train Platform runtime sources.")
    parser.add_argument("--source", type=Path, default=Path("train_platform"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pyarmor", "pyc", "none"), default=os.getenv("SOURCE_PROTECTION_MODE", "pyarmor"))
    parser.add_argument("--protect", action="append", default=None)
    parser.add_argument("--pyarmor-platform", default=os.getenv("PYARMOR_PLATFORM", ""))
    parser.add_argument("--allow-fallback", default=os.getenv("ALLOW_SOURCE_PROTECTION_FALLBACK", "1"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"source package not found: {source}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    protected_paths = [_resolve_protected_path(source, item) for item in (args.protect or DEFAULT_PROTECTED_PATHS)]
    _copy_package(source, output / source.name)
    _strip_bom_in_tree(output / source.name)
    _remove_build_helper(output / source.name)

    if args.mode == "none":
        print("source protection disabled")
        return 0
    if args.mode == "pyc":
        _compile_all(output, source, protected_paths)
        return 0

    try:
        _run_pyarmor(source, output, protected_paths, platform=args.pyarmor_platform)
    except SystemExit:
        if str(args.allow_fallback).strip().lower() not in {"1", "true", "yes", "y", "on"}:
            raise
        print("WARNING: PyArmor protection failed; falling back to pyc-only protection.", flush=True)
        _compile_all(output, source, protected_paths)
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


def _compile_all(output: Path, source: Path, protected_paths: list[Path]) -> None:
    for path in protected_paths:
        rel = path.relative_to(source)
        _compile_pyc_tree(output / source.name / rel)


def _compile_pyc_tree(root: Path) -> None:
    for py_file in sorted(root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        pyc_file = py_file.with_suffix(".pyc")
        py_compile.compile(str(py_file), cfile=str(pyc_file), doraise=True, optimize=2)
        py_file.unlink()


def _run_pyarmor(source: Path, output: Path, protected_paths: list[Path], *, platform: str) -> None:
    pyarmor = shutil.which("pyarmor")
    if not pyarmor:
        raise SystemExit("pyarmor executable not found. Install pyarmor in the build stage.")

    with tempfile.TemporaryDirectory(prefix="train-platform-protect-") as temp_dir:
        temp_root = Path(temp_dir)
        temp_source = temp_root / source.name
        _copy_package(source, temp_source)
        _strip_bom_in_tree(temp_source)

        obf_dir = temp_root / "obfuscated"
        cmd = [pyarmor, "gen", "-r", "-O", str(obf_dir)]
        if platform:
            cmd.extend(["--platform", platform])
        cmd.extend(str(Path(source.name) / path.relative_to(source)) for path in protected_paths)

        print("+ " + " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True, cwd=temp_root)
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                "PyArmor protection failed. Register PyArmor during Docker build or allow fallback."
            ) from e

        for protected_path in protected_paths:
            rel = protected_path.relative_to(source)
            generated = obf_dir / rel.name
            nested_generated = obf_dir / source.name / rel
            if not generated.exists() and nested_generated.exists():
                generated = nested_generated
            target = output / source.name / rel
            if not generated.exists():
                raise SystemExit(f"PyArmor output not found for {protected_path}: {generated}")
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(generated, target)

        for runtime_dir in obf_dir.glob("pyarmor_runtime_*"):
            if runtime_dir.is_dir():
                shutil.copytree(runtime_dir, output / runtime_dir.name, dirs_exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
