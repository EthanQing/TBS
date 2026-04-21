from __future__ import annotations

import fnmatch
import os
import re
import shutil
import sys
from pathlib import Path

try:
    from Cython.Build import cythonize
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("Cython is required. Install it with: pip install cython") from e

try:
    from setuptools import Extension, find_packages, setup
    from setuptools.command.build_ext import build_ext
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("setuptools is required. Install it with: pip install setuptools wheel") from e


ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "train_platform"
BUILD_DIR = ROOT / "build"
CYTHON_BUILD_DIR = BUILD_DIR / "cython"
EXT_BUILD_LIB_DIR = BUILD_DIR / "lib"
EXT_BUILD_TEMP_DIR = BUILD_DIR / "temp"
RUNTIME_DIR = BUILD_DIR / "runtime"
RUNTIME_ROOT_PATHS = [Path("alembic.ini")]


def _read_version() -> str:
    """
    Keep version in-sync with `train_platform/__init__.py` without importing the package.
    """
    init_py = PKG_DIR / "__init__.py"
    try:
        text = init_py.read_text(encoding="utf-8")
    except Exception:
        return "0.0.0"

    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return m.group(1) if m else "0.0.0"


def _env_csv(name: str) -> list[str]:
    v = os.environ.get(name, "")
    return [x.strip() for x in v.split(",") if x.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _module_name_from_path(py_file: Path) -> str:
    # e.g. train_platform/workers/worker.py -> train_platform.workers.worker
    return ".".join(py_file.relative_to(ROOT).with_suffix("").parts)


def _iter_train_platform_py_files() -> list[Path]:
    if not PKG_DIR.exists():
        raise RuntimeError(f"Package dir not found: {PKG_DIR}")

    return sorted(p for p in PKG_DIR.rglob("*.py") if p.is_file())


def _build_extensions() -> list[Extension]:
    # Notes:
    # - We always keep __init__.py as Python to avoid package import edge cases.
    # - If you still start modules via `python -m some.module`, DO NOT compile that module,
    #   because `-m` relies on source code objects. Use CYTHON_EXCLUDE_MODULES to skip them.
    #
    # Selection:
    # - If CYTHON_INCLUDE_GLOBS is set (comma-separated), only matching files are compiled.
    # - Excludes are applied after includes.
    include_globs = _env_csv("CYTHON_INCLUDE_GLOBS")
    exclude_modules = set(_env_csv("CYTHON_EXCLUDE_MODULES"))

    # Safe defaults: Alembic loads migration scripts by reading .py files from disk.
    # If you compile/remove these, `alembic upgrade head` can break.
    #
    # FastAPI also inspects Python call signatures for route handlers/dependencies.
    # Leaving the HTTP entry layer as `.py` avoids runtime issues like:
    #   ValueError: no signature found for builtin <built-in function ...>
    default_exclude_globs = [
        "train_platform/app.py",
        "train_platform/api/*.py",
        "train_platform/api/**/*.py",
        "train_platform/db/migrations/*.py",
        "train_platform/db/migrations/versions/*.py",
    ]
    exclude_globs = default_exclude_globs + _env_csv("CYTHON_EXCLUDE_GLOBS")

    extensions: list[Extension] = []
    for py in _iter_train_platform_py_files():
        if py.name == "__init__.py":
            continue

        # Glob patterns are matched against POSIX-style relative paths, e.g.:
        #   CYTHON_INCLUDE_GLOBS=train_platform/services/*.py,train_platform/services/**/*.py
        rel_posix = py.relative_to(ROOT).as_posix()
        if include_globs and not any(Path(rel_posix).match(pat) for pat in include_globs):
            continue

        mod = _module_name_from_path(py)
        if mod in exclude_modules:
            continue

        # Exclude patterns are matched against POSIX-style relative paths, e.g.:
        #   CYTHON_EXCLUDE_GLOBS=train_platform/workers/*.py
        if any(Path(rel_posix).match(pat) for pat in exclude_globs):
            continue

        extensions.append(Extension(mod, [os.fspath(py.relative_to(ROOT))]))

    if not extensions:
        raise RuntimeError("No extensions selected for cythonize (check exclude settings).")

    return extensions


def _source_copy_ignore(_src: str, names: list[str]) -> set[str]:
    patterns = {"__pycache__", "*.pyc", "*.pyo", "*.c", "*.so", "*.pyd", "*.dll"}
    ignored: set[str] = set()
    for name in names:
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            ignored.add(name)
    return ignored


def _runtime_overlay_ignore(_src: str, names: list[str]) -> set[str]:
    patterns = {"__pycache__", "*.pyc", "*.pyo"}
    ignored: set[str] = set()
    for name in names:
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            ignored.add(name)
    return ignored


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_source_copy_ignore)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _assemble_runtime_tree(compiled_module_names: set[str]) -> None:
    if not EXT_BUILD_LIB_DIR.exists():
        raise RuntimeError(f"Compiled extension output not found: {EXT_BUILD_LIB_DIR}")

    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)

    # Start from source tree so we keep pure-Python files/resources that must remain visible.
    shutil.copytree(PKG_DIR, RUNTIME_DIR / PKG_DIR.name, ignore=_source_copy_ignore)

    # Copy top-level runtime support files (for example Alembic config) if they exist.
    for rel_path in RUNTIME_ROOT_PATHS:
        src = ROOT / rel_path
        if src.exists():
            _copy_path(src, RUNTIME_DIR / rel_path)

    # Overlay compiled extension modules.
    shutil.copytree(EXT_BUILD_LIB_DIR, RUNTIME_DIR, dirs_exist_ok=True, ignore=_runtime_overlay_ignore)

    # Remove plaintext .py files for modules that were compiled into extension modules.
    for mod_name in compiled_module_names:
        rel_py = Path(*mod_name.split(".")).with_suffix(".py")
        runtime_py = RUNTIME_DIR / rel_py
        if runtime_py.exists():
            runtime_py.unlink()


def _remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


class BuildExtToBuildDir(build_ext):
    """
    Always place compiled extensions and temporary files under `build/`,
    then assemble a runnable deployment tree under `build/runtime/`.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.inplace:
            print(
                "warning: --inplace is ignored for this project; "
                "compiled outputs will be written under build/",
                file=sys.stderr,
            )
        self.inplace = False
        self.build_lib = str(EXT_BUILD_LIB_DIR)
        self.build_temp = str(EXT_BUILD_TEMP_DIR)

    def run(self) -> None:
        _remove_tree_if_exists(EXT_BUILD_LIB_DIR)
        _remove_tree_if_exists(EXT_BUILD_TEMP_DIR)
        _remove_tree_if_exists(RUNTIME_DIR)
        super().run()
        _assemble_runtime_tree(COMPILED_MODULE_NAMES)
        self.announce(f"runtime package assembled under: {RUNTIME_DIR}", level=2)


extensions = _build_extensions()
COMPILED_MODULE_NAMES = {ext.name for ext in extensions}

ext_modules = cythonize(
    extensions,
    build_dir=str(CYTHON_BUILD_DIR),
    compiler_directives={
        # Keep semantics close to CPython, but ensure Python-3 behavior.
        "language_level": "3",
        # We cythonize for obfuscation/packaging, not for static typing/perf.
        # Treat Python type annotations as *annotations* (not as Cython type declarations),
        # otherwise FastAPI-style defaults like `x: str = Query(...)` can break at import time.
        "annotation_typing": False,
        # FastAPI (and other libs) can rely on runtime signatures/annotations via inspect.signature().
        # When cythonizing such callables, you may need CYTHON_BINDING=1 / CYTHON_EMBED_SIGNATURE=1.
        # Defaults are conservative to reduce introspection/signature leakage.
        "binding": _env_bool("CYTHON_BINDING", default=False),
        "embedsignature": _env_bool("CYTHON_EMBED_SIGNATURE", default=False),
    },
    annotate=bool(int(os.environ.get("CYTHON_ANNOTATE", "0"))),
    nthreads=int(os.environ.get("CYTHON_NTHREADS", "0") or "0"),
)

setup(
    name="train-platform-backend",
    version=_read_version(),
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtToBuildDir},
    zip_safe=False,
)
