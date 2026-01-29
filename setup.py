from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from Cython.Build import cythonize
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("Cython is required. Install it with: pip install cython") from e

try:
    from setuptools import Extension, find_packages, setup
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit("setuptools is required. Install it with: pip install setuptools wheel") from e


ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "train_platform"


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
    default_exclude_globs = [
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

        extensions.append(Extension(mod, [str(py)]))

    if not extensions:
        raise RuntimeError("No extensions selected for cythonize (check exclude settings).")

    return extensions


extensions = _build_extensions()

ext_modules = cythonize(
    extensions,
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
    zip_safe=False,
)
