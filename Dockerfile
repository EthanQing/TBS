# Worker image (GPU training/inference)

# ---------- build stage: cythonize train_platform ----------
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Toolchain for building Cython extension modules.
RUN set -eux; \
    apt-get update; \
    for i in 1 2 3 4 5; do \
        if apt-get install -y --no-install-recommends build-essential; then \
            ok=1; \
            break; \
        fi; \
        echo "apt-get install failed, retrying (${i}/5)..." >&2; \
        sleep 5; \
        apt-get update; \
    done; \
    test "${ok:-}" = "1"; \
    rm -rf /var/lib/apt/lists/*

COPY requirements/build.txt ./requirements/build.txt
RUN pip install --no-cache-dir -r requirements/build.txt

COPY setup.py ./setup.py
COPY train_platform/ ./train_platform/
COPY alembic.ini ./alembic.ini

# Compile only "core" modules. FastAPI relies heavily on inspect.signature() / annotations
# for dependency injection, so we enable signature metadata for compiled callables.
#
# Keep worker entry modules as Python (`python -m ...` requires a code object; extension modules don't work with -m).
#
# NOTE: `Path.match("dir/**/*.py")` does NOT match files directly under `dir/`, so we include both patterns.
ENV CYTHON_BINDING=1
ENV CYTHON_EMBED_SIGNATURE=1
ENV CYTHON_INCLUDE_GLOBS="train_platform/*.py,train_platform/services/*.py,train_platform/services/**/*.py,train_platform/repositories/*.py,train_platform/repositories/**/*.py,train_platform/schemas/*.py,train_platform/schemas/**/*.py,train_platform/models/*.py,train_platform/models/**/*.py,train_platform/utils/*.py,train_platform/utils/**/*.py,train_platform/training/*.py,train_platform/training/**/*.py,train_platform/core/*.py,train_platform/core/**/*.py,train_platform/db/*.py,train_platform/api/*.py,train_platform/api/**/*.py"
RUN python setup.py build_ext --inplace

# Remove only Python sources that were successfully compiled to an extension module.
RUN python - <<'PY'
from __future__ import annotations

from pathlib import Path

pkg = Path("train_platform")
for py in pkg.rglob("*.py"):
    if py.name == "__init__.py":
        continue
    if any(py.parent.glob(py.stem + ".*.so")):
        py.unlink()

for pat in ("*.c", "*.cpp"):
    for f in pkg.rglob(pat):
        f.unlink()
PY
RUN rm -rf build


# ---------- runtime stage ----------
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN set -eux; \
    apt-get update; \
    for i in 1 2 3 4 5; do \
        if apt-get install -y --no-install-recommends libgl1 libglib2.0-0 default-mysql-client; then \
            ok=1; \
            break; \
        fi; \
        echo "apt-get install failed, retrying (${i}/5)..." >&2; \
        sleep 5; \
        apt-get update; \
    done; \
    test "${ok:-}" = "1"; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/worker.txt ./requirements/worker.txt
RUN pip install --no-cache-dir -r requirements/worker.txt

# Ensure torch is available from the base image.
RUN python - <<'PY'
import torch
print('torch:', torch.__version__)
PY

COPY --from=builder /app/train_platform/ ./train_platform/
COPY alembic.ini .

RUN mkdir -p /app/datasets /app/training_runs /app/temp

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "train_platform.workers.worker"]
