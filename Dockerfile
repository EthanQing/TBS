# Worker image (GPU training/inference)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/worker.txt ./requirements/worker.txt
RUN pip install --no-cache-dir -r requirements/worker.txt

# Ensure torch is available from the base image.
RUN python - <<'PY'
import torch
print('torch:', torch.__version__)
PY

COPY train_platform/ ./train_platform/
COPY alembic.ini .

RUN mkdir -p /app/datasets /app/training_runs /app/temp

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "train_platform.workers.worker"]
