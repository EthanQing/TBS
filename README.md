# TBS Backend

TBS Backend is a training platform backend built with **FastAPI**, **SQLAlchemy**, and **Alembic**. It provides APIs and worker components for dataset management, training orchestration, deployment, inference, alerting, and system resource monitoring.

> Current stable API namespace: `/api/v3`

## Features

- Dataset management, mounted imports, publishing, and augmentation
- Cancellable illegal dataset publish jobs for long-running LabelMe/JSON conversion
- Project and model version management
- Training run orchestration
- Project card training alerts for running and completed-unreviewed runs
- Pluggable training framework architecture
- Deployment and inference workflows
- Post-training model evaluation on labeled YOLO detection datasets
- Alert rules and alert event management
- System resource monitoring for CPU, memory, and GPU
- Static asset serving for datasets, thumbnails, training artifacts, and pre-trained models

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- SQLAlchemy 2.x
- Alembic
- MySQL / SQLite
- Uvicorn
- MLflow
- Ultralytics YOLO
- PaddleDetection
- ONNX Runtime

## Project Structure

```text
train_platform/
  api/v3/          API routers
  core/            configuration and shared infrastructure
  db/              database initialization and migrations
  schemas/v3/      Pydantic schemas
  services/v3/     application services
  training/        training plugins and runtime logic
  workers/         worker entrypoints

docs/              supplementary documentation
requirements/      dependency files
datasets/          dataset storage
training_runs/     training artifacts
temp/              temporary files
pretrain_models/   pre-trained model storage
PaddleDetection/   local PaddleDetection release/2.6 checkout for Paddle workers
```

## Requirements

- Python 3.10+
- MySQL 8+ (default setup)
- Windows or Linux
- CUDA / NVIDIA driver environment if GPU training or GPU monitoring is required

## Getting Started

### 1. Configure environment variables

Windows:

```bash
copy .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

### 2. Install dependencies

Using `pip` with the backend requirements file:

```bash
pip install -r requirements/backend.txt
```

Or install from the project metadata:

```bash
pip install -e .
```

For PaddleDetection training or inference, clone the official source checkout.
The `paddledet` pip package is not used because it does not bundle the official
`configs/` tree required by the platform:

```bash
git clone --branch release/2.6 --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git PaddleDetection
```

Set `PADDLE_DET_DIR` if the checkout is not under this backend directory:

```bash
set PADDLE_DET_DIR=C:\path\to\PaddleDetection
```

### 3. Initialize the database

Create the target database first.

Default database name:

- `train_backend_v2`

Then run migrations:

```bash
alembic -c alembic.ini upgrade head
```

### 4. Start the API server

```bash
uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload
```

Available endpoints after startup:

- Health check: `http://127.0.0.1:18000/health`
- Swagger UI: `http://127.0.0.1:18000/docs`
- OpenAPI schema: `http://127.0.0.1:18000/openapi.json`

## Windows Portable Launcher

The outer `train` workspace includes a Windows-only portable launcher for
customer sites that cannot install Docker or WSL. The launcher is built from
`launcher/windows/TrainPlatformLauncher.csproj` and distributed as
`TrainPlatformLauncher.exe` in a self-contained package.

Portable layout:

- `runtime/python/`: Python 3.10 x64 runtime with backend and worker
  dependencies installed.
- `runtime/mariadb/`: MariaDB/MySQL Windows ZIP runtime.
- `app/TBS/`: this backend runtime, `alembic.ini`, requirements, and optional
  PaddleDetection checkout. Customer packages use the pyc-protected runtime
  assembled by `train_platform.core.build_protected_runtime`; service modules
  and selected training worker core files are shipped as bytecode instead of raw
  sources.
- `app/TFS/dist/`: built frontend static files.
- `data/`: MySQL data, datasets, imports, training runs, temp files, and
  pre-trained models.
- `logs/`: launcher and service logs.

The launcher writes non-secret runtime settings to `app/TBS/.env`, initializes
the bundled database on first run, applies Alembic migrations, starts FastAPI on
`127.0.0.1:18001`, starts the YOLO worker, optionally starts the Paddle worker,
and serves the frontend on `127.0.0.1:18581` with `/api`, `/static`, and
WebSocket proxying to the backend. Customer packages embed `license.dat` inside
`TrainPlatformLauncher.exe`; the launcher injects it through process
environment variables and does not expose the license in the UI, `launcher.json`,
or `.env`. All child processes are attached to a Windows Job Object so stopping
the launcher stops the local runtime stack.

Build the portable package from the outer workspace:

```nu
let build_args = [
  '-NoProfile'
  '-ExecutionPolicy'
  'Bypass'
  '-File'
  'tools\windows-portable\build-windows-portable.ps1'
  '-OutputDir'
  'outputs\train-platform-windows-portable'
  '-LicenseFile'
  'C:\secrets\license.dat'
  '-BuildPythonExe'
  'C:\Python310\python.exe'
  '-PythonRuntimeDir'
  'C:\runtimes\python'
  '-MariaDbRuntimeDir'
  'C:\runtimes\mariadb'
]
^powershell.exe ...$build_args
```

## Worker Processes

Generic training worker:

```bash
python -m train_platform.workers.worker
```

YOLO worker:

```bash
python -m train_platform.workers.yolo_worker
```

Set any stable, unique `WORKER_ID` when running multiple YOLO worker instances
(for example, `yolo-worker-a` or `node42-yolo-1`). The dedicated YOLO entrypoint
falls back to `worker-yolo` when the variable is not set, and uses the provided
value as-is in queue claims, events, and logs. For multi-GPU Docker deployments,
also verify the GPU visibility inside each container because `WORKER_ID` only
identifies the queue worker; it does not bind the process to a GPU by itself.
Explicit training devices such as `device=0` and `device=1` use host GPU ids;
Docker workers restricted by numeric `NVIDIA_VISIBLE_DEVICES` remap those ids
to container-local CUDA ids before launching the training process.

The YOLO worker also polls queued model-conversion jobs from
`temp/model_conversions` and runs PT/PTH -> ONNX conversion locally in the
worker process.

Paddle training worker:

```bash
python -m train_platform.workers.paddle_worker
```

Paddle workers require a complete PaddleDetection `release/2.6` source checkout
containing both `ppdet/` and `configs/`. By default the backend resolves
`PaddleDetection/` under this directory; Docker paddle worker images clone the
same checkout into `/app/PaddleDetection`.

Generic inference worker:

```bash
python -m train_platform.workers.inference_worker
```

Paddle inference worker:

```bash
python -m train_platform.workers.paddle_inference_worker
```

Model evaluation jobs under `/api/v3/model-evaluations` reuse the same inference
workers. Keep the appropriate YOLO or Paddle inference worker running before
starting an evaluation from the deployment center.

## Configuration

### Database

- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_DATABASE`
- `DB_POOL_SIZE`
- `DB_MAX_OVERFLOW`
- `DB_POOL_TIMEOUT`
- `DB_POOL_RECYCLE`

Optional overrides:

- `DATABASE_URL`
- `ALEMBIC_DATABASE_URL`

`DB_POOL_SIZE=20`, `DB_MAX_OVERFLOW=30`, `DB_POOL_TIMEOUT=60`, and
`DB_POOL_RECYCLE=300` are the default backend pool settings for a single
backend container. Tune them together with MySQL `max_connections` when running
multiple backend replicas.

### Runtime and workers

- `BACKEND_PORT`
- `WORKER_POLL_INTERVAL`
- `WORKER_HEARTBEAT_INTERVAL`
- `WORKER_STALE_AFTER_SECONDS`
- `WORKER_BIND_HOST`

### License

- `TRAIN_PLATFORM_LICENSE_REQUIRED`
- `TRAIN_PLATFORM_LICENSE_PATH`
- `TRAIN_PLATFORM_LICENSE_DATA_B64`
- `TRAIN_PLATFORM_LICENSE_DATA`

`TRAIN_PLATFORM_LICENSE_PATH` keeps the Docker/file-based deployment flow.
Windows portable customer packages prefer `TRAIN_PLATFORM_LICENSE_DATA_B64`,
which is supplied by the launcher from its embedded license resource.

### Storage

- `TRAIN_PLATFORM_HOME`
- `BASE_DATASETS_DIR`
- `BASE_TRAINING_DIR`
- `BASE_TEMP_DIR`
- `BASE_UPLOAD_SESSIONS_DIR` (default: `BASE_DATASETS_DIR/.uploads`)
- `BASE_DATASET_STAGING_DIR` (default: `BASE_DATASETS_DIR/.staging`)
- `BASE_IMPORTS_DIR` (default: `TRAIN_PLATFORM_HOME/imports`)
- `BASE_PRETRAIN_MODELS_DIR`
- `PADDLE_DET_DIR`

### Large Dataset Uploads

Dataset ZIP uploads can use resumable chunk sessions under `/api/v3/standard-datasets/{id}/upload-sessions` and `/api/v3/illegal-datasets/{id}/upload-sessions`. `complete` merges uploaded chunks and returns a `task_id`; extraction, validation, indexing, and version updates continue in the background and can be queried with `GET /api/v3/dataset-upload-tasks/{task_id}`.

For offline deployments, place large ZIP files or extracted dataset directories under `BASE_IMPORTS_DIR` and start the same background flow with `/api/v3/standard-datasets/{id}/import-from-path` or `/api/v3/illegal-datasets/{id}/import-from-path`. Recommended deployment knobs:

- `UPLOAD_CHUNK_SIZE_MB` defaults to `64`.
- `UPLOAD_SESSION_TTL_HOURS` defaults to `24`.
- `DATASET_IMPORT_MAX_WORKERS` controls parallel JSON/stat parsing for illegal mounted LabelMe/JSON imports and defaults to `min(8, cpu_count)`.
- Mount `BASE_DATASETS_DIR`, `BASE_UPLOAD_SESSIONS_DIR`, `BASE_DATASET_STAGING_DIR`, and `BASE_IMPORTS_DIR` on large-capacity storage.
- Directory import and mounted-link import only see paths visible to the backend process or container. The outer Docker Compose files mount host `./TBS/imports` to container `/app/imports`, the default `BASE_IMPORTS_DIR`. Mount network shares into the host/container first, then expose them through `BASE_IMPORTS_DIR` or `DATASET_IMPORT_ROOTS`.
- Configure Nginx upload routes with enough `proxy_read_timeout` / `proxy_send_timeout`, and keep request-body temp storage off the container overlay filesystem.

### Inference restrictions

- `INTERNAL_API_TOKEN`
- `INFERENCE_MAX_DOWNLOAD_BYTES`
- `INFERENCE_DOWNLOAD_TIMEOUT_SEC`
- `INFERENCE_ALLOWED_SCHEMES`
- `INFERENCE_ALLOWED_HOSTS`

### System metrics

- `SYSTEM_METRICS_RETENTION_SECONDS`
- `SYSTEM_METRICS_MAX_POINTS`
- `SYSTEM_METRICS_STEP_SECONDS`

## System Metrics API

The backend exposes resource monitoring endpoints under `/api/v3/system-metrics`:

- `GET /api/v3/system-metrics/summary`
- `GET /api/v3/system-metrics/history`
- `GET /api/v3/system-metrics/nodes`

Collected metrics include:

- CPU utilization
- Memory utilization / used / total
- GPU utilization
- GPU memory usage
- Node-level historical samples

Required dependencies:

- `psutil`
- `pynvml`

## Framework Plugin System

The project includes a pluggable training framework system for:

- discovering available training plugins
- retrieving plugin configuration schemas
- validating and normalizing framework-specific training configuration

See `docs/11_辅助接口.md`.

## Documentation

- `docs/00_概述.md`
- `docs/01_项目管理.md`
- `docs/02_数据集管理.md`
- `docs/03_训练任务.md`
- `docs/04_模型版本.md`
- `docs/05_模型部署.md`
- `docs/06_推理服务.md`
- `docs/07_在线服务.md`
- `docs/08_数据集转换与增强.md`
- `docs/09_模型转换.md`
- `docs/10_系统监控与告警.md`
- `docs/11_辅助接口.md`
- `docs/12_合格模型.md`

## Development Notes

- Mainline development should be merged into `main`
- Experimental or large-scale work is better isolated in feature branches
- Before submitting changes, verify:
  - database migrations are complete
  - API routers are registered
  - schema, service, and route definitions are aligned
  - worker entrypoints start successfully

## License

This repository currently does not declare an open-source license.

If you plan to publish it publicly, add a `LICENSE` file and update this section accordingly.
