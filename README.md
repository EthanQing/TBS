# TBS Backend

TBS Backend is a training platform backend built with **FastAPI**, **SQLAlchemy**, and **Alembic**. It provides APIs and worker components for dataset management, training orchestration, deployment, inference, alerting, and system resource monitoring.

> Current stable API namespace: ` /api/v2 `

## Features

- Dataset management, conversion, and augmentation
- Project and model version management
- Training run orchestration
- Pluggable training framework architecture
- Deployment and inference workflows
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
  api/v2/          API routers
  core/            configuration and shared infrastructure
  db/              database initialization and migrations
  schemas/         Pydantic schemas
  services/        application services
  training/        training plugins and runtime logic
  workers/         worker entrypoints

docs/              supplementary documentation
requirements/      dependency files
datasets/          dataset storage
training_runs/     training artifacts
temp/              temporary files
pretrain_models/   pre-trained model storage
PaddleDetection/   local PaddleDetection checkout (optional)
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

## Worker Processes

Generic training worker:

```bash
python -m train_platform.workers.worker
```

YOLO worker:

```bash
python -m train_platform.workers.yolo_worker
```

Paddle training worker:

```bash
python -m train_platform.workers.paddle_worker
```

Generic inference worker:

```bash
python -m train_platform.workers.inference_worker
```

Paddle inference worker:

```bash
python -m train_platform.workers.paddle_inference_worker
```

## Configuration

### Database

- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_DATABASE`

Optional overrides:

- `DATABASE_URL`
- `ALEMBIC_DATABASE_URL`

### Runtime and workers

- `BACKEND_PORT`
- `WORKER_POLL_INTERVAL`
- `WORKER_HEARTBEAT_INTERVAL`
- `WORKER_STALE_AFTER_SECONDS`
- `WORKER_BIND_HOST`

### Storage

- `TRAIN_PLATFORM_HOME`
- `BASE_DATASETS_DIR`
- `BASE_TRAINING_DIR`
- `BASE_TEMP_DIR`
- `BASE_PRETRAIN_MODELS_DIR`
- `PADDLE_DET_DIR`

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

The backend exposes resource monitoring endpoints under ` /api/v2/system-metrics `:

- `GET /api/v2/system-metrics/summary`
- `GET /api/v2/system-metrics/history`
- `GET /api/v2/system-metrics/nodes`

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

See:

- `docs/framework_plugins_api.md`

## Documentation

- `docs/datasets_api.md`
- `docs/alarms_api.md`
- `docs/framework_plugins_api.md`
- `docs/paddle_local_dev.md`
- `docs/custom_script_integration/`

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
