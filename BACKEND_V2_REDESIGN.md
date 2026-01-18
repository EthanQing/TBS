# Backend v2 重构版（推倒重来）

本文件描述当前仓库的“重构版后端”最终形态：**新数据库结构 + 新 API（/api/v2）**。

> 说明：此版本不兼容旧库/旧接口，默认使用“删库重建 + Alembic upgrade head”的方式初始化。

## 1. 核心设计点

- **数据版本（dataset_versions）**：训练 run 绑定到具体数据版本，后期可做 diff/回滚/谱系。
- **训练运行（training_runs）**：把“队列字段/心跳/取消/删除”做成 run 自身字段，便于查询与多 worker 协作。
- **模型版本（model_versions）**：从训练 run 注册模型版本，支持阶段（development/testing/production/deprecated）。
- **部署（deployments）**：部署只引用 model_version，避免直接绑训练 run。
- **推理记录（inference_runs）**：记录推理输入/输出，后期可做回放与评估。

## 2. 数据库表（ERD 文字版）

### 2.1 datasets / dataset_versions
- `datasets`
  - `dataset_id` (PK)
  - `name` (unique)
  - `dataset_type`
  - `storage_path`（建议为 BASE_DATASETS_DIR 下的相对路径/token）
  - `active_version_id` → `dataset_versions.version_id`
- `dataset_versions`
  - `version_id` (PK)
  - `dataset_id` → `datasets.dataset_id`
  - `version`（单 dataset 递增）
  - `parent_version_id`（自引用，用于谱系）
  - `manifest_path` / `snapshot_path`

### 2.2 projects
- `projects`
  - `project_id` (PK)
  - `name` (unique)
  - `dataset_id` → `datasets.dataset_id`
  - `task_type`

### 2.3 model_architectures
- `model_architectures`
  - `architecture_id` (PK)
  - `family` / `variant` / `task_type`
  - `engine`（训练插件 key，如 `ultralytics-yolo`）

### 2.4 training_runs（包含队列/控制字段）
- `training_runs`
  - `run_id` (PK, uuid string)
  - `project_id` → `projects.project_id`
  - `dataset_version_id` → `dataset_versions.version_id`
  - `architecture_id` → `model_architectures.architecture_id`
  - `status`（created/queued/running/...）
  - `queued_at/claimed_at/worker_id/pid/heartbeat_at`
  - `cancel_requested_at/delete_requested_at/hidden`
- `training_run_parameters`（1:1）
- `training_run_results`（1:1）
- `training_run_epoch_metrics`（1:N）
- `training_run_events`（1:N）
- `training_run_artifacts`（1:N）
 - `training_run_meta`（1:1，可选：group/tags/notes/extra）

### 2.5 model_versions（模型注册表）
- `model_versions`
  - `model_version_id` (PK)
  - `project_id` → `projects.project_id`
  - `run_id` → `training_runs.run_id`
  - `version`（unique per project）
  - `stage`
  - `metrics` / `weights_path`

### 2.6 deployments / deployment_logs
- `deployments`
  - `deployment_id` (PK)
  - `model_version_id` → `model_versions.model_version_id`
  - `status` / `endpoint_url` / `config`
- `deployment_logs`（1:N）

### 2.7 inference_runs
- `inference_runs`
  - `inference_id` (PK)
  - `model_version_id` → `model_versions.model_version_id`
  - `deployment_id` → `deployments.deployment_id`（可选）
  - `input_path` / `output`

## 3. API v2 概览（核心资源）

Base: `/api/v2`

### datasets
- `GET /datasets`（分页）
- `POST /datasets`
- `POST /datasets/import`（上传压缩包并创建 dataset）
- `GET /datasets/{dataset_id}`
- `GET /datasets/{dataset_id}/detail`（聚合：dataset + active_version + statistics + 最近 events）
- `PATCH /datasets/{dataset_id}`
- `DELETE /datasets/{dataset_id}?delete_files=false`
- `POST /datasets/{dataset_id}/uploads/images`（向已有 dataset 追加上传图片，并记录上传历史）
- `GET /datasets/{dataset_id}/events`（数据集事件/上传历史，分页）
- `GET /datasets/{dataset_id}/statistics`（基于 active_version 或指定 version_id）
- `GET /datasets/{dataset_id}/files?version_id=...&kind=image&q=...`（基于 manifest 列文件/图片）
- `GET /datasets/{dataset_id}/versions`（分页）
- `POST /datasets/{dataset_id}/versions`
- `POST /datasets/{dataset_id}/versions/{version_id}/activate`
- `GET /datasets/{dataset_id}/versions/{version_id}/diff?base_version_id=...&limit=200`（manifest diff）

### projects
- `GET /projects`（分页）
- `POST /projects`
- `GET /projects/{project_id}`
- `PATCH /projects/{project_id}`
- `DELETE /projects/{project_id}`

### architectures
- `GET /architectures`
- `POST /architectures`

### training-runs
- `GET /training-runs`（分页 + 过滤）
- `POST /training-runs`
- `GET /training-runs/{run_id}`
- `PATCH /training-runs/{run_id}`
- `POST /training-runs/{run_id}/queue`
- `POST /training-runs/{run_id}/cancel`
- `DELETE /training-runs/{run_id}`（请求删除）
- `GET /training-runs/{run_id}/events`
- `GET /training-runs/{run_id}/metrics/epochs`
- `GET /training-runs/{run_id}/artifacts`
- `GET /training-runs/{run_id}/meta`
- `PATCH /training-runs/{run_id}/meta`
- `GET /training-runs/{run_id}/logs/tail?which=stdout&lines=200`
- `WS /training-runs/{run_id}/metrics/stream`（指标/事件流）
- `POST /training-runs/compare`

### model-versions
- `GET /model-versions`（分页）
- `POST /model-versions`
- `GET /model-versions/{model_version_id}`
- `PATCH /model-versions/{model_version_id}`

### deployments
- `GET /deployments`（分页）
- `POST /deployments`
- `GET /deployments/{deployment_id}`
- `PATCH /deployments/{deployment_id}`
- `DELETE /deployments/{deployment_id}`
- `GET /deployments/{deployment_id}/logs`
- `POST /deployments/{deployment_id}/logs`

### inference
- `POST /inference-runs`（执行一次推理并入库）
- `POST /inference-runs/upload`（上传图片到 BASE_TEMP_DIR，返回 token/path）
