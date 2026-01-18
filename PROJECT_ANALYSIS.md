# 训练平台后端 (Backend V2) - 完整分析文档

**项目名称**: Train Platform Backend (v2 rewrite)  
**技术栈**: FastAPI + SQLAlchemy + MySQL + PyTorch/YOLOv8  
**Python版本**: >= 3.10  
**API版本**: v1

---

## 一、项目概述

### 1.1 项目目标
- 完全放弃旧的 `backend/` 目录，进行从零开始的后端重写
- 将训练能力做成可插拔的 Engine 架构（支持 YOLO/MMDet/DETR 等）
- 提供现代化的 RESTful API 和 WebSocket 实时监控
- 支持分布式训练任务队列管理

### 1.2 核心功能模块
1. **数据集管理** - 支持检测、分割、分类三种类型
2. **项目管理** - 创建和组织训练项目
3. **模型架构管理** - 管理不同的预训练模型
4. **训练任务管理** - 创建、启动、监控、取消训练任务
5. **模型部署** - 支持多种部署平台（Local/Docker/K8s/AWS/Azure/GCP）
6. **推理服务** - 在线推理端点
7. **统计分析** - 项目、部署、模型性能统计

---

## 二、数据库设计架构

### 2.1 数据库连接信息
```
默认配置:
- Host: localhost
- Port: 3306
- User: root
- Password: password
- Database: train_backend_v2
- Driver: MySQL + PyMySQL
```

### 2.2 核心数据表结构

#### 2.2.1 数据集表 (datasets)
```sql
CREATE TABLE datasets (
  dataset_id          INT PRIMARY KEY AUTO_INCREMENT,
  dataset_name        VARCHAR(255) UNIQUE NOT NULL,
  dataset_path        VARCHAR(500) NOT NULL,    -- 相对路径或移植令牌
  dataset_type        ENUM('detection', 'segmentation', 'classification') NOT NULL,
  description         TEXT,
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```
**关键字段说明**:
- `dataset_type`: 支持三种数据集类型（检测/分割/分类）
- `dataset_path`: 存储相对于 BASE_DATASETS_DIR 的路径

---

#### 2.2.2 项目表 (projects)
```sql
CREATE TABLE projects (
  project_id          INT PRIMARY KEY AUTO_INCREMENT,
  project_name        VARCHAR(255) UNIQUE NOT NULL,
  description         TEXT,
  dataset_id          INT NOT NULL FOREIGN KEY,
  created_by          VARCHAR(128) DEFAULT 'system',
  is_active           BOOLEAN DEFAULT TRUE,
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP,
  updated_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```
**关键关系**:
- 与 datasets 表一对多（通过 dataset_id）
- 与 training_jobs 表一对多

---

#### 2.2.3 模型架构表 (model_architectures)
```sql
CREATE TABLE model_architectures (
  arch_id             INT PRIMARY KEY AUTO_INCREMENT,
  model_family        VARCHAR(50) NOT NULL,        -- 例: YOLOv8, YOLOv11, MMDet
  model_variant       VARCHAR(100) NOT NULL,       -- 例: yolov8n, yolov8m, rtmdet_tiny
  task_type           ENUM('detection', 'segmentation', 'classification') NOT NULL,
  pretrained_path     VARCHAR(500),                -- 预训练模型路径
  description         TEXT
);
```
**设计意义**:
- 支持多种模型框架和变体
- 方便扩展新的模型架构

---

#### 2.2.4 训练任务表 (training_jobs)
```sql
CREATE TABLE training_jobs (
  job_id              VARCHAR(36) PRIMARY KEY,     -- UUID格式
  project_id          INT NOT NULL FOREIGN KEY,
  arch_id             INT NOT NULL FOREIGN KEY,
  job_name            VARCHAR(255) NOT NULL,
  status              ENUM('pending', 'running', 'completed', 'failed', 'cancelled') NOT NULL,
  progress            INT DEFAULT 0,               -- 进度百分比 0-100
  current_epoch       INT DEFAULT 0,
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP,
  started_at          DATETIME(6),
  completed_at        DATETIME(6),
  error_message       TEXT
);
```
**状态流转**:
```
PENDING -> RUNNING -> COMPLETED (或 FAILED / CANCELLED)
```

---

#### 2.2.5 训练参数表 (training_parameters)
```sql
CREATE TABLE training_parameters (
  param_id            INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) UNIQUE NOT NULL FOREIGN KEY,
  epochs              INT DEFAULT 100,
  batch_size          INT DEFAULT 16,
  image_size          INT DEFAULT 640,
  learning_rate       DECIMAL(10, 8) DEFAULT 0.01,
  patience            INT DEFAULT 50,              -- Early stopping patience
  device              VARCHAR(32) DEFAULT 'auto',  -- cuda/cpu/auto
  workers             INT DEFAULT 8,               -- 数据加载工作进程数
  use_pretrained      BOOLEAN DEFAULT TRUE,
  optimizer           VARCHAR(64) DEFAULT 'AdamW',
  augmentation        JSON,                        -- 数据增强配置
  additional_params   JSON                         -- 其他参数
);
```

---

#### 2.2.6 训练结果表 (training_results)
```sql
CREATE TABLE training_results (
  result_id           INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) UNIQUE NOT NULL FOREIGN KEY,
  best_weights_path   VARCHAR(500),
  last_weights_path   VARCHAR(500),
  results_dir         VARCHAR(500),
  final_metrics       JSON,                        -- 最终指标
  best_metrics        JSON,                        -- 最佳指标
  training_logs       JSON,
  model_size_mb       DECIMAL(10, 2),
  inference_time_ms   DECIMAL(10, 4),
  flops               BIGINT
);
```

---

#### 2.2.7 训练任务控制表 (training_job_controls)
```sql
CREATE TABLE training_job_controls (
  job_id              VARCHAR(36) PRIMARY KEY FOREIGN KEY,
  queued_at           DATETIME(6),                 -- 入队时间
  claimed_at          DATETIME(6),                 -- Worker 认领时间
  worker_id           VARCHAR(128),                -- 执行的 Worker ID
  pid                 INT,                         -- 进程ID
  heartbeat_at        DATETIME(6),                 -- 最后心跳时间
  cancel_requested_at DATETIME(6),                 -- 取消请求时间
  cancel_reason       TEXT,
  delete_requested_at DATETIME(6),                 -- 删除请求时间
  hidden              BOOLEAN DEFAULT FALSE,       -- 隐藏状态
  stdout_log_path     VARCHAR(500),
  stderr_log_path     VARCHAR(500),
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```
**用途**: 用于跟踪训练任务的执行状态和日志位置

---

#### 2.2.8 训练任务元数据表 (training_job_meta)
```sql
CREATE TABLE training_job_meta (
  job_id              VARCHAR(36) PRIMARY KEY FOREIGN KEY,
  creator             VARCHAR(128),
  group               VARCHAR(128),                -- 任务分组标签
  tags                JSON,                        -- 自定义标签
  notes               TEXT,
  extra               JSON,                        -- 扩展字段
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```

---

#### 2.2.9 训练任务事件表 (training_job_events)
```sql
CREATE TABLE training_job_events (
  event_id            INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) NOT NULL FOREIGN KEY,
  level               VARCHAR(16) DEFAULT 'INFO',  -- DEBUG/INFO/WARNING/ERROR
  event_type          VARCHAR(64) DEFAULT 'event', -- 事件类型分类
  message             TEXT,
  data                JSON,                        -- 事件相关数据
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```

---

#### 2.2.10 训练周期指标表 (training_job_epoch_metrics)
```sql
CREATE TABLE training_job_epoch_metrics (
  metric_id           INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) NOT NULL FOREIGN KEY,
  epoch               INT NOT NULL,
  metrics             JSON NOT NULL,               -- 各类指标 (loss/accuracy/mAP等)
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY (job_id, epoch)
);
```

---

#### 2.2.11 训练工件表 (training_job_artifacts)
```sql
CREATE TABLE training_job_artifacts (
  artifact_id         INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) NOT NULL FOREIGN KEY,
  kind                VARCHAR(64) NOT NULL,        -- 工件类型 (weights/logs/etc)
  name                VARCHAR(255) NOT NULL,
  path                VARCHAR(500) NOT NULL,       -- 相对于 BASE_TRAINING_DIR
  size_bytes          BIGINT,
  sha256              VARCHAR(64),                 -- 文件完整性校验
  meta                JSON,
  created_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```

---

#### 2.2.12 模型部署表 (model_deployments)
```sql
CREATE TABLE model_deployments (
  deployment_id       INT PRIMARY KEY AUTO_INCREMENT,
  job_id              VARCHAR(36) NOT NULL FOREIGN KEY,
  deployment_name     VARCHAR(255) NOT NULL,
  platform            ENUM('local', 'docker', 'kubernetes', 'aws', 'azure', 'gcp') NOT NULL,
  endpoint_url        VARCHAR(500),
  deployment_config   JSON,
  status              ENUM('deploying', 'active', 'inactive', 'failed') NOT NULL,
  health_check_url    VARCHAR(500),
  deployed_at         DATETIME(6),
  updated_at          DATETIME(6) DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  is_active           BOOLEAN DEFAULT TRUE
);
```

---

#### 2.2.13 部署日志表 (deployment_logs)
```sql
CREATE TABLE deployment_logs (
  id                  INT PRIMARY KEY AUTO_INCREMENT,
  deployment_id       INT NOT NULL FOREIGN KEY,
  log_level           VARCHAR(20) NOT NULL,
  message             TEXT NOT NULL,
  timestamp           DATETIME(6) DEFAULT CURRENT_TIMESTAMP
);
```

---

### 2.3 ER 图 (关系映射)

```
┌─────────────────┐
│    datasets     │
├─────────────────┤
│ dataset_id (PK) │
│ dataset_name    │
│ dataset_path    │
│ dataset_type    │
│ description     │
└────────┬────────┘
         │ 1
         │ (一对多)
         │ N
         ▼
┌─────────────────┐         ┌──────────────────┐
│   projects      │────────▶│ model_architectures
├─────────────────┤ N    1  ├──────────────────┤
│ project_id (PK) │         │ arch_id (PK)     │
│ project_name    │         │ model_family     │
│ description     │         │ model_variant    │
│ dataset_id (FK) │         │ task_type        │
│ created_by      │         └──────────────────┘
│ is_active       │
└────────┬────────┘
         │ 1
         │ (一对多)
         │ N
         ▼
┌────────────────────────┐
│   training_jobs        │
├────────────────────────┤
│ job_id (PK)            │
│ project_id (FK)        │
│ arch_id (FK)           │
│ job_name               │
│ status                 │
│ progress               │
│ current_epoch          │
│ created_at/started_at  │
│ completed_at           │
└─────┬──────────┬───────┴────────┐
      │ 1        │ 1              │ 1
      │          │                │
      ▼          ▼                ▼
  ┌──────────────────────┐   ┌─────────────────┐   ┌──────────────────┐
  │ training_parameters  │   │ training_results│   │ model_deployments│
  ├──────────────────────┤   ├─────────────────┤   ├──────────────────┤
  │ param_id (PK)        │   │ result_id (PK)  │   │deployment_id(PK) │
  │ job_id (FK, unique)  │   │ job_id (FK,uni) │   │ job_id (FK)      │
  │ epochs               │   │ best_weights    │   │ deployment_name  │
  │ batch_size           │   │ results_dir     │   │ platform         │
  │ image_size           │   │ final_metrics   │   │ endpoint_url     │
  │ learning_rate        │   │ best_metrics    │   │ status           │
  │ ...                  │   │ model_size_mb   │   │ is_active        │
  └──────────────────────┘   └─────────────────┘   └──────────────────┘
                                                           │ 1
                                                           │
                                                           ▼
                                                   ┌──────────────────┐
                                                   │ deployment_logs  │
                                                   ├──────────────────┤
                                                   │ id (PK)          │
                                                   │ deployment_id(FK)│
                                                   │ log_level        │
                                                   │ message          │
                                                   └──────────────────┘

关联表 (1:1 - 通过 job_id):
├─ training_job_controls
├─ training_job_meta
├─ training_job_events (1:N)
├─ training_job_epoch_metrics (1:N)
└─ training_job_artifacts (1:N)
```

---

### 2.4 数据库初始化流程

初始化文件位置: `train_platform/db/init_db.py`

**自动执行流程**:
1. 应用启动时调用 `init_db()`
2. 自动创建所有表结构
3. 可插入初始模型架构数据

---

## 三、API 接口详细说明

### 3.1 API 基础配置
- **基础 URL**: `/api/v1`
- **默认端口**: 18000
- **CORS 策略**: 允许所有来源 (`allow_origins=["*"]`)

### 3.2 静态资源挂载
- `/static/datasets` → `BASE_DATASETS_DIR`
- `/static/training` → `BASE_TRAINING_DIR`  
- `/static/temp` → `BASE_TEMP_DIR`

---

### 3.3 数据集管理 API (`/api/v1/datasets`)

#### 3.3.1 获取数据集列表
```http
GET /api/v1/datasets?skip=0&limit=100
```
**响应**: `List[Dataset]`

---

#### 3.3.2 创建新数据集
```http
POST /api/v1/datasets
Content-Type: application/json

{
  "dataset_name": "my_detection_dataset",
  "dataset_path": "path/to/dataset",
  "dataset_type": "detection",
  "description": "Optional description"
}
```
**响应**: `Dataset` (200 OK)

---

#### 3.3.3 获取数据集详情
```http
GET /api/v1/datasets/{dataset_id}
GET /api/v1/datasets/{dataset_id}/detail
```
**响应**: `Dataset` 或 `DatasetDetail` (包含完整统计信息)

---

#### 3.3.4 获取数据集统计信息
```http
GET /api/v1/datasets/{dataset_id}/statistics
```
**响应**: 
```json
{
  "dataset_id": 1,
  "total_images": 5000,
  "total_size_mb": 2500,
  "annotations_count": 25000
}
```

---

#### 3.3.5 删除数据集
```http
DELETE /api/v1/datasets/{dataset_id}
```
**参数**: `delete_files=true` (删除关联文件)

**响应**:
```json
{
  "message": "Dataset deleted successfully",
  "dataset_name": "my_detection_dataset",
  "files_deleted": 123
}
```

---

#### 3.3.6 上传数据集文件
```http
POST /api/v1/datasets/upload
Content-Type: multipart/form-data

File: dataset.zip
Params:
  - dataset_name: "my_dataset"
  - dataset_type: "detection"
```
**限制**: 最大 5000MB

**响应**: 上传成功的文件信息

---

### 3.4 项目管理 API (`/api/v1/projects`)

#### 3.4.1 获取项目列表
```http
GET /api/v1/projects?skip=0&limit=100&dataset_id=1
```
**查询参数**:
- `skip`: 分页偏移
- `limit`: 分页大小
- `dataset_id`: 按数据集过滤

**响应**: `List[Project]`

---

#### 3.4.2 创建新项目
```http
POST /api/v1/projects
Content-Type: application/json

{
  "project_name": "detection_project_v1",
  "description": "Object detection project",
  "dataset_id": 1
}
```
**响应**: `Project` (201 Created)

---

#### 3.4.3 获取项目详情
```http
GET /api/v1/projects/{project_id}
```
**响应**: `ProjectDetail` (包含所有关联的训练任务)

---

#### 3.4.4 更新项目
```http
PUT /api/v1/projects/{project_id}
Content-Type: application/json

{
  "project_name": "new_name",
  "description": "updated description",
  "is_active": true
}
```

---

#### 3.4.5 删除项目
```http
DELETE /api/v1/projects/{project_id}
```

---

#### 3.4.6 获取项目模型大小统计
```http
GET /api/v1/projects/{project_id}/model-size
```
**响应**:
```json
{
  "project_id": 1,
  "total_models": 5,
  "total_size_mb": 1200.5,
  "largest_model_mb": 450
}
```

---

### 3.5 模型架构 API (`/api/v1/architectures`)

#### 3.5.1 列出所有模型架构
```http
GET /api/v1/architectures
  ?family=YOLOv8
  &task_type=detection
  &q=yolov8n
```
**查询参数**:
- `family`: 按模型系列过滤 (YOLOv8/YOLOv11/MMDet等)
- `task_type`: detection/segmentation/classification
- `q`: 按模型变体模糊搜索

**响应**: `List[ModelArchitecture]`

**示例返回值**:
```json
[
  {
    "arch_id": 1,
    "model_family": "YOLOv8",
    "model_variant": "yolov8n",
    "task_type": "detection",
    "pretrained_path": "/models/yolov8n.pt",
    "description": "Nano variant"
  },
  {
    "arch_id": 2,
    "model_family": "YOLOv8",
    "model_variant": "yolov8m",
    "task_type": "detection",
    "pretrained_path": "/models/yolov8m.pt",
    "description": "Medium variant"
  }
]
```

---

### 3.6 训练任务管理 API (`/api/v1/training-jobs`)

#### 3.6.1 创建训练任务
```http
POST /api/v1/training-jobs
Content-Type: application/json

{
  "project_id": 1,
  "arch_id": 1,
  "job_name": "exp_01",
  "parameters": {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640,
    "learning_rate": 0.01,
    "patience": 50,
    "device": "cuda",
    "workers": 8,
    "use_pretrained": true,
    "optimizer": "AdamW",
    "augmentation": {
      "hsv_h": 0.015,
      "hsv_s": 0.7,
      "hsv_v": 0.4
    },
    "additional_params": {}
  }
}
```
**响应**: `TrainingJob` (201 Created)

---

#### 3.6.2 获取训练任务列表
```http
GET /api/v1/training-jobs
  ?project_id=1
  &status=running
  &skip=0
  &limit=100
```
**响应**: `List[TrainingJob]`

---

#### 3.6.3 获取单个训练任务
```http
GET /api/v1/training-jobs/{job_id}
```
**响应**: `TrainingJob`

---

#### 3.6.4 启动训练任务
```http
POST /api/v1/training-jobs/{job_id}/start
```
**响应**: `TrainingJob` (status = running)

**流程**:
1. 将任务加入 Worker 队列
2. 第一个空闲 Worker 认领任务
3. Worker 启动训练进程

---

#### 3.6.5 取消训练任务
```http
POST /api/v1/training-jobs/{job_id}/cancel
Content-Type: application/json

{
  "reason": "Optional cancellation reason"
}
```
**响应**: `TrainingJob` (status = cancelled)

---

#### 3.6.6 删除训练任务
```http
DELETE /api/v1/training-jobs/{job_id}
```

---

#### 3.6.7 获取训练参数
```http
GET /api/v1/training-jobs/{job_id}/parameters
```
**响应**: `TrainingParameters`

---

#### 3.6.8 获取训练状态
```http
GET /api/v1/training-jobs/{job_id}/status
```
**响应**:
```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": 45,
  "current_epoch": 45,
  "worker_id": "worker-1",
  "heartbeat_at": "2024-01-14T10:30:00Z"
}
```

---

#### 3.6.9 获取训练指标
```http
GET /api/v1/training-jobs/{job_id}/metrics
```
**响应**:
```json
{
  "loss": 0.123,
  "accuracy": 0.95,
  "mAP": 0.85,
  "precision": 0.92,
  "recall": 0.88
}
```

---

#### 3.6.10 获取详细的周期指标
```http
GET /api/v1/training-jobs/{job_id}/metrics/detailed
```
**响应**:
```json
{
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.5,
      "val_loss": 2.3,
      "train_acc": 0.5,
      "val_acc": 0.55
    },
    ...
  ],
  "best_epoch": 45,
  "best_metrics": { "mAP": 0.85 }
}
```

---

#### 3.6.11 列出训练工件
```http
GET /api/v1/training-jobs/{job_id}/artifacts
```
**响应**:
```json
[
  {
    "artifact_id": 1,
    "kind": "weights",
    "name": "best.pt",
    "path": "runs/detect/exp_01/weights/best.pt",
    "size_bytes": 456789,
    "sha256": "abc123...",
    "created_at": "2024-01-14T10:00:00Z"
  }
]
```

---

#### 3.6.12 列出训练事件
```http
GET /api/v1/training-jobs/{job_id}/events?limit=200
```
**响应**:
```json
[
  {
    "event_id": 1,
    "level": "INFO",
    "event_type": "epoch_start",
    "message": "Epoch 1 started",
    "data": { "epoch": 1 },
    "created_at": "2024-01-14T10:00:10Z"
  },
  ...
]
```

---

#### 3.6.13 获取/更新训练元数据
```http
GET /api/v1/training-jobs/{job_id}/meta

PATCH /api/v1/training-jobs/{job_id}/meta
Content-Type: application/json

{
  "group": "exp_batch_1",
  "tags": ["baseline", "production"],
  "notes": "Using standard augmentation"
}
```

---

#### 3.6.14 获取训练日志尾部
```http
GET /api/v1/training-jobs/{job_id}/logs/tail
  ?which=stdout&lines=200
```
**参数**:
- `which`: stdout | stderr
- `lines`: 1-2000

**响应**:
```json
{
  "job_id": "uuid",
  "which": "stdout",
  "lines": 200,
  "text": "... log content ..."
}
```

---

#### 3.6.15 WebSocket - 实时指标流
```websocket
WS /api/v1/training-jobs/{job_id}/metrics/stream
```
**功能**: 实时推送训练指标、进度更新、事件

**消息格式**:
```json
{
  "type": "metric",
  "data": {
    "epoch": 45,
    "progress": 45,
    "loss": 0.123,
    "mAP": 0.85
  }
}
```

---

### 3.7 模型部署 API (`/api/v1/deployments`)

#### 3.7.1 获取部署列表
```http
GET /api/v1/deployments?job_id=uuid&skip=0&limit=100
```
**响应**: `List[ModelDeployment]`

---

#### 3.7.2 创建新部署
```http
POST /api/v1/deployments
Content-Type: application/json

{
  "job_id": "uuid",
  "deployment_name": "prod_deployment_v1",
  "platform": "docker",
  "endpoint_url": "http://localhost:5000/predict",
  "deployment_config": {
    "container_image": "my_model:latest",
    "replicas": 3,
    "resources": {
      "memory": "4Gi",
      "cpu": "2"
    }
  }
}
```
**支持的平台**: local, docker, kubernetes, aws, azure, gcp

**响应**: `ModelDeployment`

---

#### 3.7.3 获取部署详情
```http
GET /api/v1/deployments/{deployment_id}
```
**响应**: `ModelDeployment`

---

#### 3.7.4 更新部署
```http
PUT /api/v1/deployments/{deployment_id}
Content-Type: application/json

{
  "deployment_name": "updated_name",
  "status": "active",
  "is_active": true
}
```

---

#### 3.7.5 删除部署
```http
DELETE /api/v1/deployments/{deployment_id}
```

---

### 3.8 推理 API (`/api/v1/inference`)

#### 3.8.1 运行推理
```http
POST /api/v1/inference/run
Content-Type: application/json

{
  "job_id": "uuid",
  "image_url": "http://example.com/image.jpg",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45
}
```
**响应**: `InferenceResult`

```json
{
  "job_id": "uuid",
  "inference_time_ms": 45.3,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [10, 20, 200, 300],
      "area": 52000
    }
  ],
  "image_url": "http://example.com/result.jpg"
}
```

---

#### 3.8.2 上传推理图片
```http
POST /api/v1/inference/upload
Content-Type: multipart/form-data

File: image.jpg
```
**支持格式**: jpg, jpeg, png, bmp, tiff, webp

**响应**:
```json
{
  "path": "/static/temp/uuid.jpg",
  "image_url": "http://localhost:18000/static/temp/uuid.jpg"
}
```

---

### 3.9 统计分析 API (`/api/v1/statistics`)

#### 3.9.1 获取项目统计
```http
GET /api/v1/statistics/projects
```
**响应**:
```json
{
  "total_projects": 10,
  "active_projects": 8,
  "total_jobs": 150,
  "completed_jobs": 100,
  "running_jobs": 20,
  "failed_jobs": 30
}
```

---

#### 3.9.2 获取部署统计
```http
GET /api/v1/statistics/deployments
```
**响应**:
```json
{
  "total_deployments": 15,
  "active_deployments": 12,
  "status_breakdown": {
    "active": 12,
    "inactive": 2,
    "failed": 1
  }
}
```

---

#### 3.9.3 获取模型大小统计
```http
GET /api/v1/statistics/model-sizes
```
**响应**:
```json
{
  "total_models": 50,
  "total_size_mb": 25000,
  "average_size_mb": 500,
  "distribution": [...]
}
```

---

## 四、核心功能说明

### 4.1 数据集管理功能
- ✅ 上传和存储多种格式的数据集
- ✅ 支持三种任务类型 (检测/分割/分类)
- ✅ 自动计算数据集统计信息
- ✅ 级联删除关联的项目

### 4.2 项目管理功能
- ✅ 创建独立的项目空间
- ✅ 关联数据集到项目
- ✅ 追踪项目创建者和时间
- ✅ 项目激活/禁用状态管理

### 4.3 训练管理功能
- ✅ 支持多个预训练模型架构
- ✅ 灵活的超参数配置
- ✅ 任务状态机 (pending → running → completed/failed/cancelled)
- ✅ Worker 分布式任务队列
- ✅ 心跳检测和超时管理
- ✅ 优雅的任务取消和删除
- ✅ 详细的日志记录 (stdout/stderr)

### 4.4 监控与跟踪功能
- ✅ 实时训练进度更新
- ✅ 每个 epoch 的详细指标
- ✅ 训练事件日志系统
- ✅ WebSocket 实时指标推送
- ✅ 工件 (权重、日志、结果) 管理

### 4.5 推理功能
- ✅ 在线推理端点
- ✅ 图片上传和存储
- ✅ 结果可视化
- ✅ 批量推理支持 (可扩展)

### 4.6 部署功能
- ✅ 支持多种部署平台
- ✅ 部署配置管理
- ✅ 部署日志跟踪
- ✅ 健康检查 URL
- ✅ 灰度更新支持

---

## 五、项目结构详解

### 5.1 目录结构
```
train_platform/
├── app.py                      # FastAPI 应用入口
├── core/
│   └── config.py              # 配置管理 (Settings 类)
├── db/
│   ├── base.py                # SQLAlchemy Base 类
│   ├── session.py             # 数据库会话工厂
│   └── init_db.py             # 数据库初始化
├── models/                    # SQLAlchemy ORM 模型
│   ├── dataset.py
│   ├── project.py
│   ├── training.py
│   ├── deployment.py
│   ├── run_tracking.py
│   └── enums.py
├── schemas/                   # Pydantic 请求/响应模型
│   ├── dataset.py
│   ├── project.py
│   ├── training.py
│   ├── deployment.py
│   ├── inference.py
│   ├── run_tracking.py
│   └── common.py
├── repositories/              # 数据访问层
│   ├── base.py                # 基础 CRUD 仓库
│   ├── dataset_repo.py
│   ├── project_repo.py
│   ├── training_repo.py
│   └── deployment_repo.py
├── services/                  # 业务逻辑层
│   ├── dataset_service.py
│   ├── project_service.py
│   ├── training_service.py
│   ├── deployment_service.py
│   ├── inference_service.py
│   └── file_service.py
├── api/
│   ├── deps.py                # 依赖注入 (get_db 等)
│   └── v1/                    # API v1 端点
│       ├── datasets.py
│       ├── projects.py
│       ├── training.py
│       ├── deployments.py
│       ├── inference.py
│       ├── architectures.py
│       └── statistics.py
├── workers/
│   ├── worker.py              # 任务 Worker 进程
│   └── training/
│       └── train_entry.py     # 训练入口点
├── training/
│   ├── registry.py            # 模型架构注册表
│   └── plugins/
│       ├── base.py            # 训练引擎基类
│       ├── ultralytics_yolo.py # YOLOv8 实现
│       └── stubs.py           # 其他模型存根
└── utils/
    ├── exceptions.py          # 自定义异常
    └── path_utils.py          # 路径工具函数
```

### 5.2 关键类和接口

#### 5.2.1 Settings (配置)
```python
@dataclass(frozen=True)
class Settings:
    mysql_*              # MySQL 连接参数
    home_dir            # 项目根目录
    datasets_dir        # 数据集存储目录
    training_dir        # 训练结果存储目录
    temp_dir            # 临时文件目录
    database_url        # MySQL 连接字符串
```

#### 5.2.2 BaseRepository (基础仓库)
```python
class BaseRepository:
    def get(db, id)              # 按 ID 获取单条
    def get_multi(db, skip, limit) # 分页获取
    def create(db, obj_in)       # 创建
    def update(db, db_obj, obj_in) # 更新
    def delete(db, id)           # 删除
```

#### 5.2.3 TrainingService (训练服务)
```python
class TrainingService:
    def create_training_job_from_params() # 创建训练任务
    def enqueue_training_job()            # 加入队列
    def request_cancel()                  # 请求取消
    def delete_training_job()             # 删除任务
    def get_training_status()             # 获取状态
    def get_training_metrics()            # 获取指标
    def get_meta()                        # 获取元数据
    def update_meta()                     # 更新元数据
    def list_events()                     # 列出事件
    def list_artifacts()                  # 列出工件
```

---

## 六、技术栈和依赖

### 6.1 核心依赖
- **FastAPI** (v0.111.0) - Web 框架
- **Uvicorn** (v0.29.0) - ASGI 服务器
- **SQLAlchemy** (v2.0.30) - ORM
- **PyMySQL** (v1.1.0) - MySQL 驱动

### 6.2 ML/训练相关
- **PyTorch** (v2.3.1) - 深度学习框架
- **torchvision** (v0.18.1) - 视觉工具库
- **ultralytics** (v8.2.58) - YOLOv8 实现
- **Pillow** (v10.4.0) - 图像处理
- **OpenCV** (v4.9.0.80) - 计算机视觉库

### 6.3 数据处理
- **pandas** (v2.2.2) - 数据分析
- **PyYAML** (v6.0.1) - 配置文件解析
- **requests** (v2.32.3) - HTTP 客户端

### 6.4 安全和工具
- **cryptography** (>=46.0.3) - 密码学库
- **python-dotenv** (v1.0.1) - 环境变量管理
- **python-multipart** (v0.0.9) - 文件上传处理

---

## 七、运行和部署

### 7.1 本地开发运行
```bash
# 配置环境变量
cd backend_v2
cp .env.example .env

# 安装依赖
pip install -e .

# 启动 API 服务器 (端口 18000)
python -m uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload

# 在另一个终端启动 Worker
python -m train_platform.workers.worker
```

### 7.2 Docker Compose 运行
```bash
cd backend_v2
cp .env.example .env
docker compose up --build
```

### 7.3 环境变量配置
```
# MySQL 配置
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=train_backend_v2

# 存储目录
TRAIN_PLATFORM_HOME=/path/to/backend_v2
BASE_DATASETS_DIR=/path/to/datasets
BASE_TRAINING_DIR=/path/to/training_runs
BASE_TEMP_DIR=/path/to/temp
```

---

## 八、设计模式和最佳实践

### 8.1 分层架构
```
API Layer (路由处理)
    ↓
Service Layer (业务逻辑)
    ↓
Repository Layer (数据访问)
    ↓
Database (SQLAlchemy ORM)
```

### 8.2 依赖注入
使用 FastAPI 的 `Depends()` 机制：
```python
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    # db 自动注入
```

### 8.3 异常处理
定义自定义异常：
- `NotFoundError` - 资源不存在
- `ConflictError` - 业务冲突 (如项目已存在)

### 8.4 CORS 配置
允许所有来源和方法，便于前端开发。

---

## 九、扩展性设计

### 9.1 可插拔的训练引擎
```python
# 位置: train_platform/training/plugins/
- base.py           # 抽象基类
- ultralytics_yolo.py # YOLOv8 实现
- stubs.py          # 其他引擎存根

# 未来可添加:
- mmdet_plugin.py   # MMDet 实现
- detr_plugin.py    # DETR 实现
```

### 9.2 支持新的部署平台
在 `DeploymentPlatform` 枚举中添加新平台，然后在部署服务中实现相应的逻辑。

### 9.3 支持新的任务类型
在 `TaskType` 和 `DatasetType` 枚举中添加新类型。

---

## 十、总结

本后端项目是一个现代化的 **AI 模型训练管理平台**，具有以下特点：

| 特性 | 说明 |
|------|------|
| **架构** | 分层设计，支持扩展 |
| **数据库** | MySQL + SQLAlchemy ORM，包含 13 个核心表 |
| **API** | RESTful + WebSocket，7 个主要模块 |
| **功能** | 数据集、项目、训练、部署、推理、统计分析 |
| **监控** | 实时进度、指标、事件、日志跟踪 |
| **分布式** | Worker 队列，支持多 Worker 并行训练 |
| **部署** | 支持本地、Docker、K8s、云平台 |
| **可扩展性** | 可插拔的训练引擎、部署平台 |

---

**文档版本**: v1.0  
**最后更新**: 2026-01-14  
**作者**: AI Code Assistant

