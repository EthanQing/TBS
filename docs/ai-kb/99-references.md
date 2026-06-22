# 参考信息

## 关键路径

- 应用入口：`train_platform/app.py`
- 配置：`train_platform/core/config.py`
- License 校验：`train_platform/core/license.py`
- DB session：`train_platform/db/session.py`
- DB 初始化：`train_platform/db/init_db.py`
- migration 配置：`alembic.ini`
- migration 目录：`train_platform/db/migrations/versions/`
- API 聚合：`train_platform/api/v3/__init__.py`
- 通用 schema：`train_platform/schemas/v3/common.py`
- 枚举：`train_platform/models/v3/enums.py`
- 异常：`train_platform/utils/exceptions.py`

## 常用命令

安装：

```bash
pip install -r requirements/backend.txt
pip install -e .
```

迁移：

```bash
alembic -c alembic.ini upgrade head
```

启动 API：

```bash
uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload
```

Worker：

```bash
python -m train_platform.workers.worker
python -m train_platform.workers.yolo_worker
python -m train_platform.workers.paddle_worker
python -m train_platform.workers.inference_worker
python -m train_platform.workers.paddle_inference_worker
```

知识库本地状态：

```bash
Get-ChildItem docs/ai-kb
git check-ignore -v docs/ai-kb/00-index.md
```

## 主要环境变量

- 数据库：`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`DATABASE_URL`、`ALEMBIC_DATABASE_URL`、`DB_POOL_SIZE`、`DB_MAX_OVERFLOW`、`DB_POOL_TIMEOUT`、`DB_POOL_RECYCLE`、`DB_POOL_PRE_PING`
- 运行目录：`TRAIN_PLATFORM_HOME`、`BASE_DATASETS_DIR`、`BASE_TRAINING_DIR`、`BASE_TEMP_DIR`、`BASE_UPLOAD_SESSIONS_DIR`、`BASE_DATASET_STAGING_DIR`、`BASE_IMPORTS_DIR`、`BASE_PRETRAIN_MODELS_DIR`、`PADDLE_DET_DIR`
- 数据导入/发布：`DATASET_IMPORT_ROOTS`、`DATASET_IMPORT_MAX_WORKERS`、`ILLEGAL_DATASET_PUBLISH_MAX_WORKERS`
- 上传：`UPLOAD_CHUNK_SIZE_MB`、`UPLOAD_SESSION_TTL_HOURS`、`UPLOAD_PART_MAX_RETRIES`、`UPLOAD_MAX_PARALLEL_PARTS`
- Worker：`WORKER_POLL_INTERVAL`、`WORKER_HEARTBEAT_INTERVAL`、`WORKER_STALE_AFTER_SECONDS`、`WORKER_BIND_HOST`
- 推理：`INTERNAL_API_TOKEN`、`INFERENCE_MAX_DOWNLOAD_BYTES`、`INFERENCE_DOWNLOAD_TIMEOUT_SEC`、`INFERENCE_ALLOWED_SCHEMES`、`INFERENCE_ALLOWED_HOSTS`
- 缩略图/索引：`THUMBNAIL_MAX_WORKERS`、`THUMBNAIL_FIRST_PAGE_PREWARM`、`THUMBNAIL_SIZE`、`VIEW_INDEX_MAX_WORKERS`
- ID 起始段：`ILLEGAL_DATASET_ID_START`、`STANDARD_DATASET_ID_START`
- MLflow：`MLFLOW_ENABLE`、`MLFLOW_TRACKING_URI`、`MLFLOW_EXPERIMENT_NAME`
- License：`TRAIN_PLATFORM_LICENSE_REQUIRED`、`TRAIN_PLATFORM_LICENSE_PATH`、`TRAIN_PLATFORM_LICENSE_DATA_B64`、`TRAIN_PLATFORM_LICENSE_DATA`

`PADDLE_DET_DIR` 必须指向完整 PaddleDetection `release/2.6` 源码 checkout；未设置时默认使用后端目录下的 `PaddleDetection/`。

数据库连接池默认 `DB_POOL_SIZE=20`、`DB_MAX_OVERFLOW=30`、`DB_POOL_TIMEOUT=60`、`DB_POOL_RECYCLE=300`，`DB_POOL_PRE_PING` 默认启用。单后端容器可使用默认值；多副本部署时要结合 MySQL `max_connections` 调整每个实例的池大小。

在外层 compose 部署中，backend 默认把宿主 `./TBS/imports` 挂载到容器 `/app/imports`，对应默认 `BASE_IMPORTS_DIR`。离线目录导入只看 backend 容器可见路径；额外宿主路径需要挂载进容器并通过 `DATASET_IMPORT_ROOTS` 暴露。

## 人工文档索引

- 总览：`docs/00_概述.md`
- 项目：`docs/01_项目管理.md`
- 数据集：`docs/02_数据集管理.md`
- 训练：`docs/03_训练任务.md`
- 模型版本：`docs/04_模型版本.md`
- 部署：`docs/05_模型部署.md`
- 推理：`docs/06_推理服务.md`
- 在线服务：`docs/07_在线服务.md`
- 数据转换与增强：`docs/08_数据集转换与增强.md`
- 模型转换：`docs/09_模型转换.md`
- 监控告警：`docs/10_系统监控与告警.md`
- 辅助接口：`docs/11_辅助接口.md`
- 合格模型：`docs/12_合格模型.md`

## 依赖入口

- 项目元数据：`pyproject.toml`
- 锁文件：`uv.lock`
- 后端依赖：`requirements/backend.txt`
- Worker 依赖：`requirements/worker*.txt`
- 打包脚本：`setup.py`
- 受保护运行时构建：`train_platform/core/build_protected_runtime.py` 调用 `setup.py build_ext`，用 Cython 编译 `services/` 和 `workers/` 核心实现。
- Docker：`Dockerfile.backend`、`Dockerfile.worker.yolo`、`Dockerfile.worker.paddle`、`docker/entrypoint.sh`
