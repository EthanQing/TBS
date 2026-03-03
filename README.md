## Backend v2 (rewrite)

目标：完全放弃旧 `backend/` 目录，重写后端，并把训练能力做成可插拔的 engine（YOLO/MMDet/DETR…）。

API：`/api/v2`（已推倒重来，不兼容旧接口）

### 运行（Docker Compose）
```bash
cd backend_v2
copy .env.example .env
docker compose up --build
```

### 运行（本地）
```bash
# 1) 配置环境变量
copy .env.example .env

# 2) 安装依赖
pip install -r requirements.txt

# 3) 初始化数据库（Alembic 迁移）
# 需要提前创建好 MYSQL_DATABASE 指定的库（默认：train_backend_v2）
# 本版本不兼容旧库，建议直接删库重建后再执行。
python -m alembic -c alembic.ini upgrade head

# 4) 启动 API
python -m uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload
```

Worker：
```bash
python -m train_platform.workers.worker
```

### 环境变量（可选）
- `MYSQL_USER`/`MYSQL_PASSWORD`/`MYSQL_HOST`/`MYSQL_PORT`/`MYSQL_DATABASE`
- `TRAIN_PLATFORM_HOME`：数据根目录（默认：仓库 `backend_v2/`）
- `BASE_DATASETS_DIR` / `BASE_TRAINING_DIR` / `BASE_TEMP_DIR`：覆盖默认路径

### Paddle local development

If you are developing Paddle training locally (without Docker), see:

- `docs/paddle_local_dev.md`

Recommended worker entrypoint for Paddle jobs:

```bash
python -m train_platform.workers.paddle_worker
```
