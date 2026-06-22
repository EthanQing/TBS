# 开发工作流

本文件是 agent 修改项目时的检查清单。

## 新增或修改 API

1. 在 `train_platform/api/v3/` 找到对应 router，确认 prefix。
2. 在 `schemas/v3/` 新增或调整请求/响应 schema。
3. 把业务逻辑放进 `services/v3/`，router 只做 HTTP 编排。
4. 如涉及数据库，更新 `models/v3/` 和 Alembic migration。
5. 如已有 repository，优先复用 `repositories/v3/`。
6. 确认 `api/v3/__init__.py` 是否需要注册新 router。
7. 更新 `docs/ai-kb/03-domain-modules.md` 和相关人工文档引用。

## 数据库变更

1. 修改 SQLAlchemy model。
2. 新增 Alembic migration 到 `train_platform/db/migrations/versions/`。
3. 不要让 `init_db()` 创建业务表。
4. 如果新增 seed 数据，把逻辑放在 `db/init_db.py` 或 `db/seed_data.py` 的合适位置。
5. 检查枚举、默认值、索引和迁移降级路径。

## 新增配置项

1. 在 `train_platform/core/config.py` 的 `Settings` 中添加字段。
2. 如果是路径配置，考虑 `Path(...).resolve()` 和 `ensure_dirs()`。
3. 更新 `.env.example`。
4. 更新 `docs/ai-kb/99-references.md`。
5. 如果影响部署或大文件目录，更新 `90-pitfalls.md`。

## Worker 或后台任务

1. 先确定任务状态落在哪里：数据库、文件状态，还是两者都有。
2. 检查取消、重试、stale lock、heartbeat、日志和最终状态。
3. 保持 API 查询任务状态的响应模型稳定。
4. 修改训练任务时同时检查通用 worker、YOLO worker、Paddle worker。
5. 修改推理任务时同时检查 YOLO 和 Paddle 推理 worker。

## 镜像构建

完成工作后，如果修改会进入 Docker 镜像，需要构建更新后的相关镜像：

- 后端 API、service、model、migration、配置、依赖或 `Dockerfile.backend` 变化：`docker build -f Dockerfile.backend -t tbs-backend:plain .`
- YOLO worker、训练插件、推理 worker、worker 依赖或 `Dockerfile.worker.yolo` 变化：`docker build -f Dockerfile.worker.yolo -t tbs-worker-yolo:plain .`
- Paddle worker、PaddleDetection 相关逻辑、worker 依赖或 `Dockerfile.worker.paddle` 变化：`docker build -f Dockerfile.worker.paddle -t tbs-worker-paddle:plain .`
- 仅文档、测试或不会进入镜像的运行时数据变化不需要构建镜像，但最终回复要说明跳过原因。

Docker 客户镜像统一使用 `.pyc` 保护核心代码。`build_protected_runtime` 会复制运行时源码树，然后把 `train_platform/services/` 和少量训练 worker 核心文件编译为同目录 `.pyc` 并删除对应 `.py`。API、migration、FastAPI sidecar、推理任务和模型转换 worker 保持 `.py`，避免框架反射或 `python -m` 入口失败。默认受保护 worker 文件是 `worker_impl.py`、`yolo_worker_impl.py`、`paddle_worker_impl.py`、`workers/training/train_entry_impl.py`、`workers/training/vdl_bridge.py`。

## 训练插件

1. 先看 `training/plugins/base.py` 的协议。
2. 在 `training/registry.py` 确认插件注册和选择规则。
3. 新增插件应提供配置 schema、校验/归一化逻辑和训练执行入口。
4. 训练产物、日志、metrics 和异常状态要与 `TrainingRunService` 兼容。

## 文档维护

- 新增模块：更新 `00-index.md` 和 `03-domain-modules.md`。
- 改运行流程：更新 `04-runtime-workflows.md`。
- 改开发约定：更新本文件。
- 发现坑点：更新 `90-pitfalls.md`。
- 改命令、路径、环境变量：更新 `99-references.md`。

## 建议验证

文档类修改：

```bash
Get-ChildItem docs/ai-kb
git check-ignore -v docs/ai-kb/00-index.md
```

后端代码修改：

```bash
python -m compileall train_platform
alembic -c alembic.ini upgrade head
uvicorn train_platform.app:app --host 0.0.0.0 --port 18000 --reload
```

是否运行完整命令取决于任务风险、环境可用性和用户要求。
