# AGENTS.md

适用范围：整个仓库。

你正在一个带有本地知识库的长期项目中工作。这个项目是智能训练平台后端，核心代码位于 `train_platform/`，运行时以代码中的 `/api/v3` 为准。

## 外层工作区

如果当前 `TBS` 位于外层 `train` 工作区下，开始任务时先阅读 `../docs/ai-kb/00-index.md` 获取前端、后端和部署的全局上下文，然后再阅读本文件与 `docs/ai-kb/00-index.md`。

## 开始任务前

1. 先阅读本文件。
2. 阅读 `docs/ai-kb/00-index.md`。
3. 根据任务内容，只阅读相关知识库文档。
4. 不要默认扫描整个知识库。
5. 修改代码前先说明计划。
6. 优先遵守现有架构和约定。
7. 如果知识库与代码不一致，以代码为准，并更新知识库。

## 修改代码时

- API 层优先查看 `train_platform/api/v3/`、`train_platform/schemas/v3/`、`train_platform/services/v3/`、`train_platform/models/v3/`。
- 数据库 schema 变更必须走 Alembic migration，不要依赖 `Base.metadata.create_all()`。
- 配置项集中在 `train_platform/core/config.py`，新增配置后同步更新 `.env.example`、README 或知识库引用。
- Worker 相关逻辑优先查看 `train_platform/workers/` 和 `train_platform/training/`。
- 文档中的旧 `/api/v2` 路径可能过时；实际注册路径以 `train_platform/app.py` 和 `train_platform/api/v3/__init__.py` 为准。

## 完成任务后

1. 检查是否需要更新 `docs/ai-kb/`。
2. 更新相关模块文档。
3. 如果流程变化，更新 `docs/ai-kb/04-runtime-workflows.md` 或 `docs/ai-kb/05-dev-workflows.md`。
4. 如果发现新坑，更新 `docs/ai-kb/90-pitfalls.md`。
5. 如果配置、命令、路径变化，更新 `docs/ai-kb/99-references.md`。
6. 如果本次修改影响 Docker 镜像内容，构建更新后的相关镜像：
   - 后端 API、service、model、migration、配置、依赖或 `Dockerfile.backend` 变化：运行 `docker build -f Dockerfile.backend -t tbs-backend:plain .`。
   - YOLO worker、训练插件、推理 worker、worker 依赖或 `Dockerfile.worker.yolo` 变化：运行 `docker build -f Dockerfile.worker.yolo -t tbs-worker-yolo:plain .`。
   - Paddle worker、PaddleDetection 相关逻辑、worker 依赖或 `Dockerfile.worker.paddle` 变化：运行 `docker build -f Dockerfile.worker.paddle -t tbs-worker-paddle:plain .`。
   - 仅文档、测试或不会进入镜像的运行时数据变化不需要构建镜像。
7. 在最终回复中说明更新了哪些知识库文件，以及构建了哪些镜像；如果跳过构建，说明原因。

除非用户明确要求，否则不要重写整个知识库。
