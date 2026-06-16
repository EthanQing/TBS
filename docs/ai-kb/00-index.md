# 本地项目知识库索引

这个目录是给 Codex agent 使用的本地知识库。它帮助 agent 快速理解项目、定位代码、识别常见风险，并在后续任务中持续维护。

## 事实优先级

当信息冲突时，按下面顺序判断：

1. 当前代码：`train_platform/`、`alembic.ini`、`train_platform/db/migrations/`。
2. 当前配置样例：`.env.example`、`pyproject.toml`、`requirements/`。
3. 本知识库：`docs/ai-kb/`。
4. README。
5. 历史人工文档：当前工作区已清理非 KB 文档，若后续恢复再以代码为准。

当前重要事实：应用在 `train_platform/app.py` 中注册 `train_platform.api.v3`，统一前缀是 `/api/v3`。当前工作区的普通人工文档已清理；如遇历史材料写 `/api/v2`，以代码为准。

## 阅读顺序

- 通用上手：读 `01-project-overview.md`。
- 改 API、schema、service、model：读 `02-architecture-map.md` 和 `03-domain-modules.md`。
- 改训练、推理、部署、上传、导入等流程：读 `04-runtime-workflows.md`。
- 新增功能或迁移：读 `05-dev-workflows.md`。
- 排查异常或避免踩坑：读 `90-pitfalls.md`。
- 找命令、路径、环境变量和外部文档：读 `99-references.md`。

## 代码入口地图

- 应用入口：`train_platform/app.py`
- 路由聚合：`train_platform/api/v3/__init__.py`
- API 路由：`train_platform/api/v3/`
- Pydantic schema：`train_platform/schemas/v3/`
- 业务服务：`train_platform/services/v3/`
- Repository：`train_platform/repositories/v3/`
- SQLAlchemy model：`train_platform/models/v3/`
- 配置：`train_platform/core/config.py`
- 数据库 session：`train_platform/db/session.py`
- 初始化与种子数据：`train_platform/db/init_db.py`
- Alembic migration：`train_platform/db/migrations/versions/`
- 训练插件：`train_platform/training/plugins/`
- Worker：`train_platform/workers/`

## 模块导航

- 项目、架构、框架插件：`projects`、`architectures`、`frameworks`
- 数据集：`standard-datasets`、`illegal-datasets`、`dataset-imports`、`dataset-upload-tasks`
- 数据增强：`standard-datasets/{id}/augmentations`
- 训练：`training-runs`、`training-reports`
- 模型资产：`model-versions`、`qualified-models`、`pretrain-models`
- 部署与在线服务：`deployments`、`deployment-runs`、`serving`
- 推理：`inference-runs`、`inference-jobs`
- 模型转换：`model-conversions`
- 监控告警：`system-metrics`、`alarms`
- 辅助能力：`stats`、`thumbnails`、`chart-configs`

## 维护规则

- 只更新与任务相关的知识库文件。
- 发现代码与知识库不一致时，修正知识库。
- 新增配置、命令、外部路径时更新 `99-references.md`。
- 新增流程、状态机、后台任务时更新 `04-runtime-workflows.md`。
- 新增常见误区或部署风险时更新 `90-pitfalls.md`。
