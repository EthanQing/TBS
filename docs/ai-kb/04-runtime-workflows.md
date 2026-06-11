# 运行时流程

本文件记录 agent 修改流程代码前应先理解的主路径。

## 应用启动

1. `uvicorn train_platform.app:app` 导入 app。
2. `create_app()` 创建 FastAPI 实例，注册 `/api/v3` router 和静态目录。
3. lifespan 启动时调用 `settings.ensure_dirs()` 创建运行目录。
4. 调用 license 校验。
5. `init_db()` 检查 v3 表是否完整并写入默认架构、默认告警规则。
6. 清理过期数据集上传 session。

如果数据库未迁移到位，启动阶段会报缺表，并提示运行 `alembic -c alembic.ini upgrade head`。

## 数据集上传与导入

- 普通上传、分片上传和离线导入最终都进入数据集 service 的解包、校验、索引、统计和版本更新流程。
- 分片上传状态持久化在数据库和 `BASE_UPLOAD_SESSIONS_DIR`。
- 离线导入根目录来自 `BASE_IMPORTS_DIR` 和 `DATASET_IMPORT_ROOTS`。
- 上传/导入完成后通常返回 `task_id`，前端通过 `/api/v3/dataset-upload-tasks/{task_id}` 查询后台进度。
- ZIP 上传任务会先进入 `extracting` 阶段，`safe_extract_zip` 按已解压文件数回调更新任务进度；解压完成后再进入 `validating`、索引和版本更新流程。违规数据集版本创建会在导入事务之外更新任务进度，依次暴露 `validating`、`materializing`、`indexing`、`finalizing`，但不再生成样本预览索引或缩略图，避免大数据集刷新后长时间停在 75%。

修改这条链路时同时检查标准数据集和违规数据集是否需要保持一致。

## 违规数据集发布

- 违规数据集维护原始标签、标签映射、版本和事件。
- 发布逻辑位于 `illegal_dataset_publish_service.py`。
- 发布任务逻辑位于 `illegal_dataset_publish_job_service.py`。
- 前端应通过 `/api/v3/illegal-datasets/{id}/publish-jobs` 创建后台发布任务，并轮询 `/publish-jobs/{job_id}` 展示 `phase`、`progress`、`processed/total`、`logs` 和 `error_message`；同步 `/publish` 接口已移除。
- 发布任务以数据库表 `illegal_dataset_publish_jobs` 为状态源，并把状态镜像写到 `temp/illegal_dataset_publish_jobs/<dataset_id>/<job_id>/status.json` 兼容旧轮询排查。幂等键由源违规数据集、源版本、最终生效标签映射、过滤、切片、拆分和 publish_config 生成，排除 `name`/`description`；同一请求重复提交返回已有 queued/running/completed 任务，failed/cancelled 可重置后重试。
- 发布转换会先按图片/JSON 基名配对，并兼容 `images/`、`json/`、`annotations/` 等顶层目录别名；缺图片或缺 JSON 的孤立文件会记录为 skipped/warnings 后跳过，不阻塞还有有效成对样本的发布。
- 违规数据集版本统一使用 `manifest_path`；挂载导入的图片条目可引用挂载源文件，生成的 YOLO labels 和配置文件仍记录在 manifest 中。历史 `manifest_path` 为空的版本不再支持访问。违规数据集不再对外提供文件列表、原图打开、图片标注查看、样本预览和缩略图；发布转换仍由后端内部读取 manifest 和源文件。
- 遥感大图或窗口读取相关逻辑也在发布 service 中，修改时注意内存和切片边界。

## 训练任务

1. API 创建训练任务，写入 `TrainingRun` 及参数。
2. 任务被 queue 后进入数据库队列。
3. `workers/worker.py` 或框架专用 worker 轮询任务。
4. Worker 启动 `workers/training/train_entry.py` 子进程。
5. `train_entry.py` 根据 framework registry 获取训练插件。
6. 插件执行训练并写入事件、日志、epoch metrics 和 artifacts。
7. Worker 维护 heartbeat、取消、失败和最终状态。

训练相关状态枚举在 `models/v3/enums.py` 的 `TrainingRunStatus`。

## 推理任务

- 轻量推理由 API service 调用内部 worker 能力完成。
- 批量或视频推理任务由 `inference_job_service.py` 和 `workers/inference_job_task.py` 管理。
- 推理任务状态、结果和渲染产物通常落在 `BASE_TEMP_DIR` 下。
- 内部 worker HTTP 请求可能需要 `INTERNAL_API_TOKEN`。

## 模型转换

- API 创建转换任务后，任务状态文件位于 `temp/model_conversions`。
- YOLO worker 会轮询队列并执行 PT/PTH 到 ONNX 的转换。
- 转换逻辑和性能测试在 `workers/model_conversion_task.py`。
- 队列锁和 stale lock 处理在 `workers/model_conversion_queue.py`。

## 部署运行

- 部署实体由 `deployment_service.py` 管理。
- 部署执行生成 `DeploymentRun`。
- `deployment_runtime_service.py` 负责运行阶段推进。
- `deployment_adapters.py` 提供实际部署适配层，目前以本地 gateway 适配为核心。
- 阶段枚举在 `DeploymentRunPhase`。

## 系统监控与告警

- 系统监控 service 采集 CPU、内存和 GPU 指标。
- 告警 service 管理规则、评估、活跃告警、确认和历史。
- 应用启动时会尝试 seed 默认告警规则。
