# 常见坑点

## API 版本容易混淆

README 顶部和旧人工 docs 中仍有 `/api/v2` 描述，但当前代码在 `train_platform/app.py` 中注册 `api.v3`，统一前缀是 `/api/v3`。改接口或排查路由时以代码为准。

## `docs/` 当前被 `.gitignore` 忽略

`.gitignore` 包含 `docs/`。已有 `docs/*.md` 是已跟踪文件，但新建的 `docs/ai-kb/` 默认是本地专用、不会自动进入版本控制。若未来要团队共享，需要单独调整 ignore 规则。

## 初始化不会自动建业务表

`train_platform/db/init_db.py` 明确不调用 `Base.metadata.create_all()`。缺表时应运行 Alembic migration，而不是在启动逻辑里偷偷建表。

## Windows 路径分隔

`DATASET_IMPORT_ROOTS` 支持逗号和分号，不按冒号拆分，因为 Windows 盘符含冒号。新增 path list 配置时参考 `core/config.py` 的 `_path_list_env()`。

## 静态目录必须存在

Starlette 挂载 `StaticFiles` 前要求目录存在。`create_app()` 和 lifespan 都会调用 `settings.ensure_dirs()`。新增静态挂载时要同步创建目录。

## 大文件目录不是源码

`datasets/`、`training_runs/`、`temp/`、`runs/`、`PaddleDetection/`、模型权重和日志通常是运行时产物或大文件。修改功能时不要把这些目录当成源码处理。

## 依赖版本有强约束

`pyproject.toml` 为 Windows CUDA 11.8 生态固定了 Torch、PaddlePaddle、PaddleDetection、numpy、opencv、protobuf 等版本，并通过 `tool.uv.override-dependencies` 覆盖 opencv 冲突。升级依赖前要同时验证 YOLO、PaddleDetection 和 ONNX Runtime。

## PaddleDetection 本地路径

`PADDLE_DET_DIR` 默认指向仓库下 `PaddleDetection/`。该目录通常是本地 checkout 且被忽略。Paddle 相关代码可能依赖本地 ppdet 可导入。

## 内部推理鉴权

推理 worker 暴露内部接口时使用 `X-Internal-Token` 或相关 Bearer token 校验。`INTERNAL_API_TOKEN` 为空时要确认当前部署是否允许这种模式。

## 上传与导入是后台流程

大数据集上传或离线导入完成接口可能只返回 `task_id`，真正的解包、校验、索引和版本更新在后台继续。前端和测试应查询 `dataset-upload-tasks`。

## 标准数据集文件存在但页面显示空

标准数据集详情页依赖 `standard_dataset_images` 和 `.dataset_stats.json` / `.dataset_view_index.json`，不是直接实时扫目录。若本地 `datasets/standard/<id>/images` 有文件、卡片有大小但图片数/目标数为 0，通常是发布或导入过程中索引/缓存未刷新完整。后端应在统计或视图缓存为 0 但数据库索引或目录存在图片时自动重建索引与缓存；发布转换标准数据集时避免在复制文件和索引完成前提前提交半成品数据集。

## 标准数据集主键必须由数据库生成

`standard_datasets.standard_dataset_id` 依赖数据库自增和迁移设置的 ID 段。不要在应用层用 `MAX(standard_dataset_id)+1` 或固定值生成主键；并发转换/重复提交时这种写法会让多个任务拿到同一个 ID，触发 `Duplicate entry ... for key 'standard_datasets.PRIMARY'`。

## 违规数据集追加导入必须创建新版本

违规数据集的归档上传、目录导入、挂载导入和直接图片上传都会创建新的 `illegal_dataset_versions` 记录。`append=True` 时应以当前最大 `version` 递增，并在创建前锁定对应 `illegal_datasets` 行，避免多个后台任务并发拿到相同版本号。若日志出现 `(1062, "Duplicate entry '<dataset_id>-1' for key 'illegal_dataset_versions.uq_illegal_dataset_versions_dataset_version'")`，通常是运行代码仍复用 `v1` 或旧服务未更新，应先确认后端进程/镜像和版本创建逻辑。

## 违规数据集版本必须有 manifest

详情统计和发布都依赖 `illegal_dataset_versions.manifest_path`。挂载导入也必须生成 manifest，图片条目可以引用挂载源文件。违规数据集已取消文件列表、原图打开、图片标注查看、样本预览和缩略图生成，不要为了恢复浏览能力而恢复 `snapshot_path`、无 manifest 历史版本目录扫描或逐文件 exists 检查，否则 200-300G 或大量小文件数据集会重新出现每次进详情等待十几秒的问题。

## 违规数据集列表应容忍坏版本

`GET /api/v3/illegal-datasets` 是管理页入口，不应因为某一条历史/异常数据集的 active version 缺少 manifest 而整体 404。列表构建统计时应对 `NotFoundError` / manifest 校验错误降级为空统计；违规数据集预览图固定为空，且不提供文件浏览接口。详情、统计和发布接口仍应保留对坏版本的明确错误。

## 数据集列表排序需要数据库索引

标准数据集和违规数据集列表接口按 `updated_at DESC` 分页。生产 MySQL 必须应用包含 `ix_standard_datasets_updated_at` / `ix_illegal_datasets_updated_at` 的 Alembic migration；缺索引时，`GET /api/v3/standard-datasets` 可能报 `(1038, 'Out of sort memory, consider increasing server sort buffer size')`，导致发布/转换后的标准数据集已经入库但前端列表 500、刷新也看不到。优先补迁移和索引，再考虑调大 MySQL sort buffer。

## Alembic revision id 可能超过版本表列宽

旧库的 `alembic_version.version_num` 可能是 `VARCHAR(32)`。如果 migration 使用较长 revision id，MySQL 会在迁移 DDL 执行完成、Alembic 更新版本号时抛 `(1406, "Data too long for column 'version_num' at row 1")`。`0018_illegal_publish_jobs_idempotency` 已在 `upgrade()` 开头将 MySQL 版本表扩到 `VARCHAR(255)`；后续新增 migration 时也应避免过长 revision id，或在需要时先扩版本表列宽。

## 挂载导入的内部清单不是业务标注

通过文件挂载导入的 LabelMe/JSON 违规数据集会生成 `.mounted_manifest.json`，它只记录原始挂载路径、链接方式和图片列表。发布标准数据集或排查 `No image/json pairs found` 时不要把它当标注 JSON；发布逻辑应回到版本 `meta.source_root` 指向的原始挂载目录做图片/JSON 配对和标签映射。若原始挂载目录已移走或不在允许导入根下，应先恢复挂载路径或重新导入。

## 违规数据集标签映射要按规范化 key 去重

`illegal_dataset_label_mappings` 对 `illegal_dataset_id + raw_label` 有唯一约束。原始标签可能混入全角百分号、零宽字符或首尾空白；保存映射时后端会按规范化 key 合并重复项，并在事务内替换当前数据集的旧映射。MySQL 下写入使用 `ON DUPLICATE KEY UPDATE` 兜底，避免 MySQL collation 认为重复的标签触发 duplicate key。若日志仍出现该表的 `uq_illegal_dataset_label_mapping_raw_label`，优先检查是否存在多实例并发保存或数据库中已手工插入的异常记录。

## 模型转换依赖本地队列文件

模型转换任务通过 `temp/model_conversions` 下的状态文件和锁文件协调。排查卡住任务时检查 stale lock、worker 是否启动、GPU/ONNX provider 是否可用。

## Windows 下轮询状态文件可能碰到短暂文件锁

违规数据集发布任务状态以数据库表 `illegal_dataset_publish_jobs` 为准，同时镜像写入 `temp/illegal_dataset_publish_jobs/<dataset_id>/<job_id>/status.json` 便于排查。后台线程会频繁原子写入镜像，前端会频繁轮询；Windows 下 `replace` / `read_text` 可能短暂互斥。状态文件读写应使用同一 job 级锁并对 `PermissionError`、短暂 `JSONDecodeError` 做小间隔重试，避免轮询接口偶发 500 或进度任务被状态写入异常带崩。

## Worker 状态需要成套维护

训练、推理、部署和转换都有排队、运行、取消、失败、完成等状态。修改某个状态时要同步检查 API 响应、数据库/文件状态、WebSocket 或轮询查询逻辑。
