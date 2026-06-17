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
- ZIP 上传任务会先进入 `extracting` 阶段，`safe_extract_zip` 按已解压文件数回调更新任务进度；解压完成后再进入 `validating`、索引和版本更新流程。违规数据集版本创建会在导入事务之外更新任务进度，依次暴露 `validating`、`materializing`、`indexing`、`finalizing`，并通过 `dataset_upload_tasks.processed_count`、`total_count`、`current_item`、`detail_message` 返回处理数量和当前项。违规数据集版本创建不再生成样本预览索引或缩略图，避免大数据集刷新后长时间停在 75%。违规数据集挂载导入使用轻量 manifest：LabelMe/JSON 目录只做图片/JSON 配对、并行读取 JSON 标签、记录挂载源文件和版本索引，不读取图片尺寸、不生成 YOLO labels/data.yaml；YOLO 转换延后到发布标准数据集阶段。并行度由 `DATASET_IMPORT_MAX_WORKERS` 控制，默认 `min(8, cpu_count)`。

修改这条链路时同时检查标准数据集和违规数据集是否需要保持一致。

## 违规数据集发布

- 违规数据集维护原始标签、标签映射、版本和事件。
- 发布逻辑位于 `illegal_dataset_publish_service.py`。
- 发布任务逻辑位于 `illegal_dataset_publish_job_service.py`。
- 前端应通过 `/api/v3/illegal-datasets/{id}/publish-jobs` 创建后台发布任务，并轮询 `/publish-jobs/{job_id}` 展示 `phase`、`progress`、`processed/total`、`logs` 和 `error_message`；重新进入详情页时可调用 `/publish-jobs/active` 恢复最新 queued/running 任务；同步 `/publish` 接口已移除。
- 发布任务以数据库表 `illegal_dataset_publish_jobs` 为状态源，并把状态镜像写到 `temp/illegal_dataset_publish_jobs/<dataset_id>/<job_id>/status.json` 兼容旧轮询排查。幂等键由源违规数据集、源版本、最终生效标签映射、过滤、切片、拆分和 publish_config 生成，排除 `name`/`description`；同一请求重复提交返回已有 queued/running/completed 任务，failed/cancelled 可重置后重试。
- 标签映射 `status=delete` 或 `__DISCARD__` 表示丢弃；发布任务幂等快照和发布转换都会保留删除语义，并按 `label_separator` 把父级删除扩展到 descendants，防止旧映射或缺省映射让子标签原样转出。删除映射变化属于幂等键输入，会生成新的发布任务。
- 发布转换会先按图片/JSON 基名配对，并兼容 `images/`、`json/`、`annotations/` 等顶层目录别名；缺图片、缺 JSON、图片截断或图片不可解码的样本会记录为 skipped/warnings 后跳过，不阻塞还有有效样本的发布。
- LabelMe/JSON 转 YOLO 会按 JSON 顶层 `version` 判断坐标原点：`version` 为 `1`、`1.0` 或 `"1"` 时按左下角原点解析，对每个点执行 `y = image_height - y` 后进入现有左上角坐标流程；其他版本（如 `"5.0.1"`）不转换。该转换只应发生在发布标准数据集阶段。
- 违规数据集版本统一使用 `manifest_path`；挂载导入的图片和 JSON 条目可引用挂载源文件，轻量 manifest 记录原始标签用于映射。历史 `manifest_path` 为空的版本不再支持访问。违规数据集不再对外提供文件列表、原图打开、图片标注查看、样本预览和缩略图；发布转换仍由后端内部读取 manifest 或挂载源文件。
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
训练参数 `lr_scheduler` 默认 `linear`；选择 `cosine` 时，Ultralytics 使用 `cos_lr=true`，PaddleDetection 使用 `CosineDecay` 替换主学习率调度器并保留 warmup。
PaddleDetection 训练插件通过 `utils/paddledet_paths.py` 解析完整 `release/2.6` 源码 checkout，先把 repo root 加入 `sys.path` 再导入 `ppdet`。不要依赖 `paddledet` pip 包；该包不包含平台需要的官方 `configs/` YAML 树。

训练完成后的导出接口是 `POST /api/v3/training-runs/{run_id}/export` 和随后返回的 `/export/download`。默认下载单个 `.pt` 或 `.onnx` 文件；`include_report=true` 时，下载接口会调用 `TrainingRunService().build_report()` 和 DOCX 生成工具即时生成训练报告，并把模型文件与报告打包成 ZIP 返回。

YOLO 专用 worker 会优先使用环境变量 `WORKER_ID`，未设置时回退到 `worker-yolo`。多容器或多 GPU 部署时应为每个 worker 设置任意稳定且唯一的 `WORKER_ID`；代码会原样使用该值，不要求容器名或 ID 遵循特定格式。这样 `TrainingRun.worker_id`、事件和日志才能区分实际领取任务的实例。队列并发以任务为粒度：一个 worker 同时只执行一个训练子进程，多 worker 只会并行领取多个 queued 任务，不会自动拆分单个训练任务。

GPU 绑定由容器运行时和训练参数共同决定。`device=auto` 会继承容器内可见 CUDA 设备；显式 `device=0` 会在训练子进程内设置 `CUDA_VISIBLE_DEVICES=0` 并传给框架本地设备 0。排查多 GPU 部署时，应在每个 worker 容器内检查 `NVIDIA_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`、`nvidia-smi -L` 和 `torch.cuda.device_count()`，确认容器确实只看到预期 GPU。

## 推理任务

- 轻量推理由 API service 调用内部 worker 能力完成。
- 批量或视频推理任务由 `inference_job_service.py` 和 `workers/inference_job_task.py` 管理。
- 推理任务状态、结果和渲染产物通常落在 `BASE_TEMP_DIR` 下。
- 内部 worker HTTP 请求可能需要 `INTERNAL_API_TOKEN`。
- Paddle 推理的配置 YAML 解析与 Paddle 训练共用 `utils/paddledet_paths.py`，默认使用 `PADDLE_DET_DIR` 或后端目录下的 `PaddleDetection/`。

## 模型评估任务

- API 入口为 `/api/v3/model-evaluations`，服务位于 `services/v3/model_evaluation_service.py`。
- 任务状态和结果落在 `BASE_TEMP_DIR/model_evaluations/<job_id>/`，不写数据库表。
- 首版只支持标准 YOLO 检测数据集；`scope=all` 会评估所有有标签图片，`test`/`val`/`train` 按数据集 split 过滤。
- Ultralytics YOLO 评估会先生成仅包含有标签图片的临时 `eval_images.txt` / `eval_data.yaml`，再调用 YOLO inference sidecar 的 `/internal/model-evaluations/yolo-val`，由 `YOLO(...).val()` 计算 Precision、Recall、F1、mAP50、mAP50-95；没有任何有效 YOLO 标签时创建任务阶段直接失败。`train_platform.workers.yolo_worker` 默认会自动拉起本机 `train_platform.workers.inference_worker` sidecar，端口从 `INFERENCE_WORKER_URL` / `INFERENCE_WORKER_PORT` 推导，默认 `18002`；如需禁用自动拉起，可设置 `YOLO_WORKER_START_INFERENCE=0`。Paddle 或非原生路径保留逐图推理回退。
- 取消评估会立即把状态文件置为 `cancelled` 并释放 active job；后台线程后续的进度、结果或异常写入必须检查终态，不能把已取消任务覆盖为 running/completed/failed。

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
