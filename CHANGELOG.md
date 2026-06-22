# Changelog

## Unreleased

- Fixed multi-GPU Docker YOLO worker routing so workers restricted by
  `NVIDIA_VISIBLE_DEVICES` only claim matching explicit GPU tasks and remap host
  GPU ids to container-local CUDA ids before launching training.
- Switched protected Docker runtimes back to `.pyc` protection for core service
  code and selected training worker modules, keeping FastAPI sidecars as Python
  source so route signature inspection works.
- Fixed backend database connection pool exhaustion by making pool sizing
  configurable and releasing DB sessions during long dataset import, publish,
  model evaluation, and deployment runtime work.
- Added a Windows portable launcher workflow for no-Docker/no-WSL customer
  deployments, including local MariaDB orchestration, Alembic migration,
  backend/worker startup, frontend static serving, and self-contained package
  scripts in the outer workspace.
- Hardened the Windows portable customer package path so the launcher can inject
  an embedded license through environment variables and the package can use the
  same protected runtime strategy as the Docker images.
- Improved dataset management list performance by adding a lightweight
  `include_statistics=false` mode for illegal dataset lists and supporting
  front-end tab caching with lazy per-dataset statistics loading.
- Added cancellation for illegal dataset publish jobs so stuck conversions can
  be stopped after backend restarts.
- Sped up illegal dataset publish conversion by processing LabelMe/JSON
  image pairs in parallel with configurable
  `ILLEGAL_DATASET_PUBLISH_MAX_WORKERS`.
- Added detailed dataset import task progress fields and sped up illegal
  mounted LabelMe/JSON imports with parallel JSON parsing plus batched image
  indexing.
- Mounted host `./TBS/imports` to backend container `/app/imports` in the
  compose files so the default offline import directory is visible in Docker.
- Made illegal dataset mounted LabelMe/JSON imports lightweight by recording
  paired source files and raw labels in the version manifest, deferring image
  size reads and YOLO label generation until publish conversion.
- Fixed illegal dataset publish regressions so numeric `version: 1.0`
  LabelMe/JSON annotations use the same bottom-left-origin conversion as
  `version: 1`, and saved parent-label deletes remain part of publish job
  idempotency snapshots.
- Fixed illegal dataset LabelMe/JSON conversion to adapt `version: 1`
  bottom-left-origin points with `y = image_height - y` while leaving newer
  LabelMe versions unchanged.
- Added model evaluation jobs under `/api/v3/model-evaluations` for computing
  Precision, Recall, F1, mAP50, and mAP50-95 on labeled YOLO detection datasets.
- Fixed model evaluation cancellation so cancelling an active job immediately
  releases the active-job guard and late worker updates cannot revive it.
- Switched Ultralytics model evaluation to the native `YOLO.val()` worker path
  and validate labeled images before creating a job.
- Added YOLO worker auto-start for the inference sidecar used by native YOLO
  evaluation, so the usual single `yolo_worker` entrypoint is enough again.
- Fixed intermittent training-run ONNX export failures by treating empty ONNX
  files as invalid and resolving the actual Ultralytics output before download.
- Added optional `include_report` model export packaging, returning a ZIP with
  the exported weights and an on-demand DOCX training report.
- Added `lr_scheduler` to training run parameters, supporting `linear` and
  `cosine` learning-rate decay across Ultralytics YOLO and PaddleDetection.
- Switched PaddleDetection integration to a local `release/2.6` source checkout
  and removed the `paddledet` pip package dependency because the wheel does not
  include the required official config tree.
- Fixed the dedicated YOLO worker entrypoint to honor `WORKER_ID`, so multiple
  Docker worker containers can be distinguished in queue claims, events, and
  logs.
