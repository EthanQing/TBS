# Changelog

## Unreleased

- Added model evaluation jobs under `/api/v3/model-evaluations` for computing
  Precision, Recall, F1, mAP50, and mAP50-95 on labeled YOLO detection datasets.
- Fixed model evaluation cancellation so cancelling an active job immediately
  releases the active-job guard and late worker updates cannot revive it.
- Switched Ultralytics model evaluation to the native `YOLO.val()` worker path
  and validate labeled images before creating a job.
- Added an inference-worker health check before creating native YOLO evaluation
  jobs, so a stopped worker returns a clear startup error.
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
