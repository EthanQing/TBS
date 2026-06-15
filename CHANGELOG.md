# Changelog

## Unreleased

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
