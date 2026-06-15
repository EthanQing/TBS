# Changelog

## Unreleased

- Added `lr_scheduler` to training run parameters, supporting `linear` and
  `cosine` learning-rate decay across Ultralytics YOLO and PaddleDetection.
- Fixed the dedicated YOLO worker entrypoint to honor `WORKER_ID`, so multiple
  Docker worker containers can be distinguished in queue claims, events, and
  logs.
