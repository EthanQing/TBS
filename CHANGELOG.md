# Changelog

## Unreleased

- Added `lr_scheduler` to training run parameters, supporting `linear` and
  `cosine` learning-rate decay across Ultralytics YOLO and PaddleDetection.
- Switched PaddleDetection integration to a local `release/2.6` source checkout
  and removed the `paddledet` pip package dependency because the wheel does not
  include the required official config tree.
- Fixed the dedicated YOLO worker entrypoint to honor `WORKER_ID`, so multiple
  Docker worker containers can be distinguished in queue claims, events, and
  logs.
