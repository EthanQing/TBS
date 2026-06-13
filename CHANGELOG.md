# Changelog

## Unreleased

- Fixed the dedicated YOLO worker entrypoint to honor `WORKER_ID`, so multiple
  Docker worker containers can be distinguished in queue claims, events, and
  logs.
