from __future__ import annotations

import os

from train_platform.workers.worker import DbQueueWorker


def main() -> None:
    # Dedicated entrypoint for Ultralytics YOLO training jobs.
    os.environ.setdefault("WORKER_ID", "worker-yolo")
    os.environ.setdefault("WORKER_ENGINES", "ultralytics-yolo")
    DbQueueWorker().run_forever()


if __name__ == "__main__":
    main()
