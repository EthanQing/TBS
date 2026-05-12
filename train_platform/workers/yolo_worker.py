from __future__ import annotations

from train_platform.workers.worker import DbQueueWorker


def main() -> None:
    # Dedicated entrypoint for Ultralytics YOLO training jobs.
    DbQueueWorker(worker_id="worker-yolo", allowed_engines={"ultralytics-yolo"}).run_forever()


if __name__ == "__main__":
    main()
