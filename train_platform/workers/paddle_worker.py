from __future__ import annotations

from train_platform.workers.worker import DbQueueWorker


def main() -> None:
    # Dedicated entrypoint for PaddleDetection training jobs.
    DbQueueWorker(worker_id="worker-paddle", allowed_engines={"paddle-det"}).run_forever()


if __name__ == "__main__":
    main()
