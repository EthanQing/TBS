from __future__ import annotations

import sys
import time

from train_platform.core.config import settings
from train_platform.workers.model_conversion_queue import ModelConversionQueueWorker
from train_platform.workers.worker import DbQueueWorker


def main() -> None:
    # Dedicated entrypoint for Ultralytics YOLO training jobs and YOLO-side utility jobs.
    training_worker = DbQueueWorker(worker_id="worker-yolo", allowed_engines={"ultralytics-yolo"})
    conversion_worker = ModelConversionQueueWorker(worker_id=training_worker.worker_id)
    engines_text = ",".join(sorted(training_worker.allowed_engines)) if training_worker.allowed_engines else "*"
    print(f"[worker] starting worker_id={training_worker.worker_id} engines={engines_text}", flush=True)
    settings.ensure_dirs()

    while True:
        try:
            training_worker.tick()
            if getattr(training_worker, "_running", None) is None:
                conversion_worker.tick()
        except Exception as e:
            print(f"[worker] tick error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(training_worker.poll_interval)


if __name__ == "__main__":
    main()
