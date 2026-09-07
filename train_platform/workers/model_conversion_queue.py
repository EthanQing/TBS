from __future__ import annotations

import os
from typing import Optional

from train_platform.core.license import assert_valid_license
from train_platform.domains.model_assets.conversion.jobs import (
    claim_job,
    enumerate_queued_jobs,
    mark_claimed,
    release_claim,
)
from train_platform.domains.model_assets.conversion.runner import record_failure, run_job


class ModelConversionQueueWorker:
    def __init__(self, *, worker_id: str, stale_lock_seconds: Optional[int] = None) -> None:
        self.worker_id = str(worker_id)
        self.stale_lock_seconds = int(
            stale_lock_seconds
            if stale_lock_seconds is not None
            else os.getenv("MODEL_CONVERSION_STALE_LOCK_SECONDS", "1800")
        )

    def tick(self) -> bool:
        assert_valid_license()
        for job_id, _data in enumerate_queued_jobs():
            if not claim_job(job_id, self.worker_id, stale_seconds=self.stale_lock_seconds):
                continue
            try:
                if not mark_claimed(job_id, self.worker_id):
                    return False
                run_job(job_id)
                return True
            except Exception as exc:
                record_failure(job_id, exc)
                return True
            finally:
                release_claim(job_id)
        return False
