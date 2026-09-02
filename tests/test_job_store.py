from __future__ import annotations

import json
import os

import pytest

from train_platform.platform.jobs import (
    JobNotFoundError,
    JobStatus,
    JobStore,
    MalformedJobStatusError,
    is_terminal_status,
)
from train_platform.platform.jobs.filesystem import _pid_is_alive


def _store(tmp_path) -> JobStore:
    store = JobStore(tmp_path / "jobs", lock_timeout=1, lock_stale_after=1)
    store.create(
        "job-1",
        {
            "job_id": "job-1",
            "status": JobStatus.QUEUED,
            "phase": "preparing",
            "progress": 0,
            "processed": 0,
            "total": 2,
            "seq": 1,
            "last_result_id": 0,
            "cancel_requested": False,
        },
    )
    return store


def test_pid_liveness_probe_is_cross_platform() -> None:
    assert _pid_is_alive(os.getpid())
    assert not _pid_is_alive(-1)
    assert not _pid_is_alive(2**63 - 1)


def test_job_status_preserves_persisted_strings(tmp_path) -> None:
    store = _store(tmp_path)

    assert JobStatus.QUEUED.value == "queued"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.CANCELLED.value == "cancelled"
    assert store.read_status("job-1")["status"] == "queued"
    assert is_terminal_status("cancelled")


def test_terminal_job_cannot_be_reactivated_or_receive_results(tmp_path) -> None:
    store = _store(tmp_path)
    store.update("job-1", {"status": "completed", "phase": "done"})

    unchanged = store.update("job-1", {"status": "running", "progress": 10})
    appended = store.append_result("job-1", {"filename": "ignored.jpg"})

    assert unchanged["status"] == "completed"
    assert "result_id" not in appended
    assert store.read_results_since("job-1") == []


def test_cancellation_policies_are_atomic(tmp_path) -> None:
    store = _store(tmp_path)

    inference = store.cancel(
        "job-1",
        terminal_if=(JobStatus.QUEUED,),
        terminal_patch={"phase": "cancelled"},
    )
    assert inference["status"] == "cancelled"
    assert inference["cancel_requested"] is True

    store.create(
        "job-2",
        {"job_id": "job-2", "status": "running", "seq": 1, "cancel_requested": False},
    )
    evaluation = store.cancel(
        "job-2",
        terminal_if=(JobStatus.QUEUED, JobStatus.RUNNING),
        terminal_patch={"phase": "cancelled", "error_message": None},
    )
    assert evaluation["status"] == "cancelled"
    assert evaluation["phase"] == "cancelled"


def test_result_cursor_is_strictly_after_id(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_result("job-1", {"filename": "a.jpg"})
    store.append_result("job-1", {"filename": "b.jpg"})

    assert [row["result_id"] for row in store.read_results_since("job-1", after_result_id=1)] == [2]
    assert store.read_status("job-1")["last_result_id"] == 2


def test_missing_and_malformed_status_are_explicit(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")

    with pytest.raises(JobNotFoundError):
        store.read_status("missing")

    status_path = store.job_dir("broken", create=True) / "status.json"
    status_path.write_text(json.dumps({"status": "not-a-status"}), encoding="utf-8")
    with pytest.raises(MalformedJobStatusError):
        store.read_status("broken")
