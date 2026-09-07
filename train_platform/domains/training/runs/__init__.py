"""Training run lifecycle, query, reporting, and execution capabilities."""

from .artifacts import compute_epoch_metric_snapshots, index_completion_artifacts, register_reported_artifact
from .benchmarks import TrainingRunBenchmarkService
from .exports import ExportDownload, TrainingExport, download_export, export_training_run
from .lifecycle import (
    FinalizeResult,
    finalize_execution,
    mark_started,
    queue_run,
    release_stale_claim,
    request_cancel,
    request_delete,
    resume_run,
    touch_heartbeat,
)
from .logs import tail_logs, tail_text_file
from .metadata import get_meta, mark_project_card_reviewed, update_meta
from .progress import upsert_epoch_metrics
from .queries import list_artifacts, list_epoch_metrics, list_events
from .reports import FrameworkCompareConflict, build_report, compare_runs, summarize_metrics
from .service import TrainingRunService

__all__ = [
    "FinalizeResult",
    "FrameworkCompareConflict",
    "ExportDownload",
    "TrainingRunService",
    "TrainingExport",
    "TrainingRunBenchmarkService",
    "build_report",
    "compare_runs",
    "compute_epoch_metric_snapshots",
    "download_export",
    "export_training_run",
    "finalize_execution",
    "get_meta",
    "index_completion_artifacts",
    "register_reported_artifact",
    "list_artifacts",
    "list_epoch_metrics",
    "list_events",
    "mark_project_card_reviewed",
    "mark_started",
    "queue_run",
    "release_stale_claim",
    "request_cancel",
    "request_delete",
    "resume_run",
    "summarize_metrics",
    "tail_logs",
    "tail_text_file",
    "touch_heartbeat",
    "upsert_epoch_metrics",
    "update_meta",
]
