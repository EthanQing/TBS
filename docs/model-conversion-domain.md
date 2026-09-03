# Model conversion domain

`train_platform/domains/model_assets/conversion` owns asynchronous PT/PTH to
ONNX conversion. The HTTP API only adapts the uploaded stream and constructs
the download response. The YOLO worker only polls, claims, invokes the runner,
and releases its claim.

## Job persistence and execution

Conversion jobs remain filesystem-backed under the configured temporary
directory. `conversion/jobs.py` uses `platform.jobs.JobStore` and `JobStatus`
for validated, atomically replaced status documents. It owns request
validation, upload persistence, job reads, artifact resolution, capped logs,
and the conversion-specific worker claim file.

Queue discovery enumerates status paths and reads each job independently. A
missing or malformed status is skipped without preventing later queued jobs
from running. The worker claim remains separate from the `JobStore` status
lock because it represents execution ownership and stale-claim recovery rather
than a short status-file transaction.

`conversion/runner.py` owns Ultralytics export, output discovery, ONNX and
PyTorch performance measurement, and status transitions. Export failures end
the job as `failed`; performance measurement is best-effort and cannot turn a
successfully exported artifact into a failed conversion. Download URLs are
transport concerns and are derived by the API rather than persisted by the
domain.

Ultralytics model loading uses the public compatibility primitive in
`platform/runtime/ultralytics.py`, shared with training and inference without a
private cross-domain import.
