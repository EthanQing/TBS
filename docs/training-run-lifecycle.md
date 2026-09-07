# Training Run lifecycle

`train_platform/domains/training/runs` owns the Training Run aggregate lifecycle.
The database `TrainingRun` row is authoritative for user intent, execution
ownership, liveness, progress, and terminal state.

## Responsibilities

- `service.py` owns run creation, read-only get/list queries, name updates, user
  queue/resume/cancel/delete orchestration, and the existing cross-domain
  reference checks required by force deletion.
- `lifecycle.py` is the only implementation of lifecycle field combinations
  and transition events. It owns queueing, resume reset, execution start,
  heartbeat, stale claim release, cancellation/deletion requests, and terminal
  finalization.
- `progress.py` persists epoch metrics and updates epoch/progress only while the
  authoritative run remains `RUNNING`.
- `artifacts.py` owns Training Run artifact/result indexing and metric snapshot
  derivation. It also owns persistence and authoritative path validation for
  artifacts reported by custom trainers. Lifecycle finalization invokes
  completion indexing only for a genuine transition to `COMPLETED`.

## State and intent

User requests and observed execution results are separate:

- Queue moves an eligible run to `QUEUED`.
- Resume moves `FAILED` or `CANCELLED` to `QUEUED`, resetting execution state.
- A cancel request records `cancel_requested_at`. `CREATED` and `QUEUED` runs
  become `CANCELLED` immediately; a `RUNNING` run remains active until process
  termination is observed.
- A delete request records `delete_requested_at`, hides the run, and also
  requests cancellation. Non-running runs become `DELETED` immediately;
  running runs wait for observed termination.

The worker owns candidate selection, device eligibility, subprocess spawning,
termination, and exit observation. It delegates all state changes to the
lifecycle capabilities. The training subprocess owns execution setup, trainer
selection, MLflow/VisualDL integration, and invokes the shared heartbeat,
progress, and finalization capabilities. MLflow is an optional Training-owned
integration under `domains/training/integrations`: the subprocess explicitly
loads and persists its `TrainingRunMeta.extra` binding with its own database
session, while logger initialization, metric writes, and termination remain
best-effort external side effects.

## Finalization and recovery

`finalize_execution` locks and reloads the authoritative row. It only finalizes
a `RUNNING` execution, and applies one terminal priority rule:

1. delete requested -> `DELETED`;
2. cancel requested -> `CANCELLED`;
3. zero exit code -> `COMPLETED`;
4. otherwise -> `FAILED`.

Terminal rows are idempotent no-ops, so the subprocess and queue worker may both
observe completion without duplicating events or artifact indexing. Heartbeat
only updates liveness for an active execution and never revives a terminal run.

Because one run can be resumed into multiple executions, active mutations are
also bound to the current `TrainingRun.pid`. The training subprocess supplies
its own process ID, the supervising worker supplies the spawned process ID, and
stale reconciliation supplies the PID observed on the stale row. A heartbeat,
progress callback, or finalization request whose expected PID no longer matches
the authoritative row is a no-op. This prevents a callback from an older
execution from changing a resumed execution of the same run.

Stale queued claims are released back to `QUEUED`. A stale `RUNNING` row is
finalized as `FAILED` through the same lifecycle owner. Stdout, weights, MLflow,
and result files are not used to infer business state. Normal get/list queries
perform no repair, commits, artifact indexing, or alarm evaluation.

Monitoring remains an integration seam. API and worker entrypoints invoke the
Monitoring domain after lifecycle operations; the training subprocess and
Training domain do not depend on Monitoring.

## Read, report, benchmark, and export capabilities

Training Run application capabilities outside lifecycle also live in the runs
domain:

- `queries.py` owns event, epoch-metric, and artifact reads.
- `metadata.py` owns run metadata updates and project-card review state.
- `logs.py` owns validated stdout/stderr tail reads.
- `reports.py` owns report construction, result/artifact enrichment, framework
  comparison, and the shared metric alias and fallback semantics used by both
  reports and comparisons.
- `benchmarks.py` owns inference-latency measurement, YOLO model statistics,
  FLOPs enrichment, and cached benchmark result updates.
- `exports.py` owns training weight selection, safe export paths, ONNX export
  orchestration, export artifact indexing, download resolution, and optional
  report ZIP packaging.

ONNX export communicates through the existing
`platform.runtime.ModelWorkerClient`; API routes do not call inference worker
HTTP endpoints directly. Report DOCX rendering and MLflow query fallback remain
external seams. Epoch metric reads preserve `mlflow` as MLflow-only and `auto`
as MLflow-first with database fallback. The Training domain does not depend on Alarm/Monitoring, and
none of these capabilities changes Training Run lifecycle semantics.

## Framework execution boundary

`train_platform/domains/training/frameworks` owns the framework execution
contract, static framework registry, and the Ultralytics/PaddleDetection
adapters. `TrainingExecutionSpec` is an immutable, process-memory description
of one execution. It exposes resolved dataset and output paths, architecture
identity, standard training parameters, resume/pretrained intent, requested and
runtime devices, and a filtered framework-specific configuration mapping.
`TrainingCallbacks` exposes only cancellation observation and epoch-metric
recording; framework code does not receive heartbeat or lifecycle capabilities.

`workers/training/train_entry_impl.py` is the sole ORM-to-execution adapter. It
loads the authoritative run and relationships once, resolves dataset/device
state, normalizes the selected plugin configuration, materializes the typed
specification, and wires callbacks to PID-bound run progress plus MLflow. The
framework adapters do not import ORM models, SQLAlchemy sessions, repositories,
or Training Run lifecycle modules.

The PaddleDetection adapter separates three framework-specific capabilities:
YOLO-to-COCO dataset preparation, Paddle configuration transformation, and
runtime compatibility patches. Its plugin module retains the readable training
orchestration and native checkpoint handling. Ultralytics remains a cohesive
single adapter because its compatibility, argument construction, callbacks,
and invocation flow are already readable together. Registry membership remains
a simple static list of the three supported plugins; there is no dynamic discovery
or execution framework.

## Custom-source runtime v1

Custom model manifests currently support only the `pytorch-default` runtime
profile, so the existing `ultralytics-yolo` PyTorch worker also claims
`custom-source` runs without changing either engine identity. The worker gives
custom-source cancellation to the inner runtime first and uses a longer hard
fallback only if the supervising `train_entry` process does not exit. Existing
built-in engines retain immediate outer-worker termination.

The custom-source adapter verifies the immutable package from the
`TrainingRun.custom_model_package_id` and
`TrainingRun.custom_model_source_sha256` execution snapshot, extracts it into
the run workspace, and starts the trusted/internal Python entrypoint in a
separate process group. The child owns no TrainingRun lifecycle or database
persistence. SDK metric and log events use the private
`custom_model/custom_training.events.jsonl` channel; ordinary child stdout and
stderr inherit the normal `train_entry` logs. Cancellation uses a marker file
for cooperative `ctx.should_cancel()` handling, followed by best-effort child
process-tree termination after the inner grace period.

`TrainingContext.report_artifact()` reports a semantic role and a path relative
to `ctx.output_dir` through the same JSONL channel. The parent independently
resolves every reported file beneath the run-local `custom_model/output`
directory and rejects absolute paths, parent traversal, missing or non-file
targets, and symlink escapes before the Training Run domain persists it. The
child SDK, custom entrypoint, runtime, and framework adapter never own artifact
ORM rows.

Artifact `kind` remains the broad storage category, while nullable `role`
records platform meaning. `best_weights` and `last_weights` are singleton roles
whose latest reports update the current row; other valid roles are stored as
generic artifacts without result-projection semantics. Reported artifacts
survive completion indexing, including when a run later fails or is cancelled.
Built-in filename discovery remains a compatibility adapter: known Ultralytics
and Paddle best/last checkpoints receive the same semantic roles. Successful
completion derives `TrainingRunResult.best_weights_path`,
`last_weights_path`, and model size from role-bearing artifact rows without
depending on filename extensions.

Custom model package storage is configured centrally through
`Settings.custom_models_dir` / `BASE_CUSTOM_MODELS_DIR`, defaulting to
`TRAIN_PLATFORM_HOME/custom_models`. Because the backend uploads packages and
the PyTorch worker consumes them, both processes or containers must mount the
same package filesystem or volume.
