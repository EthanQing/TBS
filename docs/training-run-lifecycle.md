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
lifecycle capabilities. It sets `PYTHONIOENCODING=utf-8` for the training
subprocess and its descendants so redirected stdout/stderr match the UTF-8
log files even on Windows. Opening the parent file handles with UTF-8 alone
does not configure the child Python streams.
The training subprocess owns execution setup, trainer
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
also bound to the current `TrainingRun.pid`. The supervising worker supplies the
spawned process ID. Before setup, the training subprocess waits up to four seconds
for a `RUNNING` claim with a non-null PID, using a fresh session every 75 ms.
It accepts that PID only when it is its own interpreter PID or a verified
ancestor PID (Windows uv/venv launchers). It logs both the actual and guard PIDs
and uses the resolved claim PID for metrics, reported artifacts, heartbeat, and
finalization. Missing, timed-out, or unrelated claims fail closed; unresolved
executions do not finalize the run. The subprocess never rewrites the claim PID.
Stale reconciliation supplies the PID observed on the stale row. A heartbeat,
progress callback, or finalization request whose expected PID no longer matches
the authoritative row is a no-op. This prevents a callback from an older
execution from changing a resumed execution of the same run.
Metric and reported-artifact PID rejections are logged with the run and PID
context; metric persistence failures include an exception traceback.

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
orchestration and native checkpoint handling. The Ultralytics adapter owns
argument construction, callbacks, and invocation; its importable trainer
subclasses apply execution paths after checkpoint argument restoration and
before the framework creates output directories, including in DDP children.
Registry membership remains
a simple static list of the three supported plugins; there is no dynamic discovery
or execution framework.

## Ultralytics execution paths

`domains/training/execution_paths.py` owns filesystem layout resolution shared
by framework execution and the runs domain. `TrainingExecutionSpec.run_dir`
and `TrainingRunResult.results_dir` continue to identify the task root under
`settings.training_dir / run_id`.

New Ultralytics executions place platform configuration in
`runtime/data.runtime.yaml` and `runtime/layout.json`, worker streams in
`logs/train.stdout.log` and `logs/train.stderr.log`, and all framework output
in `output/`. The version 1 manifest records engine `ultralytics-yolo` and the
relative `output_dir`. Execution preparation creates directories and writes
the manifest atomically; path reads do not create directories. Fresh
Ultralytics execution preparation resets only `output/`, including weights, exports,
CSV, and plots. Required model inputs stored inside output are first copied to
`runtime/inputs/<id>/`, and the prepared model/pretrained paths point there.
The reset rejects redirected output directories and preserves runtime and logs.
Ranks never initialize or reset the output directory.

A recorded layout is authoritative for artifact discovery, including weights,
configuration, CSV, and plots. Unrecorded historical tasks use the original
root layout. Historical layouts without execution metadata retain the root-level
`weights/last.pt` resume fallback. New preparation records an `execution` object
with `mode`, `reset_state`, and the explicitly selected `resume_checkpoint`.
A fresh boundary is recorded before output removal; incomplete initialization
blocks reads and resume. Once fresh initialization succeeds, resume uses only
that execution's output checkpoint, returning no checkpoint if none exists.
The adapter resolves the source checkpoint before preparing the current layout.
Resume records that source so it remains usable until the resumed execution
writes its own checkpoint, without an unconditional legacy-directory fallback.
Both same-task and cross-task resume retain framework checkpoint state while applying the
current task's data, project, name, and save directory after framework resume
argument handling. The source checkpoint is not relocated.

When resume switches output directories, the adapter captures the actual
checkpoint's parent output directory and its `epoch` / `train_results` before
preparing the current layout. `prepare_ultralytics_resume_output` receives
these plain values without loading framework objects. It copies source
`weights/best.pt` only when the current output has no best weight, preserving
an existing current best on repeated preparation. Source files remain intact;
checkpoint best fitness, optimizer, and epoch restoration stay framework-owned.

CSV history prefers checkpoint `train_results`, retaining column names and
order. If unavailable, it reads the captured source output's `results.csv`.
Checkpoint epochs are zero-based, while CSV epochs are one-based: carried
records stop at `checkpoint_epoch + 1`. Matching current history is retained
without duplicate epochs; later records are excluded before training appends
the next epoch. Carryover is a no-op when source and target output are the
same. Extra post-training validation explicitly uses the current runtime YAML,
project, name, and save directory even when the best weight originated in a
different task.

Indexed artifact paths remain relative to `settings.training_dir`, for example
`run_id/output/weights/best.pt`, and semantic weight roles update the result
projection. ONNX export writes beside its selected PT source; ONNX downloads
accept indexed exports only within the effective output layout and otherwise
resolve that layout directly. Reports refresh stale weight projections against
the effective layout before benchmark enrichment; failed refreshes cannot reuse
old weight paths.
PaddleDetection retains its native layout, and custom-source retains reported
semantic artifacts. Deletion removes the whole task root.

## Platform-managed Ultralytics multi-GPU execution

Explicit selection of multiple GPUs uses one queue claim and one `RunningJob`:
`DbQueueWorker -> train_entry -> torch.distributed.run -> Rank processes`.
`TrainingRun.pid` remains the existing train_entry execution guard PID.
`TrainingExecutionSpec.execution_owner` carries that PID, its process creation
time, worker ID, and process scope; it contains no database objects. Single-GPU and CPU
executions continue training and extra validation inside train_entry.

`platform/runtime/process_scope.py` owns scope acquisition and comparison.
Linux scope combines `/proc/sys/kernel/random/boot_id` with the device and inode
of `/proc/self/ns/pid`. Worker identity is independent of this process scope.
The Worker records `run_id` and the complete execution owner atomically in
`runtime/execution.json` after spawning the supervisor and before publishing
the RUNNING claim. This identifies preparation-stage executions even before
any DDP attempt exists. train_entry validates the record against its database
claim and actual guard identity; Rank entry validates its actual scope before
process registration. Context, process registrations, and pending cleanup
identities retain the same scope through execution and cleanup handoff.

The adapter prepares the layout, runtime dataset YAML, resume best weight and
CSV carryover once. `PreparedUltralyticsExecution` contains only serializable
paths, model type (`yolo` or `rtdetr`), arguments, settings, and ownership.
The CPU model used during preparation is released before launching Rank
processes. Temporary pretrained weights remain available through the entire
supervised execution. Fresh output initialization happens before Rank startup;
checkpoint state restoration remains owned by the native Trainer. Platform
Trainer subclasses reapply current execution paths and AMP after check_resume.

`platform/runtime/ultralytics_ddp.py` owns per-attempt control files under
`runtime/ddp/<attempt_id>/`: `context.json`, `metrics.jsonl`, and `processes/`.
It launches the dedicated `workers/training/ultralytics_ddp_entry` module using
the current Python executable, a unique c10d rendezvous ID, an automatically
assigned local port, and zero restarts. Stdout and stderr inherit train_entry's
worker log streams. The lightweight entry remains Python source; its
implementation module is included in protected runtime builds.

All Ranks inherit the same final container `CUDA_VISIBLE_DEVICES` mask and bind
their local GPU using `LOCAL_RANK`. Ultralytics receives that frozen mask as
`device`, because select_device writes it back into the environment. Rank code
does not repeat host-to-container mapping. The total batch is passed unchanged;
the framework divides it by world size. Each Rank applies safe loading,
pin-memory and AMP settings, disables built-in MLflow, and uses the existing
Platform Trainer selection. Distributed final_eval stays inside the Trainer;
the parent does not invoke train or val in this branch.

Shared callbacks mark a pending zero-based epoch at on_train_epoch_start and
consume it at on_fit_epoch_end. Metrics combine validation results, labelled
training losses, and `trainer.lr`. final_eval has no pending training epoch,
so it does not replace the last epoch's metrics. Only Rank 0 appends flushed
JSONL events. The supervisor incrementally reads complete lines, checks the
run/attempt identity, retains partial tails, and forwards metrics to the
existing PID-guarded database and MLflow callback. Exit status, rather than
events, CSV, or logs, determines execution success.

Launcher, Rank, and observed descendant registrations contain PID, creation
time, Linux process group, and execution identity. The supervisor also retains
observed process handles in memory so registration failure cannot prevent
cleanup. Cancellation first terminates the launcher with a bounded grace,
then terminates and kills remaining matching processes. Signal handling and
callback errors enter the same cleanup path; tail events are drained before
returning. User cancellation raises `UltralyticsDDPCancelled`; nonzero launcher
exit or an external termination signal raises `UltralyticsDDPError`.
The Worker grants 15 seconds for cooperative multi-GPU cancellation before its
outer fallback. On supervisor exit it checks matching registrations even when
the root process is already gone, and does not finish Worker cleanup while
registered processes remain alive. Old execution identities are excluded.

Cleanup round survivors are candidates, not the final result. After launcher
reaping and registered/observed process cleanup, PID and creation time are
revalidated and exited, zombie, or reused identities are removed. An unresolved
process state raises `UltralyticsDDPCleanupIncomplete`, preserving the original
training error and cleanup errors. `cleanup-pending.json` carries remaining
observed identities across the supervisor-to-Worker handoff. train_entry exits
nonzero and skips lifecycle finalization for this exception, retaining the claim.
The Worker retries cleanup with the same execution identity and keeps its
RunningJob and log handles until cleanup succeeds, then finalizes with the
original exit code and authoritative cancel/delete intent. Stale DDP claims
likewise require a stopped supervisor and confirmed child cleanup before terminal
finalization; an expired heartbeat alone is insufficient.

PID queries and cleanup first require the recorded scope to match the current
environment. Only within that scope can a missing PID, changed creation time,
or zombie status establish that the old process has exited. Another scope or
an unreadable/missing scope remains unconfirmed and raises cleanup-incomplete
at the cleanup boundary. It never means an empty, successfully cleaned process
set. Handoff preserves the original scope of unconfirmed identities.

Stale recovery first matches the startup execution record to the current
database claim, or uses an unambiguous scoped attempt record when the startup
record is absent. No verifiable record means deferred recovery; it does not
fall back to checking the database PID in the local container. Historical
records without scope are never assigned the current scope. A restarted Worker
in the same scope can recover a verified dead execution; a Worker in another
scope logs the reason and leaves the claim intact. No host PID namespace or
cross-node scheduling service is required by this flow.


## Custom-source runtime v1

Custom model manifests currently support only the `pytorch-default` runtime
profile, so the existing `ultralytics-yolo` PyTorch worker also claims
`custom-source` runs without changing either engine identity. The worker gives
custom-source cancellation to the inner runtime first and uses a longer hard
fallback only if the supervising `train_entry` process does not exit. Built-in
engines other than platform-managed Ultralytics multi-GPU retain immediate
outer-worker termination.

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
