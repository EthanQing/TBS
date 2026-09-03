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
  derivation. Lifecycle finalization invokes it only for a genuine transition
  to `COMPLETED`.

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
progress, and finalization capabilities.

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

Monitoring remains an integration seam. API and worker entrypoints may invoke
the legacy Alarm service after lifecycle operations; the Training domain does
not depend on `services/v3`.
