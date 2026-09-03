# Deployment domain and run pipeline

`train_platform/domains/deployment` owns Deployment configuration, serving
readiness, activation, credentials, rollback, and Deployment Run execution.
The database `DeploymentRun` row is authoritative for run lifecycle state.

## Deployment ownership

- `service.py` owns Deployment list/get/create/update/delete operations,
  deployment logs, rollback candidate/history queries, rollback orchestration,
  and serving-readiness lookup.
- `activation.py` is the only Deployment activation implementation. It locks
  the project before mutating deployments, deactivates active peers, switches
  the selected Deployment to the target model, marks it active, and synchronizes
  the target ModelVersion to `PRODUCTION` while demoting the previous project
  production version to `TESTING`.
- Normal pipeline completion, rollback, and lifecycle fields accepted by the
  Deployment PATCH endpoint all use the same activation capability.
- `credentials.py` owns API-key generation, hashing, and verification. Only the
  hash and display hint are persisted on Deployment; a raw key is returned only
  when issued.
- `adapters.py` contains the static local-gateway platform adapter. Its typed
  context contains scalar execution identity, `ModelRuntimeSpec`, and inference
  defaults; adapters do not receive ORM rows or database sessions.

Rollback validates a historically successful model candidate, then performs
the model switch, activation invariants, stage synchronization, and rollback
audit write in one database transaction. The activation result records the
authoritative previous model ID for that audit.

## Deployment Run ownership

- `runs/service.py` owns run creation, read operations, retry, cancellation,
  per-project active-run admission, process-local dispatch, and startup
  recovery. Retry always creates a new run row and execution snapshot.
- `runs/lifecycle.py` is the only implementation of run status, phase, step,
  progress, cancellation, timing, and error-field combinations. Run-log
  sequence allocation occurs while the same run row is locked.
- `runs/pipeline.py` is a fixed, top-down sequence: resolve model artifacts,
  prepare and stage local runtime metadata, run smoke inference, invoke adapter
  activation, apply shared Deployment activation, and complete the run.
- Model artifact resolution remains owned by
  `domains.model_assets.runtime.resolve_model_runtime`; smoke inference remains
  owned by `platform.runtime.ModelWorkerClient`.

Endpoint, serving defaults, runtime metadata, and a pending API-key hash remain
execution metadata until smoke inference succeeds. The final transaction
applies those values to Deployment together with activation and run completion,
so failure or cancellation does not rotate a live Deployment key or partially
switch its model stage.

## Concurrency and recovery

The project row is the serialization lock for active-run admission and
activation. A project can have at most one database run in `QUEUED` or
`RUNNING`, and activation uses the same project-first lock order before locking
Deployment and ModelVersion rows. The existing schema is sufficient; no job,
attempt, or event table is used.

`_RUN_THREADS` records only daemon threads currently executing in this process.
It is not consulted to determine run status. At backend startup, orphaned
`QUEUED` runs are dispatched again. Orphaned `RUNNING` runs are finalized as
`FAILED` with a process-restart interruption reason because the fixed pipeline
and external inference call have no safe resume checkpoint. Cancellation of a
queued run is immediate; a running run observes cancellation at step boundaries
and again atomically before final activation.

The backend container currently starts one Uvicorn worker. The recovery and
thread dispatcher are therefore process-local execution mechanisms over the
shared database truth, not a distributed queue.
