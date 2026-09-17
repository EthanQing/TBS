# GPU inventory, allocation and execution

`domains/training/resources` owns persisted GPU inventory, Worker observations,
resource-request validation, the shared database allocation ledger and admission
accounting. Scheduling defaults to disabled. Managed nodes use
`scheduler_stage=managed_allocation`; disabled, unmanaged deployments retain
the legacy single-task Worker behavior.

An enabled Worker latches its node into managed mode during registration,
even without queued tasks. Later registrations preserve the database policy.

Explicit deployment policy changes belong to `resources/policy.py`:
`apply_node_policy` validates a complete policy, locks the target node, checks
for any allocation whose state is not `released`, and creates or updates the
policy in the caller's transaction. Even an identical policy is rejected while
that node has unfinished allocations. `docker/apply_gpu_node_policy.py` owns
environment parsing, the Session and transaction, output, and error/exit-code
adaptation, including stdin execution. Applying policy remains an explicit
deployment operation; periodic reports do not overwrite database policy.

## Identity and ownership

- `GpuDevice.gpu_uuid` identifies a whole physical GPU by its complete UUID.
  Multiple Worker instances can observe the same device without creating extra
  capacity. MIG information is recorded on the physical-card observation;
  MIG instances are not additional whole-card resources.
- `GPU_NODE_ID` is the explicitly configured physical node business identifier.
  Workers on the same machine should use the same value. Missing configuration
  is stored as null. A conflicting non-null node does not overwrite the
  device's recorded owner.
- `GpuWorkerInstance.instance_id` identifies one Worker process lifetime.
  Restarting the same `WORKER_ID` creates another instance. Hostname is display
  metadata, and `process_scope` remains exclusively about process visibility
  and cleanup safety.
- `GpuWorkerObservation` stores the latest sample for one instance and UUID.
  Its probe index is not a verified CUDA local index or an execution assignment.
- `TrainingRunResourceRequest.run_id` identifies the user's request. The
  one-to-one relationship is separate from framework parameters.

## Probing and persistence

`platform/runtime/gpu_probe.py` owns NVML-first probing and the `nvidia-smi`
fallback. It has no training-framework or domain dependency. Resource memory
uses integer MiB (1,048,576 bytes): total/free round down and used rounds up.
Unknown fields remain null. Successful empty inventories, unavailable probing,
and probe failures have distinct statuses.

NVML success requires an actual readable observation, not merely a nonempty
diagnostic list. If all enumerated GPU handles fail, probing falls back to
`nvidia-smi`. Mixed results retain readable cards but make the inventory
incomplete. Failed diagnostic entries are excluded from monitoring counts;
partial observations can still be displayed without a UUID, but cannot be
registered as physical resources until a complete UUID is available.

MIG normalization preserves `enabled`, `disabled`, `not_supported`, and
`unknown`. Explicit NVML unsupported results and successful nvidia-smi MIG
fields containing N/A or Not Supported mean `not_supported`; missing fields
and probe failures remain `unknown`. Unknown MIG state does not admit sharing.

`workers/gpu_resource_reporter.py` runs periodic reports independently of task
execution. Each database transaction has its own Session. Inventory services
mutate the caller's transaction; they do not claim or finalize Training Runs.
Defaults are enabled, five-second reports, and twenty-second staleness.

For process attribution, `resources/queries.py` returns a detached ownership
snapshot (including nested execution identity data) in a short transaction.
The reporter closes that transaction before loading execution registrations
and calling the platform process-attribution implementation. Sample persistence
uses a separate transaction; filesystem reads and GPU/process probes stay
outside database transactions.

Heartbeat and last successful inventory timestamps serve different purposes.
A failed probe preserves prior observations and their sampling timestamps.
Only a complete successful inventory marks missing observations absent, and
only for that Worker instance. An offline Worker does not prove that its
training processes have exited.

## Read and request boundaries

`GET /api/v3/gpu-resources`, `GET /api/v3/gpu-workers`, and
`GET /api/v3/training-runs/{run_id}/resources` read persisted data without
probing the API container or changing training lifecycle state. A GPU summary
uses one fresh valid observation; observations from different Workers are
never summed or combined into synthetic memory values. Driver free memory is
sampling data, not schedulable quota.

The API routes only adapt query parameters and serialize response schemas.
`resources/queries.py` owns the complete overview and run-resource payloads,
including scheduler flags, target-node selection, waiting reasons, and the full
resource request. These queries do not probe, initialize policy, or commit.

`resources/requests.py` normalizes requests and validates them jointly with
engine, batch size, and the compatibility device field. New requests keep
`parameters.device=auto`. Shared requests require one GPU, a positive per-GPU
budget, and a fixed positive batch. Multiple GPUs require exclusive mode,
Ultralytics, and a fixed batch divisible by GPU count.

Run creation persists the optional request in the same transaction as the run
and its parameters. Reads load it, and resume retains it. No request is added
to historical runs by migration `0024_gpu_inventory_resources`.

The legacy candidate query still excludes resource requests before its limit.
With scheduling disabled these requests remain queued with `scheduler_disabled`.
On managed nodes, GPU execution requires a database allocation, including
internal conversion of legacy device requests to exclusive requests. Conversion
does not modify the saved user parameters. Unprovable legacy GPU executions
block admission with `legacy_execution_untracked`.

## Managed scheduling

Migration `0025_gpu_managed_scheduling` adds allocation history, per-card
allocation details, node policy, CUDA bindings and execution/wait fields.
`active_run_id` is unique while an allocation is active and becomes null at
release. Allocation history uses restrictive foreign keys to runs, Workers and
devices; deleting a display record cannot erase the ledger.

Node policy is initialized from `GPU_SCHEDULER_ENABLED`,
`GPU_SHARED_EXECUTION_ENABLED`, `GPU_MAX_SHARED_TASKS_PER_DEVICE` (2), and
`GPU_MEMORY_SAFETY_MIB` (4096). The database policy is authoritative after
initialization. Managed mode is persistent; a local disabled scheduler stops
new claims while existing execution and cleanup continue.

`platform.runtime.cuda_probe` runs CUDA Driver API enumeration in a separate process
with the Worker's GPU environment. Binding UUIDs and local ordinals are stored
separately from NVML observation indices, with an environment fingerprint and
verification timestamp. Shared admission requires normal compute mode and a
whole GPU without active MIG. Ultralytics supports shared single-card and
exclusive multi-card execution; Paddle/custom-source use exclusive single-card
execution. Unsupported sharing remains queued.

The probe's thin `-m` entry delegates to `platform/runtime/cuda_probe_impl.py`.
Both are covered by the protected platform tree in pyc builds. Cython keeps
the entry as Python so `-m` can execute it while compiling the implementation.
No Worker probe entry is retained; the platform caller starts the platform
module directly, with the inherited GPU environment and timeout unchanged.

Allocation transactions lock the run, node, Worker and sorted physical UUIDs
before reading active commitments. MySQL Worker allocation transactions use
READ COMMITTED and locking current reads for ledger decisions. Multi-card
requests reserve all devices together. GPU probing and process launch occur
outside this transaction.

Managed candidate selection filters Worker engines, queue eligibility, active
allocations and explicit node affinity before selecting a row. Workers retain
a `(queued_at, created_at, run_id)` cursor across ticks, attempt at most 50
candidates per scheduling pass, and wrap after reaching the end. Each failed
attempt commits its wait reason and releases its locks before the next attempt.
The candidate query does not lock the queue; allocation revalidates the selected
run under its ordinary resource locks. Local engine or node incompatibility
does not establish a global `no_compatible_worker` reason.

`accounting.py` is the shared calculation for admission and resource queries.
For each fresh whole-card sample, total/used/free are T/U/F, safety is S,
budgets are B, and reliably attributed usage is M:

```
E = max(0, U - sum(M))
C = E + sum(max(B, M))
B_new <= min(T - S - C, F - S - sum(max(B - M, 0)))
```

Reserved, starting, running and releasing commitments all participate.
Without `GPU_HOST_PROC_ROOT`, attribution is conservative and M receives no
credit. A configured read-only host proc view must match boot identity, PID
namespace, NSpid, exact start ticks, GPU UUID and allocation owner. Duplicate
process records are not added twice; inconsistent process/whole-card samples
cannot produce usage credits. Budgets are software admission limits, not
hardware memory partitions or hard isolation.
The nvidia-smi fallback records process memory but remains conservative because
its separate GPU and process queries do not provide an atomic snapshot.

## Launch and cleanup ownership

After reservation commits, the Worker issues one-use launch authorization.
`train_entry` first persists the supervisor registration with the prepared
execution owner and ordered GPU UUIDs, then atomically consumes authorization
before CUDA/model initialization. Activation validates that same owner, queue
state, allocation, cancellation intent, scope and assigned UUID order. A failed
base registration leaves authorization unconsumed. Revoked or expired authorization
cannot be consumed by a late child. Each child gets an independent UUID mask;
runtime devices are local 0..N-1. Saved request/device fields stay unchanged.
Ultralytics train/validation and DDP Ranks receive local `torch.device` values
to preserve the UUID mask, while JSON execution contexts stay serializable.

`DbQueueWorker` owns a collection of independent subprocess jobs, capped by
`WORKER_MAX_CONCURRENT_TRAININGS` (2). Each has its own heartbeat, cancellation
timer and asynchronous cleanup. A DDP task consumes one task slot. Conversion
still runs only when no training jobs remain.

`runtime/executions/<allocation_id>/processes` under each run stores supervisor,
Rank, launcher and descendant identities. Shared identity primitives live in
`execution_identity.py`; DDP retains its existing cleanup handoff protocol.
The child reports its result and cleans internal processes; the outer Worker
alone writes the terminal run state and releases the allocation, in one
transaction, after full scope-checked cleanup. Child-only cleanup proof cannot
release the supervisor's allocation. Unknown identities, missing registration
or surviving processes retain the allocation. Heartbeat expiry alone never
releases an activated allocation. Directory deletion happens after release.

Registration failures are recorded separately under each execution's
`registration-errors` directory. Cleanup validates the owner of each failed
identity, retries recoverable registrations, and retains unresolved gaps while
cleaning other verified processes. A legacy `registration-error.json` scan gap
can be resolved only after complete verification of the supervisor's owned
session and the remaining registrations. The watcher uses the original
supervisor PID, creation time and process scope; it does not adopt a reused PID.

Recovery uses the committed database execution owner. A live, matching
supervisor can restore its missing base registration; an exited supervisor
requires an existing matching durable identity with session evidence. A missing
registration alone never proves process exit. Worker retries retain the job and
allocation budget, expose cleanup reasons through resource `reason_details`,
and deduplicate unchanged cleanup events. Final release clears cleanup waiting
state in the same transaction as the run's terminal state.
The Worker reports cleanup facts to `resources/lifecycle.mark_cleanup_pending`,
which locks and rechecks the run and allocation owner before persisting wait
details and deduplicated events. It does not change run status or release the
allocation. The Worker also sends its observed job count to
`resources/inventory.update_worker_running_task_count`, keyed by instance ID;
this is distinct from the active allocation count. Both use caller-owned
Sessions and preserve the Worker's transaction boundaries. TrainingRun state
transitions remain in `domains/training/runs/lifecycle.py`.
DDP cleanup failures still allow the common registry recovery to run, but both
cleanup paths must confirm completion before the allocation can be released.

Metrics, heartbeat and finalization callbacks fence managed executions with
allocation UUID plus PID and creation time. Recovery retains execution owner
identity and only takes over another launcher's work after that launcher is
proven dead in the same process scope.
