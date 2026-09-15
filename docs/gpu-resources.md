# GPU inventory and training resource requests

`domains/training/resources` owns persisted GPU inventory, Worker observations,
and resource-request validation. This stage is `inventory_only`:
`allocation_enabled=false`, and no reservation or allocation records exist.

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

`workers/gpu_resource_reporter.py` runs periodic reports independently of task
execution. Each database transaction has its own Session. Inventory services
mutate the caller's transaction; they do not claim or finalize Training Runs.
Defaults are enabled, five-second reports, and twenty-second staleness.

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

`resources/requests.py` normalizes requests and validates them jointly with
engine, batch size, and the compatibility device field. New requests keep
`parameters.device=auto`. Shared requests require one GPU, a positive per-GPU
budget, and a fixed positive batch. Multiple GPUs require exclusive mode,
Ultralytics, and a fixed batch divisible by GPU count.

Run creation persists the optional request in the same transaction as the run
and its parameters. Reads load it, and resume retains it. No request is added
to historical runs by migration `0024_gpu_inventory_resources`.

The existing Worker candidate query excludes runs with resource requests
before its 50-candidate limit. These runs can queue but report
`resource_scheduler_not_enabled`; they cannot enter the legacy direct-launch
path. Requests without resource data retain legacy device handling. Request
UUIDs and budgets never become Ultralytics/Paddle training arguments.
