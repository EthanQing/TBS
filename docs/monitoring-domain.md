# Monitoring domain

`train_platform/domains/monitoring` owns system metrics and alarms as sibling
capabilities. They share no generic manager, event bus, scheduler, or rule
engine because current alarm rules only evaluate persisted Training Runs.

## Alarms

- `alarms/catalog.py` owns the two fixed rule types and their validation:
  `training_run_failed` and `training_run_stale`.
- `alarms/service.py` owns default seeding, rule administration, alert queries,
  acknowledgement metadata, and the active-alert summary. Acknowledgement does
  not resolve an alert.
- `alarms/training.py` reads the persisted `TrainingRun` model, evaluates the
  two rules with side-effect-free matching functions, and reconciles active
  alerts. The first match creates an alert regardless of cooldown. Cooldown
  only controls subsequent touches and uses `last_triggered_at` as its pivot.
  A missing Training Run does not match, so an existing active alert resolves.

Without explicit run IDs, manual evaluation targets both active training-alert
source IDs and all `RUNNING` or `FAILED` Training Runs. Default rules are seeded
at database startup; evaluation also ensures them so the explicit manual
capability remains usable when invoked independently.

Alarm evaluation is an integration concern. The queue worker invokes the
Monitoring capability after execution start, terminal finalization, and stale
claim reconciliation. Training Run mutation endpoints invoke it after state
changes that can resolve an existing failed or stale alert. The training
subprocess and Training domain do not depend on Monitoring. Integration callers
use one best-effort capability so monitoring failure cannot turn into training
failure.

### Stale timing

The stale rule matches only a `RUNNING` row whose heartbeat, or start time when
no heartbeat exists, is older than the rule threshold. There is no periodic
Monitoring scheduler. A training heartbeat refreshes liveness and does not run
alarm evaluation, because evaluating immediately after a successful heartbeat
cannot create a stale alert.

The worker's stale reconciliation uses the same configured threshold for its
lifecycle decision. It first finalizes a stale `RUNNING` row as `FAILED`, then
evaluates alarms. Consequently, the normal worker path creates the failed alert
without first exposing a stale alert. A stale alert can be observed when manual
evaluation runs after the rule threshold but before worker reconciliation. The
later worker evaluation resolves that stale alert and creates the failed alert
in the same reconciliation pass. A later explicit state change or manual
evaluation can also resolve it when the stale condition no longer matches.

## System metrics

- `metrics/collector.py` samples local CPU and memory and normalizes local GPU
  metrics. GPU probing tries NVML first and falls back to `nvidia-smi` when NVML
  is unavailable or yields no devices.
- `metrics/history.py` owns bounded retention, time-window reads, and
  downsampling over a locked process-local deque per node.
- `metrics/service.py` makes orchestration explicit: `collect_current` only
  collects, `record_snapshot` mutates history, and `sample` does both. Current
  summary and the single-backend-node overview sample and record, preserving
  their existing HTTP behavior. An empty history query samples one point so
  the API remains non-empty.

Metrics history is ephemeral process memory. It is cleared on backend restart
and is neither shared between backend processes nor persisted to the database,
Redis, Prometheus, or another time-series store. The cluster overview remains a
compatibility view of the one local backend node; it is not distributed node
discovery.
