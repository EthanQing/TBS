# Inference and model worker runtime

## Ownership

- `train_platform/platform/runtime/model_workers.py` owns HTTP communication with model inference workers. It selects the worker endpoint, adds internal authentication, applies timeouts, parses JSON, and reports transport or worker failures as `ModelWorkerError`.
- `train_platform/domains/inference/input.py` owns inference upload storage, temp-token resolution, remote image materialization, download limits, and remote URL validation. It accepts Python filenames and binary streams rather than FastAPI types.
- `train_platform/domains/inference/service.py` owns synchronous inference orchestration and `InferenceRun` persistence. Batch and video inference are application capabilities of the same domain.
- `train_platform/domains/inference/jobs.py` owns inference-job admission, filesystem job state, worker dispatch requests, cancellation, and result reads. Generic filesystem status and result operations remain in `train_platform/platform/jobs`.

## Dependency direction

Inference resolves a `ModelRuntimeSpec` through `domains/model_assets`, then passes only infrastructure primitives such as engine and artifact paths to `ModelWorkerClient`. The runtime client does not import domain modules.

Model evaluation, deployment smoke tests, and training benchmarks call `ModelWorkerClient` directly. They share worker communication infrastructure without depending on the inference business domain. Serving endpoints use the inference domain because they expose an actual synchronous inference capability.

## Failure boundary

`domains/model_assets/runtime.py` owns runtime capability admission. Only
`ultralytics-yolo` and `paddle-det` are supported. Resolution requires an existing
ModelVersion, a weights file, its source TrainingRun, and its Architecture with
an explicitly supported engine. Missing or unsupported engines raise
`ConflictError` (HTTP 409); Paddle still requires a valid config file.
`custom-source` remains a training foundation and may have registered
ModelVersions, but `runtime_profile=pytorch-default` grants no runtime capability.

Inference and evaluation candidates exclude missing or unsupported engines.
Training benchmarks validate the engine before using cached runtime measurements
or dispatching work; best-effort YOLO statistics only run for explicit YOLO engines.
The worker client independently rejects empty and unsupported engines before HTTP
dispatch and never selects YOLO as a default.

`ModelWorkerClient` raises explicit errors for request failures, non-JSON responses, non-success HTTP responses, worker-reported errors, and missing required output. Business owners decide how those failures affect their state:

- synchronous persisted inference records worker execution failures in `InferenceRun.error_message`;
- inference jobs mark dispatch failures as failed job state;
- deployment smoke tests fail their deployment step;
- training report statistics remain explicitly best effort;
- evaluation failures propagate to evaluation job failure handling.

Model conversion, training process execution, deployment state transitions, and worker process internals are outside this runtime boundary.
