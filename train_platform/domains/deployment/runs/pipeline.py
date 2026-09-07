from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from PIL import Image

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.domains.deployment import activation
from train_platform.domains.deployment.adapters import DeploymentAdapterContext, get_deployment_adapter
from train_platform.domains.deployment.runs import lifecycle
from train_platform.domains.model_assets.runtime import ModelRuntimeSpec, resolve_model_runtime
from train_platform.models.v3.deployment import Deployment
from train_platform.models.v3.deployment_run import DeploymentRun
from train_platform.platform.runtime import ModelWorkerClient
from train_platform.utils.exceptions import NotFoundError


def _defaults(snapshot: dict[str, Any]) -> dict[str, float]:
    raw = snapshot.get("defaults") if isinstance(snapshot.get("defaults"), dict) else {}
    return {
        "conf": float(raw.get("conf", 0.25)),
        "iou": float(raw.get("iou", 0.45)),
    }


def _run_context(run_id: str) -> dict[str, Any] | None:
    with session_scope() as db:
        run = lifecycle.get_locked_run(db, run_id)
        if run.status.value not in {"queued", "running"}:
            return None
        deployment = (
            db.query(Deployment)
            .filter(Deployment.deployment_id == int(run.deployment_id))
            .first()
        )
        if not deployment:
            raise NotFoundError("Deployment not found")
        snapshot = deepcopy(run.snapshot) if isinstance(run.snapshot, dict) else {}
        snapshot["platform"] = (
            deployment.platform.value if hasattr(deployment.platform, "value") else str(deployment.platform)
        )
        lifecycle.update_execution_metadata(db, run_id, platform=snapshot["platform"])
        return {
            "run_id": str(run.run_id),
            "deployment_id": int(run.deployment_id),
            "project_id": int(run.project_id),
            "model_version_id": int(run.model_version_id),
            "platform": str(snapshot["platform"]),
            "defaults": _defaults(snapshot),
        }


def _validate_artifacts(run_id: str) -> ModelRuntimeSpec | None:
    with session_scope() as db:
        run = lifecycle.get_locked_run(db, run_id)
        model_version_id = int(run.model_version_id)
        if lifecycle.begin_step(db, run_id, key="validate_artifacts", progress=10) is None:
            return None

    with session_scope() as db:
        model = resolve_model_runtime(db, model_version_id=model_version_id)

    with session_scope() as db:
        lifecycle.complete_step(
            db,
            run_id,
            key="validate_artifacts",
            progress=25,
            message="Artifacts validated",
            detail={"engine": model.engine, "weights_path": str(model.weights_path)},
        )
    return model


def _adapter_context(context: dict[str, Any], model: ModelRuntimeSpec) -> DeploymentAdapterContext:
    defaults = context["defaults"]
    return DeploymentAdapterContext(
        deployment_id=int(context["deployment_id"]),
        run_id=str(context["run_id"]),
        model=model,
        conf=float(defaults["conf"]),
        iou=float(defaults["iou"]),
    )


def _materialize_runtime(run_id: str, *, adapter_output: dict[str, Any]) -> bool:
    with session_scope() as db:
        if lifecycle.begin_step(db, run_id, key="materialize_runtime", progress=35) is None:
            return False
        lifecycle.update_execution_metadata(db, run_id, materialized=deepcopy(adapter_output))
        lifecycle.complete_step(
            db,
            run_id,
            key="materialize_runtime",
            progress=55,
            message="Runtime materialized",
            detail={
                "endpoint_url": adapter_output.get("endpoint_url"),
                "health_check_url": adapter_output.get("health_check_url"),
            },
        )
    return True


def _smoke_test(run_id: str, model: ModelRuntimeSpec, *, defaults: dict[str, float], worker: ModelWorkerClient) -> bool:
    with session_scope() as db:
        if lifecycle.begin_step(db, run_id, key="smoke_test", progress=65) is None:
            return False

    smoke_dir = (settings.temp_dir / "deployment_smoke").resolve()
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_image = smoke_dir / f"{run_id}.jpg"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(smoke_image)

    started = time.perf_counter()
    try:
        output = worker.execute_model(
            engine=model.engine,
            weights_path=model.weights_path,
            image_path=smoke_image,
            conf=float(defaults["conf"]),
            iou=float(defaults["iou"]),
            config_path=model.config_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Smoke test failed: {type(exc).__name__}: {exc}") from exc
    inference_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
    output = output if isinstance(output, dict) else {}
    predictions = output.get("predictions")
    detections = len(predictions) if isinstance(predictions, list) else 0

    with session_scope() as db:
        lifecycle.complete_step(
            db,
            run_id,
            key="smoke_test",
            progress=80,
            message="Smoke test passed",
            detail={"detections": detections, "inference_time_ms": inference_time_ms},
        )
    return True


def _apply_materialized(deployment: Deployment, *, materialized: dict[str, Any], run: DeploymentRun, defaults: dict[str, float]) -> None:
    endpoint = str(materialized.get("endpoint_url") or "").strip()
    health = str(materialized.get("health_check_url") or "").strip()
    if endpoint:
        deployment.endpoint_url = endpoint
    if health:
        deployment.health_check_url = health

    config = deepcopy(deployment.config) if isinstance(deployment.config, dict) else {}
    serving_defaults = deepcopy(config.get("serving_defaults")) if isinstance(config.get("serving_defaults"), dict) else {}
    serving_defaults["conf"] = float(defaults["conf"])
    serving_defaults["iou"] = float(defaults["iou"])
    config["serving_defaults"] = serving_defaults
    config["last_deployment_run_id"] = str(run.run_id)
    config["last_materialized_at"] = lifecycle.utcnow().isoformat()
    deployment.config = config

    snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
    pending_hash = str(snapshot.get("pending_api_key_hash") or "").strip()
    if pending_hash:
        deployment.api_key_hash = pending_hash
        deployment.api_key_hint = str(snapshot.get("api_key_hint") or "") or None


def _activate(
    run_id: str,
    *,
    adapter_output: dict[str, Any],
    defaults: dict[str, float],
) -> bool:
    with session_scope() as db:
        run = lifecycle.begin_activation(db, run_id)
        if run is None:
            return False
        deployment_id = int(run.deployment_id)
        model_version_id = int(run.model_version_id)
        snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
        materialized = snapshot.get("materialized") if isinstance(snapshot.get("materialized"), dict) else {}
        merged = dict(materialized)
        merged.update({key: value for key, value in adapter_output.items() if value is not None})
        activation_result = activation.activate_deployment(
            db,
            deployment_id=deployment_id,
            model_version_id=model_version_id,
        )
        deployment = activation_result.deployment
        _apply_materialized(deployment, materialized=merged, run=run, defaults=defaults)
        lifecycle.complete_step(
            db,
            run_id,
            key="activate",
            progress=100,
            message="Deployment activated",
            detail={
                "endpoint_url": deployment.endpoint_url,
                "health_check_url": deployment.health_check_url,
                "api_key_hint": deployment.api_key_hint,
            },
        )
        lifecycle.mark_completed(db, run_id)
        return True


def _ensure_not_cancelled(run_id: str) -> bool:
    with session_scope() as db:
        return lifecycle.cancel_if_requested(db, run_id)


def execute_pipeline(run_id: str, *, worker: ModelWorkerClient | None = None) -> None:
    try:
        with session_scope() as db:
            if lifecycle.mark_running(db, str(run_id)) is None:
                return
        context = _run_context(str(run_id))
        if context is None:
            return
        worker_client = worker or ModelWorkerClient()
        model = _validate_artifacts(str(run_id))
        if model is None:
            return
        if _ensure_not_cancelled(str(run_id)):
            return
        with session_scope() as db:
            lifecycle.update_execution_metadata(db, run_id, model_context=model.to_payload())

        adapter = get_deployment_adapter(context["platform"])
        prepared = adapter.prepare(_adapter_context(context, model))
        if not _materialize_runtime(str(run_id), adapter_output=prepared):
            return
        if _ensure_not_cancelled(str(run_id)):
            return

        if not _smoke_test(str(run_id), model, defaults=context["defaults"], worker=worker_client):
            return
        if _ensure_not_cancelled(str(run_id)):
            return

        activated = adapter.activate(_adapter_context(context, model))
        _activate(str(run_id), adapter_output=activated, defaults=context["defaults"])
    except Exception as exc:
        with session_scope() as db:
            lifecycle.mark_failed(db, run_id, error=f"{type(exc).__name__}: {exc}")


__all__ = ["execute_pipeline"]
