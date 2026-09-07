from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.standard_dataset import StandardDataset
from train_platform.models.v3.training_run import (
    TrainingRun,
    TrainingRunEpochMetric,
    TrainingRunResult,
)
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

from .artifacts import compute_epoch_metric_snapshots, index_completion_artifacts
from .benchmarks import TrainingRunBenchmarkService
from .service import TrainingRunService


ENGINE_FRAMEWORK_MAP: dict[str, tuple[str, str]] = {
    "ultralytics-yolo": ("pytorch", "PyTorch"),
    "paddle-det": ("paddle", "Paddle"),
}

# One metric vocabulary is shared by report core metrics and compare summaries.
METRIC_SEMANTICS: dict[str, tuple[str, ...]] = {
    "metrics/mAP50-95(B)": (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
        "mAP50-95",
        "mAP",
        "map",
        "bbox_map",
        "bbox_mAP",
        "eval/bbox_mAP",
        "eval/bbox_map",
    ),
    "metrics/mAP50(B)": (
        "metrics/mAP50(B)",
        "metrics/mAP50(M)",
        "mAP50",
        "AP50",
        "ap50",
        "bbox_ap50",
        "bbox_AP50",
        "eval/bbox_AP50",
        "eval/bbox_ap50",
    ),
    "metrics/mAP75(B)": (
        "metrics/mAP75(B)",
        "metrics/mAP75(M)",
        "mAP75",
        "AP75",
        "ap75",
        "bbox_ap75",
        "bbox_AP75",
        "eval/bbox_AP75",
        "eval/bbox_ap75",
    ),
    "metrics/precision(B)": (
        "metrics/precision(B)",
        "metrics/precision(M)",
        "precision",
        "Precision",
        "bbox_precision",
        "eval/bbox_precision",
    ),
    "metrics/recall(B)": (
        "metrics/recall(B)",
        "metrics/recall(M)",
        "recall",
        "Recall",
        "bbox_recall",
        "eval/bbox_recall",
    ),
}
COMPARE_METRIC_KEYS: tuple[str, ...] = (
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
)
REPORT_CORE_METRICS: dict[str, str] = {
    "mAP50-95": "metrics/mAP50-95(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
}


class FrameworkCompareConflict(ConflictError):
    def __init__(self, message: str, framework_groups: Dict[str, List[str]]) -> None:
        super().__init__(message)
        self.framework_groups = framework_groups


def _ensure_aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number == number else None


def _pick_metric_value(metrics: Dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    if not isinstance(metrics, dict):
        return None
    lowered = {str(key).lower(): key for key in metrics.keys()}
    for key in candidates:
        if key in metrics:
            number = _as_number(metrics.get(key))
            if number is not None:
                return number
        actual = lowered.get(str(key).lower())
        if actual is not None:
            number = _as_number(metrics.get(actual))
            if number is not None:
                return number
    return None


def summarize_metrics(
    db: Session,
    run: TrainingRun,
    *,
    keys: tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """Normalize result/epoch metrics into the shared canonical metric vocabulary."""

    metric_keys = keys or tuple(METRIC_SEMANTICS)
    best: Dict[str, float] = {}
    final: Dict[str, float] = {}
    used_result = False
    used_epoch = False

    result_best = run.result.best_metrics if run.result and isinstance(run.result.best_metrics, dict) else {}
    result_final = run.result.final_metrics if run.result and isinstance(run.result.final_metrics, dict) else {}
    for canonical in metric_keys:
        candidates = METRIC_SEMANTICS[canonical]
        best_value = _pick_metric_value(result_best, candidates)
        final_value = _pick_metric_value(result_final, candidates)
        if best_value is not None:
            best[canonical] = best_value
            used_result = True
        if final_value is not None:
            final[canonical] = final_value
            used_result = True

    missing = [key for key in metric_keys if key not in best or key not in final]
    if missing:
        rows = (
            db.query(TrainingRunEpochMetric)
            .filter(TrainingRunEpochMetric.run_id == str(run.run_id))
            .order_by(TrainingRunEpochMetric.epoch.asc())
            .all()
        )
        epoch_best: Dict[str, float] = {}
        epoch_final: Dict[str, float] = {}
        for row in rows:
            metrics = row.metrics if isinstance(row.metrics, dict) else {}
            for canonical in missing:
                number = _pick_metric_value(metrics, METRIC_SEMANTICS[canonical])
                if number is None:
                    continue
                epoch_final[canonical] = number
                previous = epoch_best.get(canonical)
                if previous is None or number > previous:
                    epoch_best[canonical] = number
        if epoch_best or epoch_final:
            used_epoch = True
        for canonical in missing:
            if canonical not in best and canonical in epoch_best:
                best[canonical] = epoch_best[canonical]
            if canonical not in final and canonical in epoch_final:
                final[canonical] = epoch_final[canonical]

    source = None
    if used_result and used_epoch:
        source = "mixed"
    elif used_epoch:
        source = "epoch_fallback"
    elif used_result:
        source = "result"
    return {"best": best, "final": final, "source": source}


def _compare_metric_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "best": {key: summary.get("best", {}).get(key) for key in COMPARE_METRIC_KEYS if key in summary.get("best", {})},
        "final": {key: summary.get("final", {}).get(key) for key in COMPARE_METRIC_KEYS if key in summary.get("final", {})},
        "source": summary.get("source"),
    }


def _extract_core_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    best = summary.get("best") if isinstance(summary, dict) else {}
    final = summary.get("final") if isinstance(summary, dict) else {}
    out: Dict[str, float] = {}
    for label, canonical in REPORT_CORE_METRICS.items():
        value = best.get(canonical) if isinstance(best, dict) else None
        if value is None and isinstance(final, dict):
            value = final.get(canonical)
        if value is not None:
            out[label] = value
    return out


def _resolve_framework(engine: str | None) -> tuple[str, str]:
    raw = str(engine or "").strip().lower()
    if not raw:
        return "engine:unknown", "Engine: unknown"
    mapped = ENGINE_FRAMEWORK_MAP.get(raw)
    if mapped:
        return mapped
    return f"engine:{raw}", f"Engine: {raw}"


def _duration_seconds(started_at: Optional[datetime], finished_at: Optional[datetime]) -> float | None:
    start = _ensure_aware_utc(started_at)
    end = _ensure_aware_utc(finished_at)
    if start is None or end is None:
        return None
    try:
        return max(0.0, round((end - start).total_seconds(), 3))
    except Exception:
        return None


def _save_period_from_params(additional_params: Optional[Dict[str, Any]]) -> int | None:
    add = additional_params if isinstance(additional_params, dict) else {}
    for key in ("save_period", "snapshot_epoch"):
        raw = add.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def _run_image_size(run: TrainingRun) -> int:
    params = getattr(run, "parameters", None)
    raw = getattr(params, "image_size", None) if params is not None else None
    try:
        value = int(float(raw))
    except Exception:
        return 640
    return value if value > 0 else 640


def _ensure_report_metric_snapshots(db: Session, run: TrainingRun) -> TrainingRunResult | None:
    result = run.result
    best_present = bool(result and isinstance(result.best_metrics, dict) and result.best_metrics)
    final_present = bool(result and isinstance(result.final_metrics, dict) and result.final_metrics)
    if best_present and final_present:
        return result

    best_metrics, final_metrics = compute_epoch_metric_snapshots(db, str(run.run_id))
    if best_metrics is None or final_metrics is None:
        return result

    if result is None:
        result = TrainingRunResult(run_id=str(run.run_id))
        db.add(result)
        db.flush()
        run.result = result
    result.best_metrics = best_metrics
    result.final_metrics = final_metrics
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def _ensure_report_artifacts(db: Session, run: TrainingRun) -> TrainingRunResult | None:
    result = run.result
    needs_index = result is None
    if result is not None:
        needs_index = any(
            value is None
            for value in (result.best_weights_path, result.last_weights_path, result.model_size_mb)
        )
    if needs_index:
        try:
            index_completion_artifacts(db, str(run.run_id))
            db.commit()
            db.refresh(run)
            result = run.result
        except Exception:
            db.rollback()
            result = run.result

    benchmark = TrainingRunBenchmarkService()
    needs_flops = result is not None and (_as_number(result.flops) in (None, 0))
    needs_latency = result is not None and (_as_number(result.inference_time_ms) in (None, 0))
    if result is not None and (needs_flops or needs_latency):
        try:
            stats = benchmark.measure_yolo_stats(run, result)
            changed = False
            flops = _as_number(stats.get("flops"))
            if needs_flops and flops and flops > 0:
                result.flops = int(flops)
                changed = True
            latency = _as_number(stats.get("inference_time_ms"))
            if needs_latency and latency and latency > 0:
                result.inference_time_ms = latency
                changed = True
            if changed:
                db.add(result)
                db.commit()
                db.refresh(result)
        except Exception:
            db.rollback()

    engine = str(getattr(run.architecture, "engine", "") or "ultralytics-yolo").strip().lower()
    if result is not None and engine != "ultralytics-yolo" and (_as_number(result.inference_time_ms) in (None, 0)):
        try:
            measured = benchmark.measure_inference_latency(
                db,
                run=run,
                benchmark_image=benchmark.ensure_benchmark_image(),
                conf=0.25,
                iou=0.45,
                warmup=1,
                iters=5,
            )
            result.inference_time_ms = measured
            db.add(result)
            db.commit()
            db.refresh(result)
        except Exception:
            db.rollback()
    return result


def build_report(db: Session, run_id: str) -> Dict[str, Any]:
    run = TrainingRunService().get_run(db, run_id)
    if run.status != TrainingRunStatus.COMPLETED:
        raise ValidationError("训练尚未完成，报告不可用")

    result = _ensure_report_metric_snapshots(db, run)
    if result is None:
        result = run.result
    result = _ensure_report_artifacts(db, run) or result

    arch = run.architecture
    if arch is None:
        arch = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id == int(run.architecture_id)).first()
    if arch is None:
        raise NotFoundError("Architecture not found")

    dataset = run.standard_dataset
    if dataset is None:
        dataset = (
            db.query(StandardDataset)
            .filter(StandardDataset.standard_dataset_id == int(run.standard_dataset_id))
            .first()
        )
    params = run.parameters
    if params is None:
        raise NotFoundError("Training run parameters not found")

    engine = str(getattr(arch, "engine", "") or "").strip().lower()
    framework_key, framework_label = _resolve_framework(engine)
    best_metrics = result.best_metrics if result and isinstance(result.best_metrics, dict) else None
    final_metrics = result.final_metrics if result and isinstance(result.final_metrics, dict) else None
    summary = summarize_metrics(db, run)
    core_metrics = _extract_core_metrics(summary)

    model_size_mb = _as_number(getattr(result, "model_size_mb", None) if result else None)
    inference_time_ms = _as_number(getattr(result, "inference_time_ms", None) if result else None)
    if inference_time_ms is not None and inference_time_ms <= 0:
        inference_time_ms = None
    flops = _as_number(getattr(result, "flops", None) if result else None)
    flops_out = int(flops) if flops is not None and flops > 0 else None

    return {
        "basic": {
            "run_id": str(run.run_id),
            "name": run.name,
            "framework_label": framework_label,
            "framework_key": framework_key,
            "engine": engine,
            "status": str(getattr(run.status, "value", run.status) or ""),
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration_seconds": _duration_seconds(run.started_at, run.finished_at),
        },
        "dataset": {
            "dataset_id": int(dataset.standard_dataset_id) if dataset is not None else None,
            "dataset_name": str(dataset.name) if dataset is not None else None,
            "dataset_version": None,
        },
        "architecture": {
            "architecture_id": int(arch.architecture_id),
            "family": str(getattr(arch, "family", "") or ""),
            "variant": str(getattr(arch, "variant", "") or ""),
            "task_type": str(getattr(getattr(arch, "task_type", ""), "value", getattr(arch, "task_type", "")) or ""),
            "description": getattr(arch, "description", None),
            "pretrained_path": getattr(arch, "pretrained_path", None),
        },
        "parameters": {
            "epochs": int(params.epochs),
            "batch_size": int(params.batch_size),
            "image_size": int(params.image_size),
            "learning_rate": _as_number(params.learning_rate),
            "lr_scheduler": str(getattr(params, "lr_scheduler", None) or "linear"),
            "patience": int(params.patience),
            "device": str(params.device),
            "workers": int(params.workers),
            "optimizer": str(params.optimizer),
            "use_pretrained": bool(params.use_pretrained),
            "save_period": _save_period_from_params(params.additional_params),
            "augmentation": params.augmentation if isinstance(params.augmentation, dict) else None,
            "loss_weights": params.loss_weights if isinstance(params.loss_weights, dict) else None,
            "additional_params": params.additional_params if isinstance(params.additional_params, dict) else None,
        },
        "metrics": {
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "core_metrics": core_metrics or None,
        },
        "artifacts": {
            "best_weights_path": result.best_weights_path if result else None,
            "last_weights_path": result.last_weights_path if result else None,
            "model_size_mb": model_size_mb,
            "inference_time_ms": inference_time_ms,
            "flops": flops_out,
        },
    }


def compare_runs(db: Session, run_ids: List[str]) -> Dict[str, Any]:
    ids: List[str] = []
    seen = set()
    for value in run_ids or []:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ids.append(normalized)
    if len(ids) < 2:
        raise ValidationError("At least 2 distinct run_ids are required for comparison")

    runs_out: List[Dict[str, Any]] = []
    params_by_run: Dict[str, Dict[str, Any]] = {}
    framework_groups: Dict[str, List[str]] = {}
    for run_id in ids:
        run = TrainingRunService().get_run(db, run_id)
        arch = run.architecture
        engine = str(getattr(arch, "engine", "") or "").strip().lower()
        framework_key, framework_label = _resolve_framework(engine)
        framework_groups.setdefault(framework_key, []).append(str(run.run_id))

        params: Dict[str, Any] = {}
        if run.parameters is not None:
            params = {
                "epochs": int(run.parameters.epochs),
                "batch_size": int(run.parameters.batch_size),
                "image_size": int(run.parameters.image_size),
                "learning_rate": float(run.parameters.learning_rate),
                "lr_scheduler": str(getattr(run.parameters, "lr_scheduler", None) or "linear"),
                "patience": int(run.parameters.patience),
                "device": str(run.parameters.device),
                "workers": int(run.parameters.workers),
                "use_pretrained": bool(run.parameters.use_pretrained),
                "optimizer": str(run.parameters.optimizer),
                "augmentation": run.parameters.augmentation,
                "loss_weights": run.parameters.loss_weights,
            }
            additional = run.parameters.additional_params or {}
            if isinstance(additional, dict):
                for key, value in additional.items():
                    if key not in params:
                        params[key] = value

        best_metrics = run.result.best_metrics if run.result is not None else None
        final_metrics = run.result.final_metrics if run.result is not None else None
        model_size_mb = _as_number(run.result.model_size_mb) if run.result is not None else None
        inference_time_ms = _as_number(run.result.inference_time_ms) if run.result is not None else None
        if inference_time_ms is not None and inference_time_ms <= 0:
            inference_time_ms = None
        metric_summary = _compare_metric_summary(summarize_metrics(db, run, keys=COMPARE_METRIC_KEYS))

        runs_out.append(
            {
                "run_id": run.run_id,
                "name": run.name,
                "status": run.status,
                "project_id": int(run.project_id),
                "standard_dataset_id": int(run.standard_dataset_id),
                "architecture_id": int(run.architecture_id),
                "created_at": run.created_at,
                "engine": engine or None,
                "framework_key": framework_key,
                "framework_label": framework_label,
                "family": str(getattr(arch, "family", "") or "") or None,
                "variant": str(getattr(arch, "variant", "") or "") or None,
                "parameters": params,
                "best_metrics": best_metrics,
                "final_metrics": final_metrics,
                "metric_summary": metric_summary,
                "model_size_mb": model_size_mb,
                "inference_time_ms": inference_time_ms,
            }
        )
        params_by_run[run.run_id] = params

    if len(framework_groups) > 1:
        grouped = {key: sorted(value) for key, value in framework_groups.items()}
        raise FrameworkCompareConflict("Only runs from the same framework can be compared", grouped)

    def _norm(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(value)

    all_keys = sorted({key for values in params_by_run.values() for key in values.keys()})
    diff: Dict[str, Dict[str, Any]] = {}
    for key in all_keys:
        values = {run_id: params_by_run[run_id].get(key) for run_id in params_by_run.keys()}
        if len({_norm(value) for value in values.values()}) > 1:
            diff[key] = values
    return {"runs": runs_out, "parameter_diff": diff}


__all__ = [
    "COMPARE_METRIC_KEYS",
    "ENGINE_FRAMEWORK_MAP",
    "FrameworkCompareConflict",
    "build_report",
    "compare_runs",
    "summarize_metrics",
]
