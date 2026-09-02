from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from train_platform.platform.runtime import ModelWorkerClient
from train_platform.services.v3.dataset_common import guess_label_path, read_yolo_boxes
from train_platform.utils.exceptions import ValidationError

from .metrics import compute_detection_metrics
from .preparation import PreparedEvaluation, materialize_ultralytics_eval_data


class EvaluationObserver(Protocol):
    def is_cancel_requested(self) -> bool: ...

    def update_phase(
        self,
        phase: str,
        *,
        progress: int,
        total: int | None = None,
        processed: int | None = None,
    ) -> None: ...

    def update_progress(self, *, processed: int, progress: int) -> None: ...

    def emit_item(self, item: Mapping[str, Any]) -> None: ...

    def mark_cancelled(self) -> None: ...


@dataclass(frozen=True)
class EvaluationRunResult:
    metrics: dict[str, Any]
    evaluated_images: int
    skipped_images: int
    failed_images: int
    processed: int
    elapsed_ms: float
    cancelled: bool = False


def run_evaluation(
    prepared: PreparedEvaluation,
    *,
    job_dir: Path,
    worker_client: ModelWorkerClient,
    observer: EvaluationObserver,
) -> EvaluationRunResult:
    """Execute one prepared evaluation using its engine's current strategy."""
    total = prepared.total_images
    if total <= 0:
        raise ValidationError("No labeled images were available for evaluation")

    started = time.perf_counter()
    skipped = int(prepared.skipped_images)
    failed = 0
    evaluated = 0

    def cancelled_result(processed: int) -> EvaluationRunResult:
        observer.mark_cancelled()
        return EvaluationRunResult(
            metrics={},
            evaluated_images=evaluated,
            skipped_images=skipped,
            failed_images=failed,
            processed=processed,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
            cancelled=True,
        )

    observer.update_phase("validating", progress=1, total=total, processed=0)
    engine = prepared.model.engine

    if engine == "ultralytics-yolo":
        if observer.is_cancel_requested():
            return cancelled_result(0)

        data_yaml = materialize_ultralytics_eval_data(prepared, job_dir)
        observer.update_phase("calculating", progress=5)
        metrics = worker_client.validate_ultralytics_yolo(
            weights_path=prepared.model.weights_path,
            data_yaml=data_yaml,
            conf=float(prepared.conf),
            iou=float(prepared.iou),
        )
        if observer.is_cancel_requested():
            return cancelled_result(0)

        metrics.update(
            {
                "evaluated_images": int(total),
                "skipped_images": int(skipped),
                "failed_images": 0,
                "elapsed_ms": float(
                    metrics.get("elapsed_ms")
                    or round((time.perf_counter() - started) * 1000.0, 2)
                ),
            }
        )
        return EvaluationRunResult(
            metrics=metrics,
            evaluated_images=total,
            skipped_images=skipped,
            failed_images=0,
            processed=total,
            elapsed_ms=float(metrics["elapsed_ms"]),
        )

    observer.update_phase("inferring", progress=1, processed=0)
    ground_truth_by_image: dict[str, list[dict[str, Any]]] = {}
    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    class_names = list(prepared.class_names)

    for index, rel_path in enumerate(prepared.labeled_paths, start=1):
        if observer.is_cancel_requested():
            return cancelled_result(index - 1)

        image_path = prepared.dataset_root / rel_path
        label_path = guess_label_path(prepared.dataset_root, rel_path)
        progress = int((index / total) * 100) if total else 100
        base_item = {"filename": Path(rel_path).name, "image_path": rel_path}

        if not image_path.exists() or not image_path.is_file():
            skipped += 1
            observer.emit_item(
                {
                    **base_item,
                    "status": "skipped",
                    "gt_count": 0,
                    "prediction_count": 0,
                    "error_message": "Image file not found",
                }
            )
            observer.update_progress(processed=index, progress=progress)
            continue

        if not label_path.exists() or not label_path.is_file():
            skipped += 1
            observer.emit_item(
                {
                    **base_item,
                    "status": "skipped",
                    "gt_count": 0,
                    "prediction_count": 0,
                    "error_message": "YOLO label file not found",
                }
            )
            observer.update_progress(processed=index, progress=progress)
            continue

        _width, _height, gt_boxes = read_yolo_boxes(prepared.dataset_root, rel_path, class_names)
        if not gt_boxes:
            skipped += 1
            observer.emit_item(
                {
                    **base_item,
                    "status": "skipped",
                    "gt_count": 0,
                    "prediction_count": 0,
                    "error_message": "No valid YOLO boxes",
                }
            )
            observer.update_progress(processed=index, progress=progress)
            continue

        inference_started = time.perf_counter()
        try:
            if observer.is_cancel_requested():
                return cancelled_result(index - 1)
            output = worker_client.execute_model(
                engine=engine or "ultralytics-yolo",
                weights_path=prepared.model.weights_path,
                image_path=image_path,
                conf=float(prepared.conf),
                iou=float(prepared.iou),
                config_path=prepared.model.config_path,
            )
            predictions = output.get("predictions") if isinstance(output, dict) else []
            predictions = predictions if isinstance(predictions, list) else []
            inference_time_ms = round((time.perf_counter() - inference_started) * 1000.0, 2)
            ground_truth_by_image[rel_path] = gt_boxes
            predictions_by_image[rel_path] = predictions
            evaluated += 1
            observer.emit_item(
                {
                    **base_item,
                    "status": "success",
                    "gt_count": len(gt_boxes),
                    "prediction_count": len(predictions),
                    "inference_time_ms": inference_time_ms,
                }
            )
        except Exception as exc:
            failed += 1
            observer.emit_item(
                {
                    **base_item,
                    "status": "failed",
                    "gt_count": len(gt_boxes),
                    "prediction_count": 0,
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )

        if observer.is_cancel_requested():
            return cancelled_result(index)
        observer.update_progress(processed=index, progress=progress)

    if evaluated <= 0:
        raise ValidationError("No labeled images were available for evaluation")

    if observer.is_cancel_requested():
        return cancelled_result(total)

    observer.update_phase("calculating", progress=99)
    metrics = compute_detection_metrics(
        ground_truth_by_image,
        predictions_by_image,
        iou_threshold=float(prepared.iou),
        class_names=class_names,
    )
    metrics.update(
        {
            "evaluated_images": int(evaluated),
            "skipped_images": int(skipped),
            "failed_images": int(failed),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    )
    return EvaluationRunResult(
        metrics=metrics,
        evaluated_images=evaluated,
        skipped_images=skipped,
        failed_images=failed,
        processed=total,
        elapsed_ms=float(metrics["elapsed_ms"]),
    )


__all__ = ["EvaluationObserver", "EvaluationRunResult", "run_evaluation"]
