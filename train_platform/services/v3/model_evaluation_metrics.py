from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EvalBox:
    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0
    class_name: Optional[str] = None


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return safe_div(inter, union)


def _normalize_gt(raw: Dict[str, Any], image_id: str) -> Optional[EvalBox]:
    try:
        return EvalBox(
            image_id=str(image_id),
            class_id=int(raw.get("class_id")),
            class_name=str(raw.get("class_name")) if raw.get("class_name") is not None else None,
            xyxy=(
                float(raw.get("x1")),
                float(raw.get("y1")),
                float(raw.get("x2")),
                float(raw.get("y2")),
            ),
            confidence=1.0,
        )
    except Exception:
        return None


def _normalize_pred(raw: Dict[str, Any], image_id: str) -> Optional[EvalBox]:
    box = raw.get("xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return EvalBox(
            image_id=str(image_id),
            class_id=int(raw.get("class_id")),
            class_name=str(raw.get("class_name")) if raw.get("class_name") is not None else None,
            xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            confidence=float(raw.get("confidence") or 0.0),
        )
    except Exception:
        return None


def _match_counts(
    gts: List[EvalBox],
    preds: List[EvalBox],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    fp = 0
    ordered = sorted(enumerate(preds), key=lambda item: item[1].confidence, reverse=True)
    for _pred_idx, pred in ordered:
        best_i = -1
        best_iou = 0.0
        for gt_i, gt in enumerate(gts):
            if gt_i in matched:
                continue
            if gt.image_id != pred.image_id or gt.class_id != pred.class_id:
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_i = gt_i
        if best_i >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_i)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(gts) - tp)
    return tp, fp, fn


def average_precision_for_class(gts: List[EvalBox], preds: List[EvalBox], *, iou_threshold: float) -> float:
    if not gts:
        return 0.0

    matched: set[int] = set()
    tps: list[int] = []
    fps: list[int] = []
    ordered = sorted(enumerate(preds), key=lambda item: item[1].confidence, reverse=True)
    for _idx, pred in ordered:
        best_i = -1
        best_iou = 0.0
        for gt_i, gt in enumerate(gts):
            if gt_i in matched:
                continue
            if gt.image_id != pred.image_id or gt.class_id != pred.class_id:
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_i = gt_i
        if best_i >= 0 and best_iou >= float(iou_threshold):
            matched.add(best_i)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    if not tps:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    cum_tp = 0
    cum_fp = 0
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        precisions.append(safe_div(cum_tp, cum_tp + cum_fp))
        recalls.append(safe_div(cum_tp, len(gts)))

    ap = 0.0
    for step in range(101):
        recall_level = step / 100.0
        candidates = [p for p, r in zip(precisions, recalls) if r >= recall_level]
        ap += max(candidates) if candidates else 0.0
    return ap / 101.0


def compute_detection_metrics(
    ground_truth_by_image: Dict[str, List[Dict[str, Any]]],
    predictions_by_image: Dict[str, List[Dict[str, Any]]],
    *,
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    gts: list[EvalBox] = []
    preds: list[EvalBox] = []
    class_name_map: dict[int, str] = {}

    for image_id, rows in ground_truth_by_image.items():
        for row in rows or []:
            box = _normalize_gt(row, image_id)
            if not box:
                continue
            gts.append(box)
            if box.class_name:
                class_name_map.setdefault(box.class_id, box.class_name)

    for image_id, rows in predictions_by_image.items():
        for row in rows or []:
            box = _normalize_pred(row, image_id)
            if not box:
                continue
            preds.append(box)
            if box.class_name:
                class_name_map.setdefault(box.class_id, box.class_name)

    if class_names:
        for idx, name in enumerate(class_names):
            class_name_map.setdefault(int(idx), str(name))

    tp, fp, fn = _match_counts(gts, preds, iou_threshold=float(iou_threshold))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    class_ids = sorted({b.class_id for b in gts} | {b.class_id for b in preds})
    thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    class_metrics: list[dict[str, Any]] = []
    ap50_values: list[float] = []
    ap5095_values: list[float] = []

    for class_id in class_ids:
        cgts = [b for b in gts if b.class_id == class_id]
        cpreds = [b for b in preds if b.class_id == class_id]
        ctp, cfp, cfn = _match_counts(cgts, cpreds, iou_threshold=float(iou_threshold))
        cp = safe_div(ctp, ctp + cfp)
        cr = safe_div(ctp, ctp + cfn)
        cf1 = safe_div(2.0 * cp * cr, cp + cr)
        ap50 = average_precision_for_class(cgts, cpreds, iou_threshold=0.5)
        ap5095 = safe_div(
            sum(average_precision_for_class(cgts, cpreds, iou_threshold=t) for t in thresholds),
            len(thresholds),
        )
        if cgts:
            ap50_values.append(ap50)
            ap5095_values.append(ap5095)
        class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": class_name_map.get(int(class_id)),
                "gt_count": len(cgts),
                "pred_count": len(cpreds),
                "tp": ctp,
                "fp": cfp,
                "fn": cfn,
                "precision": cp,
                "recall": cr,
                "f1": cf1,
                "ap50": ap50,
                "ap50_95": ap5095,
            }
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": safe_div(sum(ap50_values), len(ap50_values)),
        "map50_95": safe_div(sum(ap5095_values), len(ap5095_values)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_targets": len(gts),
        "total_predictions": len(preds),
        "class_metrics": class_metrics,
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            out = value.tolist()
            return out if isinstance(out, list) else [out]
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def extract_ultralytics_val_metrics(results: Any, elapsed_ms: float) -> Dict[str, Any]:
    """Translate Ultralytics validation output into the evaluation result shape."""
    box = getattr(results, "box", None)
    names = getattr(results, "names", None) or {}
    if not isinstance(names, dict):
        names = {}

    precision = _as_float(getattr(box, "mp", 0.0))
    recall = _as_float(getattr(box, "mr", 0.0))
    map50 = _as_float(getattr(box, "map50", 0.0))
    map50_95 = _as_float(getattr(box, "map", 0.0))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ap_class_index = [int(x) for x in _to_list(getattr(box, "ap_class_index", []))]
    p_values = [_as_float(x) for x in _to_list(getattr(box, "p", []))]
    r_values = [_as_float(x) for x in _to_list(getattr(box, "r", []))]
    maps = [_as_float(x) for x in _to_list(getattr(box, "maps", []))]

    all_ap = _to_list(getattr(box, "all_ap", []))
    ap50_values: list[float] = []
    if all_ap:
        for row in all_ap:
            values = _to_list(row)
            ap50_values.append(_as_float(values[0] if values else 0.0))

    nt_per_class = [int(_as_float(x)) for x in _to_list(getattr(results, "nt_per_class", []))]

    name_class_ids = set()
    for key in names.keys():
        try:
            name_class_ids.add(int(key))
        except Exception:
            continue

    metric_index_by_class = {class_id: idx for idx, class_id in enumerate(ap_class_index)}
    compact_metric_arrays = bool(ap_class_index) and any(
        class_id >= len(ap_class_index) for class_id in ap_class_index
    )
    has_full_target_counts = bool(nt_per_class) and (
        not ap_class_index or len(nt_per_class) > max(ap_class_index)
    )
    target_class_ids = (
        set(range(len(nt_per_class)))
        if has_full_target_counts
        else set(ap_class_index[: len(nt_per_class)])
    )
    class_ids = sorted(set(ap_class_index) | target_class_ids | name_class_ids)

    def _metric_value(values: list[float], class_id: int) -> float:
        metric_idx = metric_index_by_class.get(class_id)
        if metric_idx is not None and metric_idx < len(values):
            return values[metric_idx]
        if not compact_metric_arrays and 0 <= class_id < len(values):
            return values[class_id]
        return 0.0

    def _target_count(class_id: int) -> int:
        metric_idx = metric_index_by_class.get(class_id)
        if has_full_target_counts and 0 <= class_id < len(nt_per_class):
            return nt_per_class[class_id]
        if metric_idx is not None and metric_idx < len(nt_per_class):
            return nt_per_class[metric_idx]
        return 0

    class_metrics: list[dict[str, Any]] = []
    total_targets = 0
    total_predictions = 0
    est_tp_total = 0
    est_fp_total = 0
    est_fn_total = 0

    for class_id in class_ids:
        p = _metric_value(p_values, class_id)
        r = _metric_value(r_values, class_id)
        ap50 = _metric_value(ap50_values, class_id)
        ap5095 = _metric_value(maps, class_id)
        gt_count = _target_count(class_id)
        est_tp = int(round(r * gt_count)) if gt_count > 0 else 0
        pred_count = int(round(est_tp / p)) if p > 0 else 0
        est_fp = max(0, pred_count - est_tp)
        est_fn = max(0, gt_count - est_tp)
        cf1 = (2.0 * p * r / (p + r)) if (p + r) else 0.0
        total_targets += gt_count
        total_predictions += pred_count
        est_tp_total += est_tp
        est_fp_total += est_fp
        est_fn_total += est_fn
        class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": str(names.get(class_id, names.get(str(class_id), class_id))),
                "gt_count": int(gt_count),
                "pred_count": int(pred_count),
                "tp": int(est_tp),
                "fp": int(est_fp),
                "fn": int(est_fn),
                "precision": float(p),
                "recall": float(r),
                "f1": float(cf1),
                "ap50": float(ap50),
                "ap50_95": float(ap5095),
            }
        )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "tp": int(est_tp_total),
        "fp": int(est_fp_total),
        "fn": int(est_fn_total),
        "total_targets": int(total_targets),
        "total_predictions": int(total_predictions),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "class_metrics": class_metrics,
    }
