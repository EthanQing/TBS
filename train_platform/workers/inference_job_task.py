from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from train_platform.core.config import settings
from train_platform.utils.exceptions import ValidationError
from train_platform.utils.path_utils import resolve_temp_path

InferenceFn = Callable[[Path], Dict[str, Any]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: Optional[datetime] = None) -> str:
    return (dt or _utcnow()).isoformat()


def _job_dir(job_id: str) -> Path:
    return settings.temp_dir / "inference_jobs" / str(job_id)


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _results_path(job_id: str) -> Path:
    return _job_dir(job_id) / "results.jsonl"


def _read_status(job_id: str) -> Dict[str, Any]:
    path = _status_path(job_id)
    if not path.exists():
        raise ValidationError("Inference job status not found")
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValidationError("Invalid inference job status payload")
    return data


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    payload = dict(data or {})
    payload["updated_at"] = _to_iso()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _write_status(job_id: str, data: Dict[str, Any]) -> None:
    _write_json_atomic(_status_path(job_id), data)


def _update_status(job_id: str, patch: Dict[str, Any], *, bump_seq: bool = True) -> Dict[str, Any]:
    current = _read_status(job_id)
    current.update(dict(patch or {}))
    current["progress"] = max(0, min(100, int(current.get("progress") or 0)))
    current["processed"] = max(0, int(current.get("processed") or 0))
    current["total"] = max(0, int(current.get("total") or 0))
    if bump_seq:
        current["seq"] = int(current.get("seq") or 0) + 1
    _write_status(job_id, current)
    return current


def _append_item(job_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    status = _read_status(job_id)
    rid = int(status.get("last_result_id") or 0) + 1
    row = dict(item or {})
    row["result_id"] = rid
    path = _results_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    status["last_result_id"] = rid
    status["seq"] = int(status.get("seq") or 0) + 1
    _write_status(job_id, status)
    return row


def _is_cancel_requested(job_id: str) -> bool:
    try:
        return bool(_read_status(job_id).get("cancel_requested"))
    except Exception:
        return True


def _static_temp_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve(strict=False).relative_to(settings.temp_dir.resolve())
        return f"/static/temp/{rel.as_posix()}"
    except Exception:
        return None


def _load_cv2():
    try:
        import cv2

        return cv2
    except Exception as e:
        raise RuntimeError(f"OpenCV is required on inference worker: {type(e).__name__}: {e}") from e


def _predictions_from_output(output: Any) -> List[Dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    preds = output.get("predictions")
    return preds if isinstance(preds, list) else []


def _draw_predictions(
    image: Any,
    predictions: List[Dict[str, Any]],
    *,
    show_labels: bool,
    show_confidence: bool,
) -> None:
    cv2 = _load_cv2()
    for pred in predictions:
        box = pred.get("xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(x))) for x in box]
        except Exception:
            continue
        cls_id = int(pred.get("class_id") or -1)
        color = (
            int((37 * (cls_id + 3)) % 255),
            int((17 * (cls_id + 7)) % 255),
            int((29 * (cls_id + 11)) % 255),
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if not show_labels and not show_confidence:
            continue
        parts = []
        if show_labels:
            parts.append(str(pred.get("class_name") or pred.get("class_id") or "obj"))
        if show_confidence:
            try:
                parts.append(f"{float(pred.get('confidence') or 0):.3f}")
            except Exception:
                pass
        if not parts:
            continue
        cv2.putText(
            image,
            " ".join(parts),
            (max(0, x1), max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def _render_image_result(
    job_id: str,
    *,
    source_path: Path,
    predictions: List[Dict[str, Any]],
    show_labels: bool,
    show_confidence: bool,
) -> Optional[str]:
    cv2 = _load_cv2()
    img = cv2.imread(str(source_path))
    if img is None:
        return None
    _draw_predictions(
        img,
        predictions,
        show_labels=show_labels,
        show_confidence=show_confidence,
    )

    out_dir = _job_dir(job_id) / "output" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{source_path.stem or 'image'}_{int(time.time() * 1000)}.jpg"
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), img):
        return None
    return _static_temp_url(out_path)


def _run_image_job(
    job_id: str,
    *,
    input_tokens: List[str],
    infer_image: InferenceFn,
    show_labels: bool,
    show_confidence: bool,
) -> None:
    total = len(input_tokens)
    _update_status(job_id, {"phase": "inferring", "total": total, "processed": 0, "progress": 0}, bump_seq=True)

    for idx, token in enumerate(input_tokens, start=1):
        if _is_cancel_requested(job_id):
            _update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
            return

        src_path = resolve_temp_path(token)
        src_url = _static_temp_url(src_path) if src_path.exists() else None
        output: Dict[str, Any] = {}
        err: Optional[str] = None
        elapsed_ms: Optional[float] = None

        if not src_path.exists() or not src_path.is_file():
            err = f"Input file not found: {token}"
        else:
            try:
                t0 = time.perf_counter()
                output = infer_image(src_path)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

        predictions = _predictions_from_output(output)
        out_url = None
        if src_path.exists() and predictions:
            try:
                out_url = _render_image_result(
                    job_id,
                    source_path=src_path,
                    predictions=predictions,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                )
            except Exception:
                out_url = None

        _append_item(
            job_id,
            {
                "filename": Path(token).name,
                "token": token,
                "status": "failed" if err else "success",
                "detections": int(len(predictions)),
                "inference_time_ms": elapsed_ms,
                "source_url": src_url,
                "output_url": out_url,
                "output": output if output else None,
                "error_message": err,
            },
        )

        progress = int((idx / total) * 100) if total > 0 else 100
        _update_status(
            job_id,
            {
                "processed": idx,
                "total": total,
                "progress": progress,
                "phase": "inferring" if idx < total else "finalizing",
            },
            bump_seq=True,
        )

    mode = str(_read_status(job_id).get("mode") or "image")
    _update_status(job_id, {"result": {"mode": mode, "items": []}, "phase": "finalizing"}, bump_seq=True)


def _run_video_job(
    job_id: str,
    *,
    video_token: str,
    infer_image: InferenceFn,
    show_labels: bool,
    show_confidence: bool,
) -> None:
    if not video_token:
        raise ValidationError("Missing video token")

    video_path = resolve_temp_path(video_token)
    if not video_path.exists() or not video_path.is_file():
        raise ValidationError(f"Video file not found: {video_token}")

    cv2 = _load_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValidationError(f"Failed to open video: {video_token}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 25.0

    first_frame = None
    if width <= 0 or height <= 0:
        ok0, frame0 = cap.read()
        if not ok0:
            cap.release()
            raise ValidationError("Video has no readable frames")
        first_frame = frame0
        h0, w0 = frame0.shape[:2]
        width, height = int(w0), int(h0)

    out_dir = _job_dir(job_id) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "output.mp4"
    tmp_dir = _job_dir(job_id) / "work"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frame_tmp = tmp_dir / "frame.jpg"

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (max(1, width), max(1, height)),
    )
    if not writer.isOpened():
        cap.release()
        raise ValidationError("Failed to create output video writer")

    start_t = time.perf_counter()
    processed = 0
    _update_status(
        job_id,
        {"phase": "inferring", "total": max(0, total_frames), "processed": 0, "progress": 0},
        bump_seq=True,
    )

    def handle_frame(frame: Any) -> Any:
        cv2.imwrite(str(frame_tmp), frame)
        output = infer_image(frame_tmp)
        predictions = _predictions_from_output(output)
        if predictions:
            _draw_predictions(
                frame,
                predictions,
                show_labels=show_labels,
                show_confidence=show_confidence,
            )
        return frame

    try:
        if first_frame is not None:
            writer.write(handle_frame(first_frame))
            processed += 1
            progress0 = int((processed / total_frames) * 100) if total_frames > 0 else 0
            _update_status(
                job_id,
                {
                    "processed": processed,
                    "total": max(total_frames, processed),
                    "progress": max(0, min(99, progress0)),
                    "phase": "inferring",
                },
                bump_seq=True,
            )

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if _is_cancel_requested(job_id):
                _update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
                return

            writer.write(handle_frame(frame))
            processed += 1
            progress = int((processed / total_frames) * 100) if total_frames > 0 else 0
            _update_status(
                job_id,
                {
                    "processed": processed,
                    "total": max(total_frames, processed),
                    "progress": max(0, min(99, progress)),
                    "phase": "inferring",
                },
                bump_seq=True,
            )
    finally:
        cap.release()
        writer.release()
        try:
            frame_tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass

    if _is_cancel_requested(job_id):
        _update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
        return

    if not out_video.exists() or out_video.stat().st_size <= 0:
        raise ValidationError("Output video was not generated")
    video_url = _static_temp_url(out_video)
    if not video_url:
        raise ValidationError("Failed to resolve output video URL")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    _update_status(
        job_id,
        {
            "phase": "finalizing",
            "progress": 100,
            "processed": processed,
            "total": max(total_frames, processed),
            "result": {
                "mode": "video",
                "video": {
                    "output_url": video_url,
                    "total_frames": max(total_frames, processed),
                    "processed_frames": processed,
                    "fps": round(float(fps), 3),
                    "width": int(width) if width > 0 else None,
                    "height": int(height) if height > 0 else None,
                    "total_time_ms": elapsed_ms,
                },
            },
        },
        bump_seq=True,
    )


def run_inference_job(
    job_id: str,
    *,
    mode: str,
    input_tokens: List[str],
    video_token: Optional[str],
    infer_image: InferenceFn,
    show_labels: bool,
    show_confidence: bool,
) -> None:
    try:
        _update_status(
            job_id,
            {"status": "running", "phase": "preparing", "progress": 0, "error_message": None},
            bump_seq=True,
        )
        if _is_cancel_requested(job_id):
            _update_status(job_id, {"status": "cancelled", "phase": "cancelled"}, bump_seq=True)
            return

        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode in {"image", "batch"}:
            tokens = [str(x).strip() for x in (input_tokens or []) if str(x).strip()]
            if normalized_mode == "image":
                tokens = tokens[:1]
            _run_image_job(
                job_id,
                input_tokens=tokens,
                infer_image=infer_image,
                show_labels=bool(show_labels),
                show_confidence=bool(show_confidence),
            )
        elif normalized_mode == "video":
            _run_video_job(
                job_id,
                video_token=str(video_token or "").strip(),
                infer_image=infer_image,
                show_labels=bool(show_labels),
                show_confidence=bool(show_confidence),
            )
        else:
            raise ValidationError(f"Unsupported inference job mode: {mode}")

        final = _read_status(job_id)
        if str(final.get("status")) == "running":
            _update_status(
                job_id,
                {"status": "completed", "phase": "done", "progress": 100, "error_message": None},
                bump_seq=True,
            )
    except Exception as e:
        try:
            _update_status(
                job_id,
                {
                    "status": "failed",
                    "phase": "failed",
                    "progress": 100,
                    "error_message": f"{type(e).__name__}: {e}",
                },
                bump_seq=True,
            )
        except Exception:
            pass


def run_video_frame_sampling(
    *,
    video_token: str,
    frame_interval: int,
    infer_image: InferenceFn,
) -> Dict[str, Any]:
    video_path = resolve_temp_path(video_token)
    if not video_path.exists() or not video_path.is_file():
        raise ValidationError(f"Video file not found: {video_token}")

    cv2 = _load_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValidationError(f"Failed to open video: {video_token}")

    frames_dir = settings.temp_dir / "inference_video_frames" / uuid.uuid4().hex
    frames_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(frame_interval or 1))
    results: List[Dict[str, Any]] = []
    total_start = time.time()
    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                result: Dict[str, Any] = {"frame_index": frame_idx}
                try:
                    result["output"] = infer_image(frame_path)
                    result["error_message"] = None
                except Exception as e:
                    result["output"] = None
                    result["error_message"] = f"{type(e).__name__}: {e}"
                results.append(result)
                processed += 1

            frame_idx += 1
    finally:
        cap.release()

    return {
        "results": results,
        "total_frames": total_frames,
        "processed_frames": processed,
        "total_time_ms": round((time.time() - total_start) * 1000.0, 1),
    }
