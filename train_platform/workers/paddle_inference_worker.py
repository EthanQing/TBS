from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from train_platform.core.config import settings

app = FastAPI(title="Paddle Inference Worker", version="1.0")


class PaddleInferenceRequest(BaseModel):
    config_path: str = Field(..., min_length=1)
    weights_path: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    conf: float = 0.5
    iou: float = 0.45


class InferenceResponse(BaseModel):
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    inference_time_ms: Optional[float] = None


_CACHE_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Tuple[Any, Dict[int, str]]] = {}


def _verify_internal_auth(x_internal_token: Optional[str] = Header(default=None, alias="X-Internal-Token")) -> None:
    expected = str(settings.internal_api_token or "").strip()
    if not expected:
        return
    provided = str(x_internal_token or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized internal request")


def _cache_key(config_path: Path, weights_path: Path) -> str:
    return f"{config_path.resolve(strict=False)}::{weights_path.resolve(strict=False)}"


def _extract_label_names(cfg: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    try:
        labels = cfg.get("label_list")
    except Exception:
        labels = None
    if isinstance(labels, (list, tuple)):
        for i, name in enumerate(labels):
            out[int(i)] = str(name)
        return out
    return out


def _load_trainer(config_path: Path, weights_path: Path) -> Tuple[Any, Dict[int, str]]:
    try:
        import paddle
        from ppdet.core.workspace import load_config
        from ppdet.engine import Trainer
    except Exception as e:
        raise RuntimeError(f"PaddleDetection runtime is unavailable: {type(e).__name__}: {e}") from e

    cfg = load_config(str(config_path))
    try:
        use_gpu = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
        if isinstance(cfg, dict):
            cfg["use_gpu"] = use_gpu
        elif hasattr(cfg, "use_gpu"):
            setattr(cfg, "use_gpu", use_gpu)
    except Exception:
        pass

    trainer = Trainer(cfg, mode="test")
    w = str(weights_path)
    if w.endswith(".pdparams"):
        w = w[: -len(".pdparams")]
    trainer.load_weights(w)
    names = _extract_label_names(cfg)
    return trainer, names


def _get_trainer(config_path: Path, weights_path: Path) -> Tuple[Any, Dict[int, str]]:
    key = _cache_key(config_path, weights_path)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        loaded = _load_trainer(config_path, weights_path)
        _MODEL_CACHE[key] = loaded
        return loaded


def _to_predictions(outs: Any, names: Dict[int, str], conf_threshold: float) -> Dict[str, Any]:
    bbox = None
    if isinstance(outs, dict):
        bbox = outs.get("bbox")
    if bbox is None:
        return {"predictions": [], "names": names}

    if hasattr(bbox, "numpy"):
        try:
            bbox = bbox.numpy()
        except Exception:
            pass
    try:
        rows = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
    except Exception:
        rows = []

    preds = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        try:
            cls_id = int(row[0])
            score = float(row[1])
            if score < float(conf_threshold):
                continue
            x1, y1, x2, y2 = [float(row[i]) for i in range(2, 6)]
        except Exception:
            continue
        preds.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id),
                "confidence": score,
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return {"predictions": preds, "names": names}


def _run_paddle_det(config_path: Path, weights_path: Path, image_path: Path, *, conf: float, iou: float) -> Dict[str, Any]:
    trainer, names = _get_trainer(config_path, weights_path)
    with _INFER_LOCK:
        results = trainer.predict(
            [str(image_path)],
            draw_threshold=float(conf),
            output_dir="output",
            save_results=False,
            visualize=False,
        )
    if not results:
        return {"predictions": [], "names": names}
    outs = results[0] if isinstance(results, list) else results
    return _to_predictions(outs, names, conf_threshold=float(conf))


@app.post("/internal/inference/paddle-det", response_model=InferenceResponse)
def run_inference(
    req: PaddleInferenceRequest,
    _: None = Depends(_verify_internal_auth),
) -> InferenceResponse:
    config_path = Path(req.config_path)
    if not config_path.exists() or not config_path.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")

    weights_path = Path(req.weights_path)
    if not weights_path.exists() or not weights_path.is_file():
        raise HTTPException(status_code=404, detail=f"Weights not found: {weights_path}")

    image_path = Path(req.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    t0 = time.perf_counter()
    try:
        output = _run_paddle_det(
            config_path=config_path,
            weights_path=weights_path,
            image_path=image_path,
            conf=float(req.conf),
            iou=float(req.iou),
        )
    except Exception as e:
        return InferenceResponse(error=f"{type(e).__name__}: {e}")
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return InferenceResponse(output=output, inference_time_ms=dt_ms)


if __name__ == "__main__":
    import uvicorn

    host = (
        str(settings.worker_bind_host).strip()
        or os.getenv("PADDLE_INFERENCE_WORKER_HOST")
        or "0.0.0.0"
    )
    port = int(os.getenv("PADDLE_INFERENCE_WORKER_PORT", "18003"))
    uvicorn.run("train_platform.workers.paddle_inference_worker:app", host=host, port=port)
