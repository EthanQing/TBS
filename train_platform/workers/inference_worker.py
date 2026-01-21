from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Inference Worker", version="1.0")


class InferenceRequest(BaseModel):
    weights_path: str = Field(..., min_length=1)
    image_path: str = Field(..., min_length=1)
    conf: float = 0.5
    iou: float = 0.45


class InferenceResponse(BaseModel):
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelConversionRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    opset: Optional[int] = None
    dynamic: bool = True


class ExportOnnxRequest(BaseModel):
    src_pt: str = Field(..., min_length=1)
    out_onnx: str = Field(..., min_length=1)
    dynamic: bool = True
    opset: Optional[int] = None
    imgsz: Optional[int] = None


class WorkerStatusResponse(BaseModel):
    status: str
    error: Optional[str] = None


def _run_ultralytics_yolo(weights_path: Path, image_path: Path, *, conf: float, iou: float) -> Dict[str, Any]:
    from ultralytics import YOLO

    # Reuse our training-side safe-load patch to avoid common torch.serialization issues.
    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass

    model = YOLO(str(weights_path))
    results = model.predict(source=str(image_path), conf=float(conf), iou=float(iou), verbose=False)
    if not results:
        return {"predictions": [], "names": {}}

    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(model, "names", None) or {}
    out: Dict[str, Any] = {"predictions": [], "names": names}

    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return out

    for b in boxes:
        try:
            cls_id = int(getattr(b, "cls")[0])
        except Exception:
            try:
                cls_id = int(b.cls)
            except Exception:
                cls_id = -1
        try:
            conf_v = float(getattr(b, "conf")[0])
        except Exception:
            try:
                conf_v = float(b.conf)
            except Exception:
                conf_v = 0.0
        try:
            xyxy = getattr(b, "xyxy")[0].tolist()
            xyxy = [float(x) for x in xyxy]
        except Exception:
            xyxy = None

        out["predictions"].append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id) if isinstance(names, dict) else None,
                "confidence": conf_v,
                "xyxy": xyxy,
            }
        )

    return out


@app.post("/internal/inference/yolo", response_model=InferenceResponse)
def run_inference(req: InferenceRequest) -> InferenceResponse:
    weights_path = Path(req.weights_path)
    if not weights_path.exists() or not weights_path.is_file():
        raise HTTPException(status_code=404, detail=f"Weights not found: {weights_path}")

    image_path = Path(req.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        output = _run_ultralytics_yolo(weights_path, image_path, conf=req.conf, iou=req.iou)
    except Exception as e:
        return InferenceResponse(error=f"{type(e).__name__}: {e}")

    return InferenceResponse(output=output)


@app.post("/internal/model-conversions/pt-to-onnx", response_model=WorkerStatusResponse)
def start_model_conversion(req: ModelConversionRequest) -> WorkerStatusResponse:
    try:
        from train_platform.api.v2.model_conversions import _run_pt_to_onnx
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"Failed to import conversion worker: {type(e).__name__}: {e}")

    def _runner() -> None:
        _run_pt_to_onnx(req.job_id, opset=req.opset, dynamic=req.dynamic)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return WorkerStatusResponse(status="started")


@app.post("/internal/training-runs/export-onnx", response_model=WorkerStatusResponse)
def export_training_onnx(req: ExportOnnxRequest) -> WorkerStatusResponse:
    src_pt = Path(req.src_pt)
    if not src_pt.exists() or not src_pt.is_file():
        raise HTTPException(status_code=404, detail=f"Weights not found: {src_pt}")

    out_onnx = Path(req.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"Ultralytics not installed: {type(e).__name__}: {e}")

    # Ensure safe torch load on Windows for Ultralytics weights.
    try:
        from train_platform.training.plugins.ultralytics_yolo import _apply_torch_safe_load_patches  # type: ignore

        _apply_torch_safe_load_patches()
    except Exception:
        pass

    try:
        model = YOLO(str(src_pt))

        export_kwargs: Dict[str, Any] = {"dynamic": bool(req.dynamic)}
        if req.opset is not None:
            export_kwargs["opset"] = int(req.opset)
        if req.imgsz is not None:
            export_kwargs["imgsz"] = int(req.imgsz)

        exported = model.export(format="onnx", **export_kwargs)

        if not out_onnx.exists():
            exported_path: Optional[Path] = None
            try:
                if exported:
                    exported_path = Path(str(exported)).resolve(strict=False)
            except Exception:
                exported_path = None

            if exported_path and exported_path.exists():
                try:
                    exported_path.replace(out_onnx)
                except Exception:
                    pass
    except Exception as e:
        return WorkerStatusResponse(status="error", error=f"{type(e).__name__}: {e}")

    if not out_onnx.exists():
        return WorkerStatusResponse(status="error", error=f"ONNX export did not produce {out_onnx.name}")

    return WorkerStatusResponse(status="ok")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("INFERENCE_WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("INFERENCE_WORKER_PORT", "18002"))
    uvicorn.run("train_platform.workers.inference_worker:app", host=host, port=port)
