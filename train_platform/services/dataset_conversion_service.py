from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from train_platform.utils.exceptions import ValidationError


class DatasetConversionService:
    def _get_bbox_from_points(self, points: List[List[float]]) -> Tuple[float, float, float, float]:
        """
        Compute bounding box [x_min, y_min, x_max, y_max] from a list of points [[x, y], ...].
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def _normalize_2_yolo(
        self, bbox: Tuple[float, float, float, float], img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Normalize bbox to YOLO format (x_center, y_center, width, height), values in [0, 1].
        """
        x_min, y_min, x_max, y_max = bbox

        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(float(img_width) - 1.0, x_max), min(float(img_height) - 1.0, y_max)

        center_x = x_min + (x_max - x_min) / 2.0
        center_y = y_min + (y_max - y_min) / 2.0

        return (
            round(center_x / img_width, 6),
            round(center_y / img_height, 6),
            round((x_max - x_min) / img_width, 6),
            round((y_max - y_min) / img_height, 6),
        )

    def _read_json(self, path: Path) -> dict:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                text = path.read_text(encoding="gbk", errors="ignore")
            except Exception as e:
                raise ValidationError(f"Failed to read json: {path.name}") from e

        try:
            data = json.loads(text)
        except Exception as e:
            raise ValidationError(f"Invalid json: {path.name}") from e

        if not isinstance(data, dict):
            raise ValidationError(f"Invalid json (expected object): {path.name}")
        return data

    def _coerce_points(self, points_obj: Any) -> list[list[float]]:
        if not isinstance(points_obj, list):
            return []
        out: list[list[float]] = []
        for p in points_obj:
            x = y = None
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
            elif isinstance(p, dict) and "x" in p and "y" in p:
                x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue
            try:
                out.append([float(x), float(y)])
            except Exception:
                continue
        return out

    def _label_file_stem(self, json_path: Path) -> str:
        # Per requirements: do not rely on imagePath/filename from json; use the json filename.
        return str(json_path.stem)

    def _extract_image_size(self, data: dict) -> tuple[int | None, int | None]:
        def _as_int(v: Any) -> int | None:
            try:
                if v is None:
                    return None
                return int(v)
            except Exception:
                return None

        # Per requirements: only use imageWidth/imageHeight from the json (case variations allowed).
        w = _as_int(data.get("imageWidth"))
        if not w:
            w = _as_int(data.get("imagewidth"))
        h = _as_int(data.get("imageHeight"))
        if not h:
            h = _as_int(data.get("imageheight"))

        if w and h:
            return w, h
        return None, None

    def _iter_shapes(self, data: dict) -> list[dict]:
        v = data.get("shapes")
        if not isinstance(v, list):
            return []
        return [x for x in v if isinstance(x, dict)]

    def _read_existing_class_names(self, path: Path) -> tuple[list[str], dict[str, int]]:
        if not path.exists() or not path.is_file():
            return [], {}
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                text = path.read_text(encoding="gbk", errors="ignore")
            except Exception:
                return [], {}

        names: list[str] = []
        mapping: dict[str, int] = {}
        for line in text.splitlines():
            s = str(line).strip()
            if not s:
                continue
            if s in mapping:
                continue
            mapping[s] = len(names)
            names.append(s)
        return names, mapping

    def conversion_json_2_yolo(
        self,
        storage_path: str,
        *,
        output_labels_dir: str | Path | None = None,
        class_names_path: str | Path | None = None,
        write_files: bool = True,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> Dict[str, List[str]]:
        """
        Convert annotations from JSON format to YOLO format.

        - Uses only: imageWidth/imageHeight + shapes[].label/points/shape_type
        - Auto-discovers classes: first time a label appears, it is appended to class_names.txt and assigned a new id.
        - Requires image width/height in json; does NOT infer size by opening image files.
        - Progress reporting: if on_progress is provided, it is called as (processed, total, current_json_name).

        Args:
            storage_path: Directory containing json files (and usually images).
            output_labels_dir: Where to write YOLO .txt labels. Defaults to <storage_path>/labels.
            class_names_path: Where to store class names (one per line). If omitted, it is inferred from output_labels_dir.
            write_files: If False, only returns the mapping without writing any files.
            on_progress: Optional callback for progress reporting.
        """
        src_dir = Path(storage_path).expanduser().resolve(strict=False)
        if not src_dir.exists() or not src_dir.is_dir():
            raise ValidationError(f"storage_path must be an existing directory: {storage_path}")

        labels_dir = Path(output_labels_dir) if output_labels_dir is not None else (src_dir / "labels")
        labels_dir = labels_dir.expanduser().resolve(strict=False)
        if write_files:
            labels_dir.mkdir(parents=True, exist_ok=True)

        # Decide where to store the class list. Prefer caller override; otherwise infer from labels_dir.
        if class_names_path is not None:
            class_names_file = Path(class_names_path).expanduser().resolve(strict=False)
        else:
            dataset_dir = src_dir
            try:
                if labels_dir.name.lower() == "labels":
                    dataset_dir = labels_dir.parent
                elif labels_dir.parent.name.lower() == "labels":
                    dataset_dir = labels_dir.parent.parent
            except Exception:
                dataset_dir = src_dir
            class_names_file = (dataset_dir / "class_names.txt").resolve(strict=False)

        class_names, class_name_to_id = self._read_existing_class_names(class_names_file)
        out: Dict[str, List[str]] = {}

        json_files = sorted([p for p in src_dir.rglob("*.json") if p.is_file()])
        if not json_files:
            raise ValidationError("No .json annotation files found")

        total = int(len(json_files))
        processed = 0

        for json_path in json_files:
            data = self._read_json(json_path)

            label_stem = self._label_file_stem(json_path)
            w, h = self._extract_image_size(data)

            if not w or not h:
                raise ValidationError(
                    f"Missing image size for {json_path.name} (imageWidth/imageHeight required in json)"
                )

            lines: list[str] = []
            for shape in self._iter_shapes(data):
                shape_type = shape.get("shape_type")
                if not isinstance(shape_type, str) or not shape_type.strip():
                    continue
                st = shape_type.strip().lower()
                # if st not in ("rectangle", "polygon", ""):
                #     continue

                raw_label = shape.get("label")
                if not isinstance(raw_label, str) or not raw_label.strip():
                    continue
                label = raw_label.strip()

                class_id = class_name_to_id.get(label)
                if class_id is None:
                    class_id = len(class_names)
                    class_name_to_id[label] = class_id
                    class_names.append(label)

                points = self._coerce_points(shape.get("points"))
                if not points:
                    continue

                bbox = self._get_bbox_from_points(points)
                x, y, bw, bh = self._normalize_2_yolo(bbox, int(w), int(h))
                lines.append(f"{int(class_id)} {x} {y} {bw} {bh}")

            out[label_stem] = lines

            if write_files:
                out_path = (labels_dir / f"{label_stem}.txt").resolve(strict=False)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")
                    else:
                        # Keep an empty file for images with no objects (common YOLO convention).
                        f.write("")

            processed += 1
            if on_progress is not None:
                try:
                    on_progress(int(processed), int(total), str(json_path.name))
                except Exception:
                    # Progress hooks must never break conversion.
                    pass

        # Persist discovered class names for data.yaml generation (FileService prefers this file).
        if write_files:
            class_names_file.parent.mkdir(parents=True, exist_ok=True)
            with open(class_names_file, "w", encoding="utf-8") as f:
                if class_names:
                    f.write("\n".join(class_names) + "\n")

        return out

    def _unzip_2_tempdir(self, zip_path: Path) -> Path:
        """
        Unzip the provided zip file to a temporary directory under BASE_TEMP_DIR.

        Returns the extracted root directory. If the archive has a single top-level directory,
        returns that directory (common exporter layout).
        """
        zip_path = Path(zip_path).expanduser().resolve(strict=False)
        if not zip_path.exists() or not zip_path.is_file():
            raise ValidationError(f"zip_path not found: {zip_path}")
        if zip_path.suffix.lower() != ".zip":
            raise ValidationError("Only .zip archives are supported")

        # Prefer the project's configured temp dir; fall back to OS temp if dependencies are missing.
        base_dir: Path
        try:
            from train_platform.core.config import settings

            settings.ensure_dirs()
            base_dir = (settings.temp_dir / "dataset_conversions").resolve(strict=False)
        except Exception:
            base_dir = (Path(tempfile.gettempdir()) / "dataset_conversions").resolve(strict=False)

        base_dir.mkdir(parents=True, exist_ok=True)
        out_dir = (base_dir / f"extract_{uuid.uuid4().hex}").resolve(strict=False)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
            # Try to reuse FileService's hardened extractor; if not available, use a minimal safe extractor.
            try:
                from train_platform.services.file_service import FileService

                FileService()._safe_extract_zip(zf, target_dir)
                return
            except Exception:
                pass

            seen: set[str] = set()
            for info in zf.infolist():
                name = str(info.filename or "")
                if not name:
                    continue
                rel = Path(name.replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValidationError("Unsafe zip content path.")

                key = rel.as_posix().lower()
                if key in seen:
                    raise ValidationError(f"Duplicate path in zip: {rel.as_posix()}")
                seen.add(key)

                dest = (target_dir / rel).resolve(strict=False)
                if target_dir not in dest.parents and dest != target_dir:
                    raise ValidationError("Unsafe zip extraction path.")

                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extract_zip(zf, out_dir)

            extracted = [p for p in out_dir.iterdir()]
            if len(extracted) == 1 and extracted[0].is_dir():
                return extracted[0]
            return out_dir
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise
