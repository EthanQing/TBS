from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from train_platform.core.config import settings
from train_platform.models.v3.enums import DatasetSplit
from train_platform.services.v3.dataset_common import (
    detect_split_from_relpath,
    image_size,
    iter_image_files,
    read_class_names,
)
from train_platform.services.v3.illegal_dataset_publish_service import bbox_to_yolo, parse_annotations
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS


MOUNTED_MANIFEST_NAME = ".mounted_manifest.json"
_SKIP_DIRS = {"labels", ".versions", ".thumbnails", "__macosx", ".git"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rel(value: str | Path | None) -> Path:
    rel = Path(str(value or "").strip().replace("\\", "/"))
    if not str(rel) or rel.is_absolute() or ".." in rel.parts:
        raise ValidationError("Invalid relative path")
    return rel


def _ensure_under(path: Path, base: Path, label: str) -> Path:
    resolved = Path(path).resolve(strict=False)
    base_resolved = Path(base).resolve(strict=False)
    try:
        resolved.relative_to(base_resolved)
    except Exception as exc:
        raise ValidationError(f"Unsafe {label}") from exc
    return resolved


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        try:
            path.rmdir()
            return
        except OSError:
            shutil.rmtree(path, ignore_errors=True)
            return
    path.unlink(missing_ok=True)


def create_directory_link(source: Path, target: Path) -> str:
    src = Path(source).resolve(strict=False)
    if not src.exists() or not src.is_dir():
        raise NotFoundError("Mounted source directory not found")
    dst = Path(target).resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(dst)
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
        return "symlink"
    except OSError as symlink_exc:
        if os.name != "nt":
            raise ValidationError(f"Failed to create directory symlink: {symlink_exc}") from symlink_exc
        try:
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            raise ValidationError(f"Failed to create Windows junction: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or str(symlink_exc)).strip()
            raise ValidationError(f"Failed to create Windows junction: {detail}")
        return "junction"


def load_mounted_manifest(root: Path) -> dict[str, Any] | None:
    path = Path(root) / MOUNTED_MANIFEST_NAME
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_mounted_file(root: Path, rel_path: str | Path) -> Path | None:
    manifest = load_mounted_manifest(root)
    if not manifest:
        return None
    rel = _safe_rel(rel_path).as_posix()
    image_prefix = str(manifest.get("image_rel_prefix") or "images").strip().strip("/\\")
    source_image_root = str(manifest.get("source_image_root") or "").strip()
    if not source_image_root:
        return None
    if rel == image_prefix:
        suffix = ""
    elif rel.startswith(f"{image_prefix}/"):
        suffix = rel[len(image_prefix) + 1 :]
    else:
        return None
    src_root = Path(source_image_root).resolve(strict=False)
    from train_platform.services.v3.dataset_import_service import DatasetImportService

    allowed_roots = DatasetImportService().allowed_roots()
    if not any(_is_relative_to(src_root, allowed.resolve(strict=False)) for allowed in allowed_roots):
        raise ValidationError("Mounted source root is not allowed")
    src = (src_root / suffix).resolve(strict=False)
    _ensure_under(src, src_root, "mounted source file path")
    if not src.exists() or not src.is_file():
        return None
    return src


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def resolve_dataset_file(root: Path, rel_path: str | Path) -> Path:
    mounted = resolve_mounted_file(root, rel_path)
    if mounted is not None:
        return mounted
    base = Path(root).resolve(strict=False)
    rel = _safe_rel(rel_path)
    path = (base / rel).resolve(strict=False)
    _ensure_under(path, base, "dataset file path")
    if not path.exists() or not path.is_file():
        raise NotFoundError("Dataset file not found")
    return path


def _iter_regular_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for cur, dirnames, filenames in os.walk(root):
        cur_path = Path(cur)
        rel = cur_path.relative_to(root)
        if rel.parts and rel.parts[0].lower() in _SKIP_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for name in filenames:
            out.append(cur_path / name)
    return out


def _pair_key(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[0].lower() in {"image", "images", "annotation", "annotations", "json", "labels"}:
        parts = parts[1:]
    return "/".join(parts).lower()


def collect_image_json_pairs(source_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    root = Path(source_root).resolve(strict=False)
    image_by_key: dict[str, Path] = {}
    json_by_key: dict[str, Path] = {}
    for path in _iter_regular_files(root):
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            image_by_key.setdefault(_pair_key(root, path), path)
        elif ext == ".json":
            json_by_key.setdefault(_pair_key(root, path), path)
    keys = sorted(set(image_by_key) & set(json_by_key))
    warnings: list[str] = []
    if len(image_by_key) > len(keys):
        warnings.append(f"Unmatched images: {len(image_by_key) - len(keys)}")
    if len(json_by_key) > len(keys):
        warnings.append(f"Unmatched json: {len(json_by_key) - len(keys)}")
    return [(image_by_key[key], json_by_key[key]) for key in keys], warnings


def choose_image_link(source_root: Path) -> tuple[Path, str]:
    source_root = Path(source_root).resolve(strict=False)
    images_dir = source_root / "images"
    if images_dir.exists() and images_dir.is_dir():
        return images_dir.resolve(strict=False), "images"
    return source_root, "images/source"


def image_rel_for_source(source_root: Path, source_image_root: Path, image_path: Path, image_rel_prefix: str) -> str:
    rel = Path(image_path).resolve(strict=False).relative_to(Path(source_image_root).resolve(strict=False)).as_posix()
    return f"{image_rel_prefix.strip('/')}/{rel}".strip("/")


def label_rel_for_image(image_rel: str | Path) -> str:
    rel = Path(str(image_rel).replace("\\", "/"))
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt").as_posix()
    return rel.with_suffix(".txt").as_posix()


def _write_class_files(root: Path, class_names: list[str]) -> list[str]:
    names = [str(name).strip() for name in class_names if str(name).strip()]
    if not names:
        names = ["class_0"]
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def _infer_class_names_from_labels(labels_root: Path) -> list[str]:
    class_names: list[str] = []
    if not labels_root.exists():
        return class_names
    for label_file in sorted(labels_root.rglob("*.txt")):
        if label_file.name.lower() in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
            continue
        try:
            lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = [p for p in line.split() if p]
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            while len(class_names) <= class_id:
                class_names.append(f"class_{len(class_names)}")
    return class_names


def write_yolo_yaml(root: Path, class_names: list[str], image_rels: list[str]) -> None:
    root = Path(root)
    names = _write_class_files(root, class_names)
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rel in sorted(image_rels):
        split = detect_split_from_relpath(rel)
        key = split.value if isinstance(split, DatasetSplit) else "train"
        if key not in buckets:
            key = "train"
        buckets[key].append(rel)
    if not buckets["train"] and (buckets["val"] or buckets["test"]):
        buckets["train"] = buckets["val"] or buckets["test"]
    for split_name, rels in buckets.items():
        if not rels and split_name != "val":
            continue
        if split_name == "val" and not rels:
            rels = buckets["train"]
        (root / f"{split_name}.txt").write_text("\n".join(rels) + ("\n" if rels else ""), encoding="utf-8")
    payload: dict[str, Any] = {
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(names),
        "names": names,
    }
    if buckets["test"]:
        payload["test"] = "test.txt"
    with (root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _copy_source_label(source_root: Path, image_path: Path, image_rel: str, target_root: Path) -> bool:
    source_root = Path(source_root).resolve(strict=False)
    rel_to_source = Path(image_path).resolve(strict=False).relative_to(source_root)
    candidates = [
        source_root / label_rel_for_image(rel_to_source),
        source_root / label_rel_for_image(image_rel.replace("images/source/", "", 1)),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            dst = target_root / label_rel_for_image(image_rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dst)
            return True
    return False


def link_yolo_source_tree(target_root: Path, source_root: Path) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = Path(source_root).resolve(strict=False)
    source_image_root, image_rel_prefix = choose_image_link(source_root)
    link_type = create_directory_link(source_image_root, target_root / image_rel_prefix)
    image_rels: list[str] = []
    copied_labels = 0
    for image_path in iter_image_files(source_image_root):
        image_rel = image_rel_for_source(source_root, source_image_root, image_path, image_rel_prefix)
        image_rels.append(image_rel)
        if _copy_source_label(source_root, image_path, image_rel, target_root):
            copied_labels += 1
        else:
            label_path = target_root / label_rel_for_image(image_rel)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")
    class_names = read_class_names(source_root) or _infer_class_names_from_labels(target_root / "labels")
    write_yolo_yaml(target_root, class_names, image_rels)
    manifest = {
        "schema_version": 1,
        "source_type": "mounted_dir_link",
        "format": "yolo",
        "source_root": str(source_root),
        "source_image_root": str(source_image_root),
        "image_rel_prefix": image_rel_prefix,
        "link_type": link_type,
        "created_at": _utcnow_iso(),
        "image_count": len(image_rels),
        "image_paths": sorted(image_rels),
        "label_count": copied_labels,
    }
    _write_json_atomic(target_root / MOUNTED_MANIFEST_NAME, manifest)
    return manifest


def link_json_source_tree(target_root: Path, source_root: Path) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = Path(source_root).resolve(strict=False)
    source_image_root, image_rel_prefix = choose_image_link(source_root)
    link_type = create_directory_link(source_image_root, target_root / image_rel_prefix)
    pairs, warnings = collect_image_json_pairs(source_root)
    if not pairs:
        raise ValidationError("No image/json pairs found in mounted directory")

    label_map: dict[str, int] = {}
    image_rels: list[str] = []
    skipped: list[str] = []
    object_count = 0
    base_cfg = {
        "label_map": label_map,
        "label_strategy": "full",
        "label_separator": "%",
        "min_probability": 0.0,
        "skip_hidden": True,
        "skip_outside": True,
    }
    for image_path, json_path in pairs:
        try:
            image_rel = image_rel_for_source(source_root, source_image_root, image_path, image_rel_prefix)
        except Exception:
            skipped.append(f"{image_path.name}: image is outside linked image root")
            continue
        width, height = image_size(image_path)
        if not width or not height:
            skipped.append(f"{image_path.name}: cannot read image size")
            continue
        try:
            bboxes, label_map = parse_annotations({**base_cfg, "label_map": label_map, "annotation_path": str(json_path)})
        except Exception as exc:
            skipped.append(f"{json_path.name}: {exc}")
            continue
        lines: list[str] = []
        for bbox in bboxes:
            bbox.x_min = max(0.0, min(float(width), float(bbox.x_min)))
            bbox.y_min = max(0.0, min(float(height), float(bbox.y_min)))
            bbox.x_max = max(0.0, min(float(width), float(bbox.x_max)))
            bbox.y_max = max(0.0, min(float(height), float(bbox.y_max)))
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            lines.append(bbox_to_yolo(bbox, int(width), int(height)))
        label_path = target_root / label_rel_for_image(image_rel)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        image_rels.append(image_rel)
        object_count += len(lines)
    if not image_rels:
        detail = "; ".join(skipped[:10])
        raise ValidationError(f"No valid image/json pairs imported. {detail}")
    class_names = [name for name, _cid in sorted(label_map.items(), key=lambda item: item[1])]
    write_yolo_yaml(target_root, class_names, image_rels)
    manifest = {
        "schema_version": 1,
        "source_type": "mounted_dir_link",
        "format": "json",
        "source_root": str(source_root),
        "source_image_root": str(source_image_root),
        "image_rel_prefix": image_rel_prefix,
        "link_type": link_type,
        "created_at": _utcnow_iso(),
        "image_count": len(image_rels),
        "image_paths": sorted(image_rels),
        "json_count": len(pairs),
        "object_count": object_count,
        "warnings": warnings + skipped[:50],
    }
    _write_json_atomic(target_root / MOUNTED_MANIFEST_NAME, manifest)
    return manifest


def link_source_tree(target_root: Path, source_root: Path, *, prefer_yolo: bool = True) -> dict[str, Any]:
    target_root = Path(target_root).resolve(strict=False)
    source_root = Path(source_root).resolve(strict=False)
    if not source_root.exists() or not source_root.is_dir():
        raise NotFoundError("Mounted source directory not found")
    target_root.mkdir(parents=True, exist_ok=True)
    has_yolo = (source_root / "labels").exists() or any((source_root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"))
    if prefer_yolo and has_yolo:
        return link_yolo_source_tree(target_root, source_root)
    return link_json_source_tree(target_root, source_root)
