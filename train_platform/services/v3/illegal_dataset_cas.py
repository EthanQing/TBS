from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import yaml

from train_platform.core.config import settings
from train_platform.services.v3.dataset_common import ensure_safe_relative_path, resolve_storage_token
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 8 * 1024 * 1024
DATA_YAML_NAMES = {"data.yaml", "dataset.yaml", "data.yml", "dataset.yml"}
SKIP_LABEL_TXT_NAMES = {"classes.txt", "train.txt", "val.txt", "test.txt"}
SKIP_JSON_LABEL_DIRS = {"labels", ".versions", ".thumbnails", "__macosx"}


def illegal_cas_root() -> Path:
    return (settings.datasets_dir / "illegal" / ".cas").resolve(strict=False)


def illegal_versions_root(illegal_dataset_id: int) -> Path:
    return (settings.datasets_dir / "illegal" / ".versions" / str(int(illegal_dataset_id))).resolve(strict=False)


def illegal_manifest_path(illegal_dataset_id: int, version: int) -> Path:
    return illegal_versions_root(int(illegal_dataset_id)) / f"v{int(version)}.manifest.json"


def illegal_dataset_temp_root() -> Path:
    # Keep temporary upload/materialization trees under BASE_DATASETS_DIR so
    # CAS/source/workdir hardlinks remain on the same filesystem.
    path = (settings.datasets_dir / "illegal" / ".tmp").resolve(strict=False)
    path.mkdir(parents=True, exist_ok=True)
    return path


def illegal_dataset_file_url(illegal_dataset_id: int, version_id: int, rel_path: str | Path) -> str:
    rel = safe_manifest_rel(rel_path)
    return (
        f"/api/v3/illegal-datasets/{int(illegal_dataset_id)}"
        f"/versions/{int(version_id)}/files/{quote(rel, safe='/')}"
    )


def safe_manifest_rel(value: str | Path | None) -> str:
    rel = ensure_safe_relative_path(value).as_posix()
    if not rel or rel == ".":
        raise ValidationError("Invalid relative path")
    return rel


def _ensure_under_base(path: Path, base: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    base_resolved = base.resolve(strict=False)
    try:
        resolved.relative_to(base_resolved)
    except Exception as exc:
        raise ValidationError(f"Unsafe {label}") from exc
    return resolved


def _chmod_writable(path: str | os.PathLike[str]) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except Exception:
        pass


def remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, raw_path, _exc_info):
        _chmod_writable(raw_path)
        func(raw_path)

    shutil.rmtree(path, onerror=_onerror)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cas_path_for_hash(file_hash: str, *, require_exists: bool = True) -> Path:
    digest = str(file_hash or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValidationError("Invalid CAS hash")
    path = illegal_cas_root() / digest[:2] / digest
    if require_exists and (not path.exists() or not path.is_file()):
        raise NotFoundError(f"CAS file missing for hash {digest}")
    return path


def store_file_in_cas(path: Path) -> dict[str, Any]:
    src = Path(path).resolve(strict=False)
    if src.is_symlink():
        raise ValidationError("Symlinks are not supported in illegal datasets")
    if not src.exists() or not src.is_file():
        raise NotFoundError(f"Source file not found: {src}")
    try:
        st = src.stat()
        size = int(st.st_size)
        mtime = float(st.st_mtime)
    except Exception as exc:
        raise ValidationError(f"Cannot stat source file: {src}") from exc

    digest = hash_file(src)
    target = cas_path_for_hash(digest, require_exists=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise ValidationError(f"CAS path is not a file: {target}")
        try:
            target.chmod(0o444)
        except Exception:
            pass
        return {"hash": digest, "size": size, "mtime": mtime}
    try:
        os.link(src, target)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ValidationError(f"Failed to hardlink file into CAS: {src} -> {target}: {exc}") from exc
    try:
        target.chmod(0o444)
    except Exception:
        pass
    return {"hash": digest, "size": size, "mtime": mtime}


def mounted_file_entry(path: Path) -> dict[str, Any]:
    src = Path(path).resolve(strict=False)
    if not src.exists() or not src.is_file():
        raise NotFoundError(f"Mounted source file not found: {src}")
    try:
        st = src.stat()
    except Exception as exc:
        raise ValidationError(f"Cannot stat mounted source file: {src}") from exc
    return {
        "storage": "mounted",
        "source_path": str(src),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def iter_regular_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def scan_tree_to_cas_files(root: Path, *, base_files: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise NotFoundError("Illegal dataset source tree not found")
    files: dict[str, dict[str, Any]] = {
        safe_manifest_rel(rel): normalize_manifest_file_entry(entry)
        for rel, entry in (base_files or {}).items()
    }
    for path in sorted(iter_regular_files(root), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValidationError("Symlinks are not supported in illegal datasets")
        rel = safe_manifest_rel(path.relative_to(root).as_posix())
        files[rel] = store_file_in_cas(path)
    return files


def _entry_hash(entry: Mapping[str, Any] | None) -> str:
    return str((entry or {}).get("hash") or "").strip().lower()


def _entry_storage(entry: Mapping[str, Any] | None) -> str:
    storage = str((entry or {}).get("storage") or "cas").strip().lower()
    return "mounted" if storage == "mounted" else "cas"


def normalize_manifest_file_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    storage = _entry_storage(entry)
    normalized = {
        "storage": storage,
        "hash": _entry_hash(entry) if storage == "cas" else "",
        "size": int(entry.get("size") or entry.get("size_bytes") or 0),
        "mtime": float(entry.get("mtime") or 0.0),
    }
    if storage == "mounted":
        source_path = str(entry.get("source_path") or entry.get("path") or "").strip()
        if not source_path:
            raise ValidationError("Mounted manifest file entry is missing source_path")
        normalized["source_path"] = source_path
    return normalized


def _entry_identity(entry: Mapping[str, Any] | None) -> str:
    if _entry_storage(entry) == "mounted":
        return f"mounted:{str((entry or {}).get('source_path') or '')}:{int((entry or {}).get('size') or 0)}:{float((entry or {}).get('mtime') or 0.0)}"
    return f"cas:{_entry_hash(entry)}"


def build_manifest(
    *,
    dataset_id: int,
    version: int,
    parent_version_id: int | None,
    files: Mapping[str, Mapping[str, Any]],
    parent_files: Mapping[str, Mapping[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    normalized_files: dict[str, dict[str, Any]] = {}
    for rel, entry in files.items():
        rel_s = safe_manifest_rel(rel)
        normalized_files[rel_s] = normalize_manifest_file_entry(entry)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": int(dataset_id),
        "version": int(version),
        "parent_version_id": int(parent_version_id) if parent_version_id is not None else None,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "files": normalized_files,
        "stats": {},
    }
    manifest["stats"] = build_manifest_stats(manifest, parent_files=parent_files)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    dst = Path(path).resolve(strict=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f"{dst.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def load_manifest_path(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NotFoundError("Illegal dataset manifest not found") from exc
    except Exception as exc:
        raise ValidationError(f"Invalid illegal dataset manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("Invalid illegal dataset manifest")
    if int(data.get("schema_version") or 0) != MANIFEST_SCHEMA_VERSION:
        raise ValidationError("Unsupported illegal dataset manifest schema")
    files = data.get("files")
    if not isinstance(files, dict):
        raise ValidationError("Invalid illegal dataset manifest files")
    # Validate path keys and normalize entry scalars defensively.
    normalized: dict[str, dict[str, Any]] = {}
    for rel, entry in files.items():
        rel_s = safe_manifest_rel(rel)
        if not isinstance(entry, dict):
            raise ValidationError("Invalid illegal dataset manifest file entry")
        normalized[rel_s] = normalize_manifest_file_entry(entry)
    data["files"] = normalized
    if not isinstance(data.get("stats"), dict):
        data["stats"] = {}
    return data


def load_manifest_token(token: str | Path | None) -> dict[str, Any] | None:
    if not token:
        return None
    path = resolve_storage_token(str(token))
    return load_manifest_path(path)


def load_version_manifest(version: Any) -> dict[str, Any]:
    token = str(getattr(version, "manifest_path", "") or "").strip()
    if not token:
        raise NotFoundError("Illegal dataset version has no manifest")
    return load_manifest_token(token)


def manifest_files(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files") if isinstance(manifest, Mapping) else {}
    if not isinstance(files, dict):
        return {}
    return files  # type: ignore[return-value]


def manifest_entry(manifest: Mapping[str, Any], rel_path: str | Path, *, required: bool = True) -> dict[str, Any] | None:
    rel = safe_manifest_rel(rel_path)
    entry = manifest_files(manifest).get(rel)
    if entry is None and required:
        raise NotFoundError("File is not present in illegal dataset version manifest")
    return entry


def manifest_file_path(manifest: Mapping[str, Any], rel_path: str | Path, *, required: bool = True) -> Path:
    entry = manifest_entry(manifest, rel_path, required=True)
    assert entry is not None
    if _entry_storage(entry) == "mounted":
        source_path = Path(str(entry.get("source_path") or "")).expanduser().resolve(strict=False)
        if required and (not source_path.exists() or not source_path.is_file()):
            raise NotFoundError("Mounted source file is no longer available")
        return source_path
    return cas_path_for_hash(str(entry.get("hash") or ""), require_exists=required)


def manifest_cas_file_path(manifest: Mapping[str, Any], rel_path: str | Path, *, required: bool = True) -> Path:
    entry = manifest_entry(manifest, rel_path, required=True)
    assert entry is not None
    if _entry_storage(entry) != "cas":
        raise ValidationError("Manifest file is not stored in CAS")
    return cas_path_for_hash(str(entry.get("hash") or ""), require_exists=required)


def read_manifest_text(manifest: Mapping[str, Any], rel_path: str | Path) -> str:
    path = manifest_file_path(manifest, rel_path, required=True)
    return path.read_text(encoding="utf-8", errors="ignore")


def image_rel_paths_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(
        rel
        for rel in manifest_files(manifest)
        if Path(rel).suffix.lower() in IMAGE_EXTS
    )


def _find_manifest_file_by_name(manifest: Mapping[str, Any], names: set[str], *, root_first: bool = True) -> str | None:
    files = manifest_files(manifest)
    if root_first:
        for name in sorted(names):
            if name in files:
                return name
    matches = sorted(rel for rel in files if Path(rel).name.lower() in names)
    return matches[0] if matches else None


def read_data_yaml_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rel = _find_manifest_file_by_name(manifest, DATA_YAML_NAMES)
    if not rel:
        return {}
    try:
        data = yaml.safe_load(read_manifest_text(manifest, rel)) or {}
    except NotFoundError:
        raise
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_class_names_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    data = read_data_yaml_from_manifest(manifest)
    names = data.get("names")
    if isinstance(names, dict):
        out: list[str] = []
        def _sort_key(item):
            try:
                return (0, int(item[0]))
            except Exception:
                return (1, str(item[0]))

        for _, value in sorted(names.items(), key=_sort_key):
            out.append(str(value))
        return out
    if isinstance(names, list):
        return [str(x) for x in names]
    rel = _find_manifest_file_by_name(manifest, {"classes.txt"})
    if rel:
        try:
            return [line.strip() for line in read_manifest_text(manifest, rel).splitlines() if line.strip()]
        except Exception:
            return []
    return []


def manifest_image_size(manifest: Mapping[str, Any], image_rel_path: str | Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    path = manifest_file_path(manifest, image_rel_path, required=True)
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def guess_label_rel_path(image_rel_path: str | Path) -> str:
    rel = ensure_safe_relative_path(image_rel_path)
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt").as_posix()
    return rel.with_suffix(".txt").as_posix()


def _parse_manifest_yolo_boxes(
    lines: list[str],
    class_names: list[str],
    *,
    width: int | None,
    height: int | None,
    include_boxes: bool,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    boxes: list[dict[str, Any]] = []
    class_ids: set[int] = set()
    object_count = 0
    for line in lines:
        parts = [p for p in line.strip().split() if p]
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            continue
        object_count += 1
        class_ids.add(class_id)
        if not include_boxes:
            continue
        if width and height:
            x1 = max(0.0, (xc - w / 2.0) * width)
            y1 = max(0.0, (yc - h / 2.0) * height)
            x2 = min(float(width), (xc + w / 2.0) * width)
            y2 = min(float(height), (yc + h / 2.0) * height)
        else:
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
                "x1": float(round(x1, 4)),
                "y1": float(round(y1, 4)),
                "x2": float(round(x2, 4)),
                "y2": float(round(y2, 4)),
            }
        )
    return boxes, object_count, sorted(class_ids)


def read_yolo_boxes_from_manifest(
    manifest: Mapping[str, Any],
    image_rel_path: str | Path,
    class_names: list[str],
) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    rel = safe_manifest_rel(image_rel_path)
    width, height = manifest_image_size(manifest, rel)
    label_rel = guess_label_rel_path(rel)
    if label_rel not in manifest_files(manifest):
        return width, height, []
    try:
        lines = read_manifest_text(manifest, label_rel).splitlines()
    except NotFoundError:
        raise
    except Exception:
        return width, height, []
    boxes, _object_count, _class_ids = _parse_manifest_yolo_boxes(
        lines,
        class_names,
        width=width,
        height=height,
        include_boxes=True,
    )
    return width, height, boxes


def read_yolo_box_summary_from_manifest(
    manifest: Mapping[str, Any],
    image_rel_path: str | Path,
    class_names: list[str],
) -> tuple[int | None, int | None, int, list[int]]:
    rel = safe_manifest_rel(image_rel_path)
    width, height = manifest_image_size(manifest, rel)
    label_rel = guess_label_rel_path(rel)
    if label_rel not in manifest_files(manifest):
        return width, height, 0, []
    try:
        lines = read_manifest_text(manifest, label_rel).splitlines()
    except NotFoundError:
        raise
    except Exception:
        return width, height, 0, []
    _boxes, object_count, class_ids = _parse_manifest_yolo_boxes(
        lines,
        class_names,
        width=width,
        height=height,
        include_boxes=False,
    )
    return width, height, object_count, class_ids


def scan_yolo_annotation_summary_from_manifest(manifest: Mapping[str, Any]) -> tuple[int, set[int]]:
    annotation_count = 0
    class_ids: set[int] = set()
    for rel in sorted(manifest_files(manifest)):
        path = Path(rel)
        if path.suffix.lower() != ".txt" or path.name.lower() in SKIP_LABEL_TXT_NAMES:
            continue
        try:
            lines = read_manifest_text(manifest, rel).splitlines()
        except NotFoundError:
            raise
        except Exception:
            continue
        for line in lines:
            parts = [p for p in line.strip().split() if p]
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            annotation_count += 1
            class_ids.add(class_id)
    return annotation_count, class_ids


def extract_json_labels_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    labels: set[str] = set()
    for rel in sorted(manifest_files(manifest)):
        path = Path(rel)
        if path.suffix.lower() != ".json":
            continue
        if any(part.lower() in SKIP_JSON_LABEL_DIRS for part in path.parts[:-1]):
            continue
        json_path = manifest_file_path(manifest, rel, required=True)
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            try:
                with json_path.open("r", encoding="gbk", errors="ignore") as f:
                    data = json.load(f)
            except Exception:
                continue
        shapes: list[Any] = []
        if isinstance(data, dict) and isinstance(data.get("shapes"), list):
            shapes = data["shapes"]
        elif isinstance(data, list):
            shapes = data
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label") or "").strip()
            if label:
                labels.add(label)
    return sorted(labels)


def build_manifest_stats(
    manifest: Mapping[str, Any],
    *,
    parent_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    files = manifest_files(manifest)
    total_files = len(files)
    total_size = sum(int((entry or {}).get("size") or (entry or {}).get("size_bytes") or 0) for entry in files.values())
    image_count = len(image_rel_paths_from_manifest(manifest))
    target_count, used_class_ids = scan_yolo_annotation_summary_from_manifest(manifest)
    declared_class_count = len(read_class_names_from_manifest(manifest))
    class_count = declared_class_count if declared_class_count > 0 else len(used_class_ids)
    parent_files = parent_files or {}
    unchanged_files = sum(
        1
        for rel, entry in files.items()
        if rel in parent_files and _entry_identity(parent_files.get(rel)) == _entry_identity(entry)
    )
    new_files = total_files - unchanged_files
    size_mb = round(float(total_size or 0) / (1024 * 1024), 2)
    return {
        "total_files": int(total_files),
        "total_size_bytes": int(total_size),
        "total_size_mb": size_mb,
        "size_mb": size_mb,
        "dataset_size_mb": size_mb,
        "image_count": int(image_count),
        "total_images": int(image_count),
        "num_images": int(image_count),
        "target_count": int(target_count),
        "annotations_count": int(target_count),
        "total_targets": int(target_count),
        "object_count": int(target_count),
        "total_objects": int(target_count),
        "class_count": int(class_count),
        "num_classes": int(class_count),
        "declared_class_count": int(declared_class_count),
        "used_class_count": int(len(used_class_ids)),
        "new_files": int(new_files),
        "unchanged_files": int(unchanged_files),
    }


def manifest_stats_to_dataset_statistics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict(manifest.get("stats") or {})
    files = manifest_files(manifest)
    total_files = int(stats.get("total_files") or len(files))
    total_size = int(
        stats.get("total_size_bytes")
        or sum(int((entry or {}).get("size") or (entry or {}).get("size_bytes") or 0) for entry in files.values())
    )
    image_count = int(stats.get("image_count") or stats.get("total_images") or len(image_rel_paths_from_manifest(manifest)))
    target_count = int(stats.get("target_count") or stats.get("annotations_count") or 0)
    class_count = int(stats.get("class_count") or stats.get("num_classes") or 0)
    declared = int(stats.get("declared_class_count") or 0)
    used = int(stats.get("used_class_count") or 0)
    size_mb = round(float(total_size or 0) / (1024 * 1024), 2)
    return {
        **stats,
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": float(stats.get("total_size_mb") or size_mb),
        "size_mb": float(stats.get("size_mb") or size_mb),
        "dataset_size_mb": float(stats.get("dataset_size_mb") or size_mb),
        "total_images": image_count,
        "num_images": image_count,
        "image_count": image_count,
        "annotations_count": target_count,
        "target_count": target_count,
        "total_targets": target_count,
        "object_count": target_count,
        "total_objects": target_count,
        "num_classes": class_count,
        "class_count": class_count,
        "declared_class_count": declared,
        "used_class_count": used,
    }


def hardlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as exc:
        raise ValidationError(f"Failed to hardlink dataset file: {src} -> {dst}: {exc}") from exc


def materialize_manifest_to_dir(manifest: Mapping[str, Any], dst: Path, *, replace: bool = True) -> None:
    dst = Path(dst).resolve(strict=False)
    if replace:
        remove_tree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for rel, entry in sorted(manifest_files(manifest).items()):
        rel_s = safe_manifest_rel(rel)
        src = manifest_file_path(manifest, rel_s, required=True)
        target = _ensure_under_base(dst / rel_s, dst, "manifest materialization path")
        hardlink_file(src, target)


def replace_dir_from_manifest(manifest: Mapping[str, Any], dst: Path) -> None:
    dst = Path(dst).resolve(strict=False)
    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(parent), prefix=f".{dst.name}.staging."))
    try:
        materialize_manifest_to_dir(manifest, staging, replace=True)
        remove_tree(dst)
        staging.replace(dst)
    except Exception:
        remove_tree(staging)
        raise


