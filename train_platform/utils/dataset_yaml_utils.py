from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_DATASET_YAML_FILENAMES: tuple[str, ...] = (
    "data.yaml",
    "data.yml",
    "dataset.yaml",
    "dataset.yml",
)


def _safe_stem(name: str | None) -> str | None:
    """
    Best-effort conversion of a dataset name into a safe filename stem.
    """
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    # Avoid path traversal / separators; keep only the last path segment.
    s = Path(s.replace("\\", "/")).name
    if not s or s in (".", ".."):
        return None
    return s


def _load_yaml_dict(path: Path) -> dict:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        obj = {}
    return obj if isinstance(obj, dict) else {}


def _looks_like_yolo_dataset_yaml(cfg: Any) -> bool:
    """
    Heuristic to avoid picking random .yml files (e.g. docker-compose.yml) as dataset configs.
    """
    if not isinstance(cfg, dict):
        return False
    # Typical YOLO dataset keys
    if any(k in cfg for k in ("train", "val", "test")):
        return True
    if "names" in cfg or "nc" in cfg:
        return True
    if "path" in cfg and ("train" in cfg or "val" in cfg):
        return True
    return False


def find_yolo_dataset_yaml(dataset_dir: Path, *, dataset_name: str | None = None) -> Path | None:
    """
    Locate a YOLO-style dataset YAML inside `dataset_dir`.

    We prefer stable/expected names first ("data.yaml"), but many public datasets ship
    configs named after the dataset itself (e.g. "HomeObjects-3K.yaml").
    """
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        return None

    candidates: list[str] = list(_DEFAULT_DATASET_YAML_FILENAMES)
    stem = _safe_stem(dataset_name)
    if stem:
        candidates.extend([f"{stem}.yaml", f"{stem}.yml"])

    for fname in candidates:
        p = (root / fname).resolve(strict=False)
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue

    yaml_files = []
    try:
        yaml_files.extend(root.glob("*.yaml"))
        yaml_files.extend(root.glob("*.yml"))
    except Exception:
        yaml_files = []

    yaml_files = [p for p in yaml_files if p.exists() and p.is_file()]
    if not yaml_files:
        return None

    if len(yaml_files) == 1:
        return yaml_files[0].resolve(strict=False)

    # Content-based selection.
    best: Path | None = None
    for p in sorted(yaml_files, key=lambda x: x.name.lower()):
        cfg = _load_yaml_dict(p)
        # Strong signal: train+val present.
        if cfg.get("train") is not None and cfg.get("val") is not None:
            return p.resolve(strict=False)
        if best is None and _looks_like_yolo_dataset_yaml(cfg):
            best = p

    return best.resolve(strict=False) if best else None

