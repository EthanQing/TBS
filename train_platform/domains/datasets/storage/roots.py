from __future__ import annotations

import json
import uuid
from pathlib import Path

from train_platform.core.config import settings
from train_platform.platform.filesystem.atomic import atomic_write_json


ROOT_STORE_NAME = "dataset_import_roots.json"


def root_store_path() -> Path:
    return settings.home_dir / ROOT_STORE_NAME


def load_user_import_roots() -> list[dict[str, str]]:
    path = root_store_path()
    if not path.exists() or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or []
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    roots: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, str):
            raw_path = item
            root_id = f"user_{uuid.uuid4().hex[:10]}"
            label = ""
        elif isinstance(item, dict):
            raw_path = str(item.get("path") or "").strip()
            root_id = str(item.get("root_id") or "").strip() or f"user_{uuid.uuid4().hex[:10]}"
            label = str(item.get("label") or "").strip()
        else:
            continue
        if not raw_path:
            continue
        if not root_id.startswith("user_"):
            root_id = f"user_{root_id}"
        roots.append({"root_id": root_id, "path": raw_path, "label": label})
    return roots


def save_user_import_roots(roots: list[dict[str, str]]) -> None:
    payload = [
        {
            "root_id": str(item.get("root_id") or "").strip(),
            "path": str(item.get("path") or "").strip(),
            "label": str(item.get("label") or "").strip(),
        }
        for item in roots
        if str(item.get("root_id") or "").strip() and str(item.get("path") or "").strip()
    ]
    atomic_write_json(root_store_path(), payload, sort_keys=False)


def allowed_import_roots() -> tuple[Path, ...]:
    roots: list[Path] = list(settings.dataset_import_roots)
    seen = {str(root.resolve(strict=False)).lower() for root in roots}
    for item in load_user_import_roots():
        try:
            root = Path(item["path"]).expanduser().resolve(strict=False)
        except Exception:
            continue
        key = str(root).lower()
        if key in seen:
            continue
        roots.append(root)
        seen.add(key)
    return tuple(root.resolve(strict=False) for root in roots)
