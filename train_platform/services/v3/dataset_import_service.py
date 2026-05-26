from __future__ import annotations

import os
import json
import string
import uuid
from pathlib import Path
from typing import Any

from train_platform.core.config import settings
from train_platform.schemas.v3.dataset_imports import (
    DatasetImportEntriesOut,
    DatasetImportEntryOut,
    DatasetImportFsEntriesOut,
    DatasetImportFsEntryOut,
    DatasetImportInspectOut,
    DatasetImportRootCreate,
    DatasetImportRootDeleteOut,
    DatasetImportRootOut,
)
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.utils.image_exts import IMAGE_EXTS


class DatasetImportService:
    _DATA_YAML_NAMES = {"data.yaml", "dataset.yaml", "data.yml", "dataset.yml"}
    _SKIP_DIRS = {".git", "__macosx", ".thumbnails", ".versions"}
    _ROOT_STORE_NAME = "dataset_import_roots.json"

    def _root_store_path(self) -> Path:
        return settings.home_dir / self._ROOT_STORE_NAME

    def _load_user_roots(self) -> list[dict[str, str]]:
        path = self._root_store_path()
        if not path.exists() or not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or []
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        out: list[dict[str, str]] = []
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
            out.append({"root_id": root_id, "path": raw_path, "label": label})
        return out

    def _save_user_roots(self, roots: list[dict[str, str]]) -> None:
        path = self._root_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "root_id": str(item.get("root_id") or "").strip(),
                "path": str(item.get("path") or "").strip(),
                "label": str(item.get("label") or "").strip(),
            }
            for item in roots
            if str(item.get("root_id") or "").strip() and str(item.get("path") or "").strip()
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _root_out(self, *, root_id: str, path: Path, label: str = "", editable: bool = False) -> DatasetImportRootOut:
        exists = path.exists() and path.is_dir()
        readable = False
        if exists:
            try:
                next(path.iterdir(), None)
                readable = True
            except StopIteration:
                readable = True
            except Exception:
                readable = False
        display_label = label.strip() if label else f"{root_id}: {path}"
        return DatasetImportRootOut(
            root_id=root_id,
            path=str(path),
            label=display_label,
            exists=bool(exists),
            readable=bool(readable),
            editable=bool(editable),
        )

    def roots(self) -> list[DatasetImportRootOut]:
        out: list[DatasetImportRootOut] = []
        for idx, root in enumerate(settings.dataset_import_roots):
            root_id = "default" if idx == 0 else f"root_{idx}"
            out.append(self._root_out(root_id=root_id, path=root, editable=False))
        static_paths = {str(Path(item.path).resolve(strict=False)).lower() for item in out}
        seen_ids = {item.root_id for item in out}
        for item in self._load_user_roots():
            raw_id = str(item.get("root_id") or "").strip()
            root_id = raw_id if raw_id and raw_id not in seen_ids else f"user_{uuid.uuid4().hex[:10]}"
            try:
                root = Path(str(item.get("path") or "")).expanduser().resolve(strict=False)
            except Exception:
                continue
            key = str(root).lower()
            if key in static_paths:
                continue
            seen_ids.add(root_id)
            out.append(self._root_out(root_id=root_id, path=root, label=str(item.get("label") or ""), editable=True))
        return out

    def allowed_roots(self) -> tuple[Path, ...]:
        return tuple(Path(item.path).resolve(strict=False) for item in self.roots())

    def add_root(self, payload: DatasetImportRootCreate) -> DatasetImportRootOut:
        raw_path = str(payload.path or "").strip()
        if not raw_path:
            raise ValidationError("Directory path is required")
        try:
            root = Path(raw_path).expanduser().resolve(strict=False)
        except Exception as exc:
            raise ValidationError("Invalid directory path") from exc
        if not root.exists() or not root.is_dir():
            raise NotFoundError("Directory not found")
        try:
            next(root.iterdir(), None)
        except StopIteration:
            pass
        except Exception as exc:
            raise ValidationError("Directory is not readable") from exc

        for existing in self.roots():
            if Path(existing.path).resolve(strict=False) == root:
                return existing

        roots = self._load_user_roots()
        root_id = f"user_{uuid.uuid4().hex[:10]}"
        roots.append({"root_id": root_id, "path": str(root), "label": str(payload.label or "").strip()})
        self._save_user_roots(roots)
        return self._root_out(root_id=root_id, path=root, label=str(payload.label or ""), editable=True)

    def delete_root(self, root_id: str) -> DatasetImportRootDeleteOut:
        wanted = str(root_id or "").strip()
        if not wanted.startswith("user_"):
            raise ValidationError("Only user-added import roots can be removed")
        roots = self._load_user_roots()
        kept = [item for item in roots if str(item.get("root_id") or "").strip() != wanted]
        if len(kept) == len(roots):
            raise NotFoundError("Dataset import root not found")
        self._save_user_roots(kept)
        return DatasetImportRootDeleteOut(root_id=wanted, deleted=True)

    def _filesystem_start_entries(self) -> DatasetImportFsEntriesOut:
        entries: list[DatasetImportFsEntryOut] = []
        candidates: list[Path] = []
        if os.name == "nt":
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    candidates.append(drive)
        else:
            candidates.extend([Path("/"), Path("/data"), Path("/mnt"), Path("/media"), Path.home()])
        candidates.extend([settings.home_dir, settings.imports_dir])
        seen: set[str] = set()
        for candidate in candidates:
            try:
                path = candidate.expanduser().resolve(strict=False)
            except Exception:
                continue
            key = str(path).lower()
            if key in seen or not path.exists() or not path.is_dir():
                continue
            seen.add(key)
            readable = True
            try:
                next(path.iterdir(), None)
            except StopIteration:
                pass
            except Exception:
                readable = False
            entries.append(
                DatasetImportFsEntryOut(
                    name=str(path),
                    path=str(path),
                    is_dir=True,
                    readable=readable,
                )
            )
        return DatasetImportFsEntriesOut(path="", parent_path=None, entries=entries)

    def browse_filesystem(self, path: str | None = None) -> DatasetImportFsEntriesOut:
        raw = str(path or "").strip()
        if not raw:
            return self._filesystem_start_entries()
        try:
            current = Path(raw).expanduser().resolve(strict=False)
        except Exception as exc:
            raise ValidationError("Invalid directory path") from exc
        if not current.exists() or not current.is_dir():
            raise NotFoundError("Directory not found")
        entries: list[DatasetImportFsEntryOut] = []
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as exc:
            raise ValidationError("Directory is not readable") from exc
        for child in children:
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue
            readable = True
            try:
                next(child.iterdir(), None)
            except StopIteration:
                pass
            except Exception:
                readable = False
            entries.append(
                DatasetImportFsEntryOut(
                    name=child.name,
                    path=str(child.resolve(strict=False)),
                    is_dir=True,
                    readable=readable,
                )
            )
        parent_path = None
        parent = current.parent.resolve(strict=False)
        if parent != current:
            parent_path = str(parent)
        return DatasetImportFsEntriesOut(path=str(current), parent_path=parent_path, entries=entries)

    def _root_by_id(self, root_id: str | None) -> tuple[str, Path, DatasetImportRootOut]:
        roots = self.roots()
        wanted = str(root_id or "default").strip() or "default"
        for item in roots:
            if item.root_id == wanted:
                return item.root_id, Path(item.path).resolve(strict=False), item
        raise NotFoundError("Dataset import root not found")

    def resolve_path(self, root_id: str | None, rel_path: str | None) -> tuple[str, Path, Path, DatasetImportRootOut]:
        root_key, root, root_out = self._root_by_id(root_id)
        if not root.exists() or not root.is_dir():
            raise NotFoundError("Dataset import root is not available")
        raw = str(rel_path or "").strip().replace("\\", "/").strip("/")
        rel = Path(raw) if raw else Path("")
        if rel.is_absolute() or ".." in rel.parts:
            raise ValidationError("Invalid import path")
        resolved = (root / rel).resolve(strict=False)
        try:
            resolved.relative_to(root.resolve(strict=False))
        except Exception as exc:
            raise ValidationError("Import path must be inside an allowed import root") from exc
        return root_key, root, resolved, root_out

    def _quick_counts(self, path: Path, *, max_items: int = 3000) -> dict[str, Any]:
        image_count = 0
        json_count = 0
        label_count = 0
        has_data_yaml = False
        scanned = 0
        if not path.exists() or not path.is_dir():
            return {
                "image_count": 0,
                "json_count": 0,
                "label_count": 0,
                "has_data_yaml": False,
                "truncated": False,
            }
        for cur, dirnames, filenames in os.walk(path):
            cur_path = Path(cur)
            rel = cur_path.relative_to(path)
            if rel.parts and rel.parts[0].lower() in self._SKIP_DIRS:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in self._SKIP_DIRS]
            for name in filenames:
                scanned += 1
                lower = name.lower()
                ext = Path(name).suffix.lower()
                if lower in self._DATA_YAML_NAMES:
                    has_data_yaml = True
                if ext in IMAGE_EXTS:
                    image_count += 1
                elif ext == ".json":
                    json_count += 1
                elif ext == ".txt" and lower not in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
                    label_count += 1
                if scanned >= max_items:
                    return {
                        "image_count": image_count,
                        "json_count": json_count,
                        "label_count": label_count,
                        "has_data_yaml": has_data_yaml,
                        "truncated": True,
                    }
        return {
            "image_count": image_count,
            "json_count": json_count,
            "label_count": label_count,
            "has_data_yaml": has_data_yaml,
            "truncated": False,
        }

    def _format_from_counts(self, counts: dict[str, Any]) -> str:
        if counts.get("image_count", 0) > 0 and (counts.get("label_count", 0) > 0 or counts.get("has_data_yaml")):
            return "yolo"
        if counts.get("image_count", 0) > 0 and counts.get("json_count", 0) > 0:
            return "json"
        if counts.get("image_count", 0) > 0:
            return "images"
        return "unknown"

    def list_entries(self, *, root_id: str = "default", path: str = "") -> DatasetImportEntriesOut:
        root_key, root, current, root_out = self.resolve_path(root_id, path)
        if not current.exists() or not current.is_dir():
            raise NotFoundError("Import directory not found")
        entries: list[DatasetImportEntryOut] = []
        for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            try:
                rel = child.resolve(strict=False).relative_to(root).as_posix()
            except Exception:
                continue
            counts = self._quick_counts(child, max_items=800)
            is_candidate = counts["image_count"] > 0 and (counts["json_count"] > 0 or counts["label_count"] > 0 or counts["has_data_yaml"])
            entries.append(
                DatasetImportEntryOut(
                    name=child.name,
                    path=rel,
                    is_dir=True,
                    is_dataset_candidate=bool(is_candidate),
                    image_count=int(counts["image_count"]),
                    json_count=int(counts["json_count"]),
                    label_count=int(counts["label_count"]),
                    has_data_yaml=bool(counts["has_data_yaml"]),
                )
            )
        current_rel = current.relative_to(root).as_posix() if current != root else ""
        parent_path = None
        if current != root:
            parent = current.parent.resolve(strict=False)
            parent_path = parent.relative_to(root).as_posix() if parent != root else ""
        return DatasetImportEntriesOut(root=root_out, path=current_rel, parent_path=parent_path, entries=entries)

    def inspect(self, *, root_id: str = "default", path: str = "") -> DatasetImportInspectOut:
        root_key, root, current, _root_out = self.resolve_path(root_id, path)
        counts = self._quick_counts(current, max_items=1_000_000)
        warnings: list[str] = []
        if counts.get("truncated"):
            warnings.append("目录较大，检查结果为快速扫描统计")
        if counts.get("image_count", 0) <= 0:
            warnings.append("未发现图片文件")
        if counts.get("image_count", 0) > 0 and counts.get("json_count", 0) <= 0 and counts.get("label_count", 0) <= 0 and not counts.get("has_data_yaml"):
            warnings.append("未发现 JSON 或 YOLO 标注")
        rel = current.relative_to(root).as_posix() if current != root else ""
        return DatasetImportInspectOut(
            root_id=root_key,
            path=rel,
            resolved_path=str(current),
            exists=current.exists(),
            is_dir=current.is_dir(),
            format=self._format_from_counts(counts),
            image_count=int(counts["image_count"]),
            json_count=int(counts["json_count"]),
            label_count=int(counts["label_count"]),
            has_data_yaml=bool(counts["has_data_yaml"]),
            warnings=warnings,
        )
