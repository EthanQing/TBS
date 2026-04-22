from __future__ import annotations

import os
import json
import concurrent.futures
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import yaml
from fastapi import UploadFile

from train_platform.core.config import settings
from train_platform.models.v3.enums import DatasetType
from train_platform.utils.dataset_yaml_utils import find_yolo_dataset_yaml
from train_platform.utils.image_exts import IMAGE_EXTS
from train_platform.utils.exceptions import ConflictError, ValidationError


class FileService:
    def upload_dataset(self, file: UploadFile, dataset_name: str, dataset_type: DatasetType) -> tuple[Path, dict]:
        dataset_name = (dataset_name or "").strip()
        if not dataset_name:
            raise ValidationError("Dataset name is required.")
        if any(sep in dataset_name for sep in ("/", "\\")) or dataset_name in (".", ".."):
            raise ValidationError("Invalid dataset name.")

        filename = file.filename or ""
        if not filename.endswith((".zip", ".tar.gz", ".tar", ".tgz")):
            raise ValidationError("Unsupported file format.")

        datasets_root = settings.datasets_dir.resolve()
        datasets_root.mkdir(parents=True, exist_ok=True)

        dataset_dir = datasets_root / dataset_name
        if dataset_dir.exists():
            raise ConflictError(f"Dataset directory '{dataset_name}' already exists.")

        temp_extract_dir = datasets_root / f"_tmp_extract_{dataset_name}"
        safe_upload_name = Path(filename).name or "upload"
        temp_file = datasets_root / f"_tmp_upload_{safe_upload_name}"

        try:
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(file.file, f, length=1024 * 1024)

            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.chmod(temp_extract_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

            if filename.endswith(".zip"):
                with zipfile.ZipFile(temp_file, "r") as zip_ref:
                    self._safe_extract_zip(zip_ref, temp_extract_dir)
            elif filename.endswith((".tar.gz", ".tgz")):
                with tarfile.open(temp_file, "r:gz") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)
            elif filename.endswith(".tar"):
                with tarfile.open(temp_file, "r") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)

            extracted = list(temp_extract_dir.iterdir())
            extracted_root = temp_extract_dir
            if len(extracted) == 1 and extracted[0].is_dir():
                extracted_root = extracted[0]

            source_root = extracted_root
            detected_format = "yolo"
            illegal_reason: str | None = None

            if dataset_type == DatasetType.DETECTION:
                info = self._detect_dataset_format(extracted_root, dataset_type)
                fmt = info.get("format")
                if fmt == "no_images":
                    raise ValidationError("No image files found in dataset directory")
                if fmt == "images_only":
                    raise ValidationError("No label files found")

                if fmt == "yolo":
                    yolo_root = info.get("yolo_root")
                    if yolo_root is not None:
                        source_root = yolo_root
                elif fmt == "labelme":
                    detected_format = "labelme"
                    illegal_reason = "labelme_json"
                elif fmt == "unknown_json":
                    detected_format = "unknown_json"
                    illegal_reason = "unsupported_json"

            # "Paste" to datasets/<dataset_name>.
            # Use move for speed; fall back to copy if needed.
            try:
                shutil.move(str(source_root), str(dataset_dir))
            except Exception:
                shutil.copytree(source_root, dataset_dir)

            if detected_format not in ("labelme", "unknown_json"):
                self._validate_dataset_structure(dataset_dir, dataset_type)

            self._safe_cleanup(temp_extract_dir, temp_file)
            return dataset_dir, {"format": detected_format, "illegal_reason": illegal_reason}
        except Exception:
            self._safe_cleanup(dataset_dir, temp_extract_dir, temp_file)
            raise

    def upload_dataset_into_existing(self, file: UploadFile, dataset_dir: Path, dataset_type: DatasetType) -> tuple[Path, dict]:
        dataset_dir = Path(dataset_dir)
        datasets_root = settings.datasets_dir.resolve()
        dataset_dir = dataset_dir.resolve()
        if dataset_dir == datasets_root or (datasets_root not in dataset_dir.parents and dataset_dir != datasets_root):
            raise ValidationError("Dataset directory must be under BASE_DATASETS_DIR")

        filename = file.filename or ""
        if not filename.endswith((".zip", ".tar.gz", ".tar", ".tgz")):
            raise ValidationError("Unsupported file format.")

        # Ensure target dir is empty (allow pre-created empty folder).
        if dataset_dir.exists():
            try:
                if any(dataset_dir.iterdir()):
                    raise ConflictError("Dataset directory is not empty.")
            except ConflictError:
                raise
            except Exception:
                pass
            # Remove empty dir so move/copy can create it.
            try:
                dataset_dir.rmdir()
            except Exception:
                pass

        temp_extract_dir = datasets_root / f"_tmp_extract_{dataset_dir.name}"
        safe_upload_name = Path(filename).name or "upload"
        temp_file = datasets_root / f"_tmp_upload_{dataset_dir.name}_{safe_upload_name}"

        try:
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(file.file, f, length=1024 * 1024)

            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.chmod(temp_extract_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

            if filename.endswith(".zip"):
                with zipfile.ZipFile(temp_file, "r") as zip_ref:
                    self._safe_extract_zip(zip_ref, temp_extract_dir)
            elif filename.endswith((".tar.gz", ".tgz")):
                with tarfile.open(temp_file, "r:gz") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)
            elif filename.endswith(".tar"):
                with tarfile.open(temp_file, "r") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)

            extracted = list(temp_extract_dir.iterdir())
            extracted_root = temp_extract_dir
            if len(extracted) == 1 and extracted[0].is_dir():
                extracted_root = extracted[0]

            source_root = extracted_root
            detected_format = "yolo"
            illegal_reason: str | None = None

            if dataset_type == DatasetType.DETECTION:
                info = self._detect_dataset_format(extracted_root, dataset_type)
                fmt = info.get("format")
                if fmt == "no_images":
                    raise ValidationError("No image files found in dataset directory")
                if fmt == "images_only":
                    raise ValidationError("No label files found")

                if fmt == "yolo":
                    yolo_root = info.get("yolo_root")
                    if yolo_root is not None:
                        source_root = yolo_root
                elif fmt == "labelme":
                    detected_format = "labelme"
                    illegal_reason = "labelme_json"
                elif fmt == "unknown_json":
                    detected_format = "unknown_json"
                    illegal_reason = "unsupported_json"

            # Move into datasets/<dataset_name>.
            try:
                shutil.move(str(source_root), str(dataset_dir))
            except Exception:
                shutil.copytree(source_root, dataset_dir)

            if detected_format not in ("labelme", "unknown_json"):
                self._validate_dataset_structure(dataset_dir, dataset_type)

            self._safe_cleanup(temp_extract_dir, temp_file)
            return dataset_dir, {"format": detected_format, "illegal_reason": illegal_reason}
        except Exception:
            # Best-effort cleanup; strict rollback for any files written.
            self._safe_cleanup(dataset_dir, temp_extract_dir, temp_file)
            raise

    def append_dataset_archive(self, file: UploadFile, dataset_dir: Path, dataset_type: DatasetType) -> tuple[Path, dict]:
        """
        Append contents from a ZIP archive to an existing (non-empty) dataset directory.
        New files are merged into the existing structure. Existing files are NOT overwritten.
        
        For DETECTION datasets:
        - Validates that new classnames.txt is compatible with existing data.yaml
        - If new classes are appended (prefix matches), updates data.yaml automatically
        - If classes are incompatible (reordered/modified), raises ValidationError
        
        Returns:
            (dataset_dir, info_dict) where info_dict contains:
                - added_classes: list of newly added class names
                - total_classes: total number of classes after merge
        """
        dataset_dir = Path(dataset_dir)
        datasets_root = settings.datasets_dir.resolve()
        dataset_dir = dataset_dir.resolve()
        if dataset_dir == datasets_root or (datasets_root not in dataset_dir.parents and dataset_dir != datasets_root):
            raise ValidationError("Dataset directory must be under BASE_DATASETS_DIR")

        filename = file.filename or ""
        if not filename.endswith((".zip", ".tar.gz", ".tar", ".tgz")):
            raise ValidationError("Unsupported file format.")

        # For append mode, directory should exist
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        temp_extract_dir = datasets_root / f"_tmp_extract_{dataset_dir.name}_append"
        safe_upload_name = Path(filename).name or "upload"
        temp_file = datasets_root / f"_tmp_upload_{dataset_dir.name}_{safe_upload_name}"

        info = {"added_classes": [], "total_classes": 0}

        try:
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(file.file, f, length=1024 * 1024)

            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.chmod(temp_extract_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

            if filename.endswith(".zip"):
                with zipfile.ZipFile(temp_file, "r") as zip_ref:
                    self._safe_extract_zip(zip_ref, temp_extract_dir)
            elif filename.endswith((".tar.gz", ".tgz")):
                with tarfile.open(temp_file, "r:gz") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)
            elif filename.endswith(".tar"):
                with tarfile.open(temp_file, "r") as tar_ref:
                    self._safe_extract_tar(tar_ref, temp_extract_dir)

            extracted = list(temp_extract_dir.iterdir())
            extracted_root = temp_extract_dir
            if len(extracted) == 1 and extracted[0].is_dir():
                extracted_root = extracted[0]

            source_root = extracted_root
            if dataset_type == DatasetType.DETECTION:
                yolo_root = self._find_yolo_export_root(extracted_root)
                if yolo_root is not None:
                    source_root = yolo_root

            # For DETECTION datasets, check class compatibility before merging
            if dataset_type == DatasetType.DETECTION:
                info = self._check_and_merge_classes(dataset_dir, source_root)

            # Merge files from source_root into dataset_dir (copy_tree with dirs_exist_ok)
            # Skip classnames.txt and similar files as they've been handled
            self._merge_directories(source_root, dataset_dir, skip_class_files=True)

            self._validate_dataset_structure(dataset_dir, dataset_type)

            self._safe_cleanup(temp_extract_dir, temp_file)
            return dataset_dir, info
        except Exception:
            self._safe_cleanup(temp_extract_dir, temp_file)
            raise

    def _check_and_merge_classes(self, dataset_dir: Path, source_root: Path) -> dict:
        """
        Check if new classnames.txt is compatible with existing data.yaml.
        If compatible (prefix matches), merge new classes into data.yaml.
        
        Returns dict with added_classes and total_classes.
        Raises ValidationError if classes are incompatible.
        """
        # Read existing classes from an existing YOLO dataset yaml (prefer data.yaml).
        existing_classes = []
        data_yaml_path = find_yolo_dataset_yaml(dataset_dir)
        if data_yaml_path is not None and data_yaml_path.exists():
            try:
                cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
                names = cfg.get("names", [])
                if isinstance(names, list):
                    existing_classes = [str(n) for n in names]
                elif isinstance(names, dict):
                    max_idx = max(int(k) for k in names.keys()) if names else -1
                    existing_classes = [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
            except Exception:
                existing_classes = []

        # Read new classes from uploaded archive's classnames.txt
        new_classes = self._read_class_names_txt(source_root)
        
        # If no classnames.txt in upload, allow merge (no class changes)
        if not new_classes:
            return {"added_classes": [], "total_classes": len(existing_classes)}

        # If no existing classes, just accept the new ones
        if not existing_classes:
            self._update_data_yaml_classes(dataset_dir, new_classes)
            return {"added_classes": new_classes, "total_classes": len(new_classes)}

        # Check compatibility: new classes must be a superset starting with existing classes
        existing_count = len(existing_classes)
        new_count = len(new_classes)

        if new_count < existing_count:
            raise ValidationError(
                f"类别不兼容：上传的压缩包包含 {new_count} 个类别，但数据集已有 {existing_count} 个类别。"
                f"无法减少类别数量。"
            )

        # Check that the first N classes match exactly
        for i in range(existing_count):
            if new_classes[i] != existing_classes[i]:
                raise ValidationError(
                    f"类别不兼容：第 {i+1} 个类别不匹配。"
                    f"现有: '{existing_classes[i]}', 上传: '{new_classes[i]}'。"
                    f"请确保上传的压缩包中 classnames.txt 的前 {existing_count} 个类别与数据集现有类别完全一致。"
                )

        # All existing classes match, extract newly added classes
        added_classes = new_classes[existing_count:]
        
        if added_classes:
            # Update data.yaml with new classes
            self._update_data_yaml_classes(dataset_dir, new_classes)

        return {"added_classes": added_classes, "total_classes": len(new_classes)}

    def _update_data_yaml_classes(self, dataset_dir: Path, class_names: list[str]) -> None:
        """Update data.yaml with new class names list."""
        data_yaml_path = dataset_dir / "data.yaml"
        cfg = {}
        
        if data_yaml_path.exists():
            try:
                cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8", errors="ignore")) or {}
            except Exception:
                cfg = {}
        
        if not isinstance(cfg, dict):
            cfg = {}

        # Preserve existing train/val paths
        cfg["names"] = class_names
        cfg["nc"] = len(class_names)
        
        # Ensure train/val paths exist
        if "train" not in cfg:
            cfg["train"] = "images"
        if "val" not in cfg:
            cfg["val"] = "images"

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def _merge_directories(self, src: Path, dst: Path, skip_class_files: bool = False) -> None:
        """
        Recursively merge src directory into dst directory.
        Files in src that already exist in dst are skipped (no overwrite).
        If skip_class_files is True, class name definition files are skipped.
        """
        # Files to skip if skip_class_files is True
        class_file_names = {"class_names.txt", "classes.txt", "obj.names", "names.txt", "classnames.txt"}
        
        if not src.is_dir():
            return
        for item in src.iterdir():
            # Skip class definition files if requested
            if skip_class_files and item.name.lower() in class_file_names:
                continue
                
            dest_item = dst / item.name
            if item.is_dir():
                dest_item.mkdir(parents=True, exist_ok=True)
                self._merge_directories(item, dest_item, skip_class_files=skip_class_files)
            else:
                if not dest_item.exists():
                    dest_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_item)




    def _safe_extract_zip(self, zip_ref: zipfile.ZipFile, target_dir: Path) -> None:
        """
        Safe ZIP extraction with basic parallelization for large archives.

        Notes:
        - We validate paths up-front to prevent path traversal.
        - We pre-create directories in a batch to reduce syscall overhead.
        - For large archives, we extract files in chunks using a ThreadPoolExecutor.
          Each worker opens its own ZipFile handle (ZipFile is not thread-safe).
        """
        infos = zip_ref.infolist()
        if not infos:
            return

        is_windows = os.name == "nt"

        # Pre-validate and detect duplicates (case-insensitive on Windows).
        seen: dict[str, str] = {}
        dir_rels: set[Path] = set()
        file_names: list[str] = []

        for info in infos:
            name = str(info.filename or "")
            if not name:
                continue

            rel = Path(name.replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValidationError("Unsafe zip content path.")

            key = rel.as_posix()
            if is_windows:
                key = key.lower()

            cur_type = "dir" if info.is_dir() else "file"
            prev_type = seen.get(key)
            # Allow duplicate directory entries (some zippers emit them), but reject any other collisions.
            if prev_type is not None and not (prev_type == "dir" and cur_type == "dir"):
                raise ConflictError(f"Duplicate path in zip: {rel.as_posix()}")
            seen[key] = cur_type

            dest = (target_dir / rel).resolve(strict=False)
            if target_dir not in dest.parents and dest != target_dir:
                raise ValidationError("Unsafe zip extraction path.")

            if info.is_dir():
                dir_rels.add(rel)
                continue

            if rel.parent and str(rel.parent) not in (".", ""):
                dir_rels.add(rel.parent)
            file_names.append(name)

        # Batch mkdir to reduce repeated directory creation work.
        for drel in sorted(dir_rels, key=lambda p: (len(p.parts), p.as_posix())):
            if not drel or str(drel) in (".", ""):
                continue
            (target_dir / drel).mkdir(parents=True, exist_ok=True)

        total = len(file_names)
        if total <= 0:
            return

        bufsize = int(os.getenv("ARCHIVE_COPY_BUFSIZE", str(1024 * 1024)))
        bufsize = max(16 * 1024, min(bufsize, 16 * 1024 * 1024))

        # Parallel extraction threshold and worker count.
        threshold = int(os.getenv("ZIP_EXTRACT_PARALLEL_THRESHOLD", "256"))
        workers = int(os.getenv("ZIP_EXTRACT_WORKERS", "8"))
        if workers <= 0:
            workers = min(8, (os.cpu_count() or 4))
        workers = max(1, workers)

        zip_path = getattr(zip_ref, "filename", None)
        zip_path_str: str | None = None
        try:
            if zip_path:
                zip_path_str = os.fspath(zip_path)
        except Exception:
            zip_path_str = None
        can_parallel = (
            workers > 1
            and total >= threshold
            and isinstance(zip_path_str, str)
            and zip_path_str
            and Path(zip_path_str).exists()
        )

        def _chunks(seq: list[str], n: int) -> list[list[str]]:
            if n <= 0:
                return [seq]
            return [seq[i : i + n] for i in range(0, len(seq), n)]

        if not can_parallel:
            for name in file_names:
                info = zip_ref.getinfo(name)
                rel = Path(str(info.filename or "").replace("\\", "/"))
                dest = (target_dir / rel).resolve(strict=False)
                if dest.exists():
                    raise ConflictError(f"Duplicate file path in zip: {rel.as_posix()}")
                with zip_ref.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=bufsize)
            return

        chunk_size = int(os.getenv("ZIP_EXTRACT_CHUNK_SIZE", "128"))
        chunk_size = max(16, min(chunk_size, 2048))

        def _extract_chunk(names: list[str]) -> int:
            assert isinstance(zip_path_str, str)  # for mypy
            with zipfile.ZipFile(zip_path_str, "r") as zf:
                for name in names:
                    info = zf.getinfo(name)
                    rel = Path(str(info.filename or "").replace("\\", "/"))
                    dest = (target_dir / rel).resolve(strict=False)
                    if dest.exists():
                        raise ConflictError(f"Duplicate file path in zip: {rel.as_posix()}")
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=bufsize)
            return len(names)

        chunks = _chunks(file_names, chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_extract_chunk, c) for c in chunks]
            for fut in concurrent.futures.as_completed(futures):
                # Propagate exceptions immediately.
                fut.result()

    def _safe_extract_tar(self, tar_ref: tarfile.TarFile, target_dir: Path) -> None:
        for member in tar_ref.getmembers():
            rel = Path(str(member.name or "").replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValidationError("Unsafe tar content path.")
            if member.issym() or member.islnk():
                raise ValidationError("Symlinks are not allowed in dataset archives.")

            dest = (target_dir / rel).resolve(strict=False)
            if target_dir not in dest.parents and dest != target_dir:
                raise ValidationError("Unsafe tar extraction path.")

            if member.isdir():
                if dest.exists() and not dest.is_dir():
                    raise ConflictError(f"Duplicate path in tar (file already exists): {rel.as_posix()}")
                dest.mkdir(parents=True, exist_ok=True)
                continue

            if dest.exists():
                # Prevent overwriting (including case-insensitive collisions on Windows).
                raise ConflictError(f"Duplicate file path in tar: {rel.as_posix()}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar_ref.extractfile(member)
            if src is None:
                continue
            with src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

    def _safe_cleanup(self, *paths: Path) -> None:
        for p in paths:
            try:
                if not p or not p.exists():
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    def _classify_json(self, path: Path) -> str:
        """
        Classify a json annotation file.
        Returns: "labelme" or "unknown".
        """
        try:
            if not path.exists() or not path.is_file():
                return "unknown"
            size = 0
            try:
                size = int(path.stat().st_size or 0)
            except Exception:
                size = 0

            # For large files, use a lightweight key scan.
            if size > 20 * 1024 * 1024:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(256 * 1024)
                    head_l = head.lower()
                    if '"shapes"' in head_l and (
                        '"imagewidth"' in head_l or '"imageheight"' in head_l or '"imagepath"' in head_l
                    ):
                        return "labelme"
                except Exception:
                    return "unknown"
                return "unknown"

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return "unknown"

            if "shapes" in data and (
                "imageWidth" in data
                or "imageHeight" in data
                or "imagePath" in data
                or "imagewidth" in data
                or "imageheight" in data
            ):
                return "labelme"
        except Exception:
            return "unknown"
        return "unknown"

    def _detect_dataset_format(self, root: Path, dataset_type: DatasetType) -> dict:
        """
        Detect dataset format for detection datasets.
        """
        result = {
            "format": "no_images",
            "yolo_root": None,
            "labelme_json": None,
        }
        if dataset_type != DatasetType.DETECTION:
            result["format"] = "yolo"
            return result
        if not root.exists() or not root.is_dir():
            return result

        # 1) YOLO (images + labels txt)
        yolo_root = self._find_yolo_export_root(root)
        if yolo_root is not None:
            result["format"] = "yolo"
            result["yolo_root"] = yolo_root
            return result

        # Scan for images + json files
        image_exts = IMAGE_EXTS
        has_images = False
        json_paths: list[Path] = []
        skip_dirs = {".versions", ".thumbnails", "__macosx"}
        max_depth = 4

        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root)
            if len(rel.parts) > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]
            for fname in filenames:
                p = cur_p / fname
                ext = p.suffix.lower()
                if ext in image_exts:
                    has_images = True
                elif ext == ".json":
                    json_paths.append(p)

        if not has_images:
            result["format"] = "no_images"
            return result

        if json_paths:
            labelme_json: Path | None = None
            unknown_json = False
            for p in json_paths:
                kind = self._classify_json(p)
                if kind == "labelme" and labelme_json is None:
                    labelme_json = p
                else:
                    unknown_json = True

            if labelme_json is not None:
                result["format"] = "labelme"
                result["labelme_json"] = labelme_json
                return result

            if unknown_json:
                result["format"] = "unknown_json"
                return result

        result["format"] = "images_only"
        return result

    def _validate_dataset_structure(self, dataset_dir: Path, dataset_type: DatasetType) -> None:
        if dataset_type != DatasetType.DETECTION:
            return

        image_exts = IMAGE_EXTS
        has_images = False
        has_labels = False
        has_json = False
        for root, _, files in os.walk(dataset_dir):
            root_p = Path(root)
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in image_exts:
                    has_images = True
                elif ext == ".txt":
                    if "labels" in {p.lower() for p in root_p.parts}:
                        has_labels = True
                elif ext == ".json":
                    has_json = True
            if has_images and (has_labels or has_json):
                break
        if not has_images:
            raise ValidationError("No image files found in dataset directory")
        if not (has_labels or has_json):
            raise ValidationError("No label files found")

        # Find or create YOLO data.yaml (do not write absolute 'path' field).
        yaml_files = list(dataset_dir.glob("*.yaml")) + list(dataset_dir.glob("*.yml"))
        if not yaml_files:
            self._create_yolo_data_yaml(dataset_dir, dataset_dir / "data.yaml")

    def _find_yolo_export_root(self, root: Path) -> Optional[Path]:
        """
        Try to locate the "real" YOLO export root inside an archive.

        Example export (labeling tools often nest this):
        rsExports/yolo/{images,labels,class_names.txt}
        """
        if not root.exists() or not root.is_dir():
            return None

        image_exts = IMAGE_EXTS

        best: tuple[int, int, Path] | None = None  # (image_count, depth, path)

        for cur, dirnames, _filenames in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root)
            depth = len(rel.parts)
            # Avoid scanning deep huge trees.
            if depth > 6:
                dirnames[:] = []
                continue

            names_l = {d.lower(): d for d in dirnames}
            if "images" not in names_l or "labels" not in names_l:
                continue

            images_dir = cur_p / names_l["images"]
            if not images_dir.exists() or not images_dir.is_dir():
                continue
            labels_dir = cur_p / names_l["labels"]
            if not labels_dir.exists() or not labels_dir.is_dir():
                continue

            label_count = 0
            try:
                for _r, _d, files in os.walk(labels_dir):
                    for fn in files:
                        if Path(fn).suffix.lower() == ".txt":
                            label_count += 1
                            break
                    if label_count > 0:
                        break
            except Exception:
                continue

            if label_count <= 0:
                continue

            image_count = 0
            # Count a few images to validate; do not fully traverse huge datasets.
            try:
                for _r, _d, files in os.walk(images_dir):
                    for fn in files:
                        if Path(fn).suffix.lower() in image_exts:
                            image_count += 1
                            if image_count >= 10:
                                break
                    if image_count >= 10:
                        break
            except Exception:
                continue

            if image_count <= 0:
                continue

            cand = (int(image_count), int(depth), cur_p)
            if best is None:
                best = cand
            else:
                # Prefer more images; tie-break on shallower depth.
                if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand

        return best[2] if best else None

    def _find_labelme_json_file(self, root: Path) -> Optional[Path]:
        """
        Scan for LabelMe-style JSON annotations (shapes + imageWidth/imageHeight).
        """
        if not root.exists():
            return None

        for cur, dirnames, filenames in os.walk(root):
            cur_p = Path(cur)
            if len(cur_p.relative_to(root).parts) > 4:
                dirnames[:] = []
                continue

            for fname in filenames:
                if not fname.lower().endswith(".json"):
                    continue
                p = cur_p / fname
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(2048)
                    if '"shapes"' in head and (
                        '"imageWidth"' in head
                        or '"imageHeight"' in head
                        or '"imagewidth"' in head
                        or '"imageheight"' in head
                    ):
                        return p
                except Exception:
                    continue

        return None

    def _read_class_names_txt(self, dataset_dir: Path) -> list[str]:
        """
        Read class names from common export files (one class per line).
        """
        candidates = ["class_names.txt", "classes.txt", "obj.names", "names.txt"]
        for name in candidates:
            p = dataset_dir / name
            if p.exists() and p.is_file():
                return self._read_lines(p)

        # Some exporters nest meta files; search a few levels but skip huge dirs.
        skip_dirs = {"images", "labels", ".versions"}
        for cur, dirnames, filenames in os.walk(dataset_dir):
            cur_p = Path(cur)
            rel = cur_p.relative_to(dataset_dir)
            depth = len(rel.parts)
            if depth > 3:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]
            for name in candidates:
                if name in filenames:
                    p = cur_p / name
                    if p.exists() and p.is_file():
                        return self._read_lines(p)
        return []

    def _read_lines(self, path: Path) -> list[str]:
        out: list[str] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                text = path.read_text(encoding="gbk", errors="ignore")
            except Exception:
                return []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out

    def _create_yolo_data_yaml(self, dataset_dir: Path, yaml_path: Path) -> None:
        train_path = None
        val_path = None

        possible_structures = [
            {"train": dataset_dir / "images" / "train", "val": dataset_dir / "images" / "val"},
            {"train": dataset_dir / "train" / "images", "val": dataset_dir / "val" / "images"},
            {"train": dataset_dir / "images", "val": dataset_dir / "images"},
        ]

        for s in possible_structures:
            if s["train"].exists():
                train_path = str(s["train"].relative_to(dataset_dir))
                break
        for s in possible_structures:
            if s["val"].exists():
                val_path = str(s["val"].relative_to(dataset_dir))
                break

        if not val_path and train_path:
            val_path = train_path

        # Prefer exporter-provided class names if present.
        class_names: list[str] = self._read_class_names_txt(dataset_dir)

        # Fallback: infer class ids from labels (names become class_0/class_1...).
        labels_dir = dataset_dir / "labels"
        if not class_names and labels_dir.exists():
            for label_file in labels_dir.rglob("*.txt"):
                try:
                    for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        class_id = int(line.split()[0])
                        while len(class_names) <= class_id:
                            class_names.append(f"class_{len(class_names)}")
                except Exception:
                    continue
        if not class_names:
            class_names = ["class_0"]

        cfg = {
            "train": train_path or "images",
            "val": val_path or "images",
            "nc": len(class_names),
            "names": class_names,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
