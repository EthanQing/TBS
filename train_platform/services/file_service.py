from __future__ import annotations

import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import yaml
from fastapi import UploadFile

from train_platform.core.config import settings
from train_platform.models.enums import DatasetType
from train_platform.utils.exceptions import ConflictError, ValidationError


class FileService:
    def upload_dataset(self, file: UploadFile, dataset_name: str, dataset_type: DatasetType) -> Path:
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
                shutil.copyfileobj(file.file, f)

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

            # "Paste" to datasets/<dataset_name>.
            # Use move for speed; fall back to copy if needed.
            try:
                shutil.move(str(source_root), str(dataset_dir))
            except Exception:
                shutil.copytree(source_root, dataset_dir)

            self._validate_dataset_structure(dataset_dir, dataset_type)

            self._safe_cleanup(temp_extract_dir, temp_file)
            return dataset_dir
        except Exception:
            self._safe_cleanup(dataset_dir, temp_extract_dir, temp_file)
            raise

    def upload_dataset_into_existing(self, file: UploadFile, dataset_dir: Path, dataset_type: DatasetType) -> Path:
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
                shutil.copyfileobj(file.file, f)

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

            # Move into datasets/<dataset_name>.
            try:
                shutil.move(str(source_root), str(dataset_dir))
            except Exception:
                shutil.copytree(source_root, dataset_dir)

            self._validate_dataset_structure(dataset_dir, dataset_type)

            self._safe_cleanup(temp_extract_dir, temp_file)
            return dataset_dir
        except Exception:
            # Best-effort cleanup; keep an empty dir so dataset record remains usable.
            self._safe_cleanup(temp_extract_dir, temp_file)
            try:
                if not dataset_dir.exists():
                    dataset_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            raise


    def _safe_extract_zip(self, zip_ref: zipfile.ZipFile, target_dir: Path) -> None:
        for info in zip_ref.infolist():
            rel = Path(str(info.filename or "").replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValidationError("Unsafe zip content path.")

            dest = (target_dir / rel).resolve(strict=False)
            if target_dir not in dest.parents and dest != target_dir:
                raise ValidationError("Unsafe zip extraction path.")

            if info.is_dir():
                if dest.exists() and not dest.is_dir():
                    raise ConflictError(f"Duplicate path in zip (file already exists): {rel.as_posix()}")
                dest.mkdir(parents=True, exist_ok=True)
                continue

            if dest.exists():
                # Prevent overwriting (including case-insensitive collisions on Windows).
                raise ConflictError(f"Duplicate file path in zip: {rel.as_posix()}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

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
                shutil.copyfileobj(src, dst)

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

    def _validate_dataset_structure(self, dataset_dir: Path, dataset_type: DatasetType) -> None:
        if dataset_type != DatasetType.DETECTION:
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        has_images = False
        for root, _, files in os.walk(dataset_dir):
            if any(Path(f).suffix.lower() in image_exts for f in files):
                has_images = True
                break
        if not has_images:
            raise ValidationError("No image files found in dataset directory")

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

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

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
