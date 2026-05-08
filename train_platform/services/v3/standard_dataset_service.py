from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.models.v3.enums import DatasetSplit, DatasetType
from train_platform.models.v3.project import Project
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent, StandardDatasetImage
from train_platform.models.v3.training_run import TrainingRun
from train_platform.repositories.v3.standard_dataset_repo import StandardDatasetRepository
from train_platform.services.v3.dataset_common import (
    build_annotations_payload,
    build_file_listing,
    build_statistics,
    build_view_payload,
    clear_directory,
    commit_refresh,
    copy_tree,
    count_tree,
    detect_split_from_relpath,
    iter_image_files,
    resolve_storage_token,
    static_dataset_url,
    dataset_thumbnail_url,
    to_storage_token,
    unpack_uploaded_archive,
    utcnow,
)
from train_platform.services.v3.file_service import FileService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


class StandardDatasetService:
    def __init__(self) -> None:
        self.repo = StandardDatasetRepository()

    def _next_dataset_id(self, db: Session) -> int:
        current_max = db.query(func.max(StandardDataset.standard_dataset_id)).scalar()
        start = int(settings.standard_dataset_id_start)
        if current_max is None:
            return start
        return max(start, int(current_max) + 1)

    def _root_path(self, dataset: StandardDataset) -> Path:
        return resolve_storage_token(dataset.storage_path)

    def _resolve_uploaded_yolo_root(self, extracted_root: Path) -> Path | None:
        root = Path(extracted_root)
        if (root / "images").exists() and (root / "labels").exists():
            return root
        if any((root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")):
            return root
        return FileService()._find_yolo_export_root(root)

    def _ensure_name_available(self, db: Session, name: str, *, exclude_id: int | None = None) -> None:
        row = self.repo.get_by_name(db, str(name).strip())
        if row and (exclude_id is None or int(row.standard_dataset_id) != int(exclude_id)):
            raise ConflictError(f"Standard dataset '{name}' already exists")

    def _index_images(self, db: Session, dataset: StandardDataset) -> None:
        root = self._root_path(dataset)
        db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)).delete()
        for image_path in iter_image_files(root):
            rel = image_path.relative_to(root).as_posix()
            db.add(
                StandardDatasetImage(
                    standard_dataset_id=int(dataset.standard_dataset_id),
                    path=rel,
                    split=detect_split_from_relpath(rel),
                )
            )
        db.flush()

    def _add_event(
        self,
        db: Session,
        dataset_id: int,
        event_type: str,
        *,
        message: str | None = None,
        created_by: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        db.add(
            StandardDatasetEvent(
                standard_dataset_id=int(dataset_id),
                event_type=str(event_type),
                message=message,
                created_by=created_by,
                data=data,
            )
        )
        db.flush()

    def _normalize_split_ratios(
        self,
        train_ratio: float,
        val_ratio: float | None,
        test_ratio: float | None,
    ) -> tuple[float, float, float]:
        try:
            tr = float(train_ratio)
        except Exception as exc:
            raise ValidationError("train_ratio must be a number") from exc
        if tr <= 0 or tr > 1:
            raise ValidationError("train_ratio must be between 0 and 1")

        vr = val_ratio
        ter = test_ratio
        if vr is None and ter is None:
            remainder = 1.0 - tr
            if remainder < 0:
                raise ValidationError("train_ratio must be less than or equal to 1")
            vr = remainder * 0.7
            ter = remainder * 0.3
        elif vr is None:
            try:
                ter = float(ter)
            except Exception as exc:
                raise ValidationError("test_ratio must be a number") from exc
            vr = 1.0 - tr - float(ter)
        elif ter is None:
            try:
                vr = float(vr)
            except Exception as exc:
                raise ValidationError("val_ratio must be a number") from exc
            ter = 1.0 - tr - float(vr)
        else:
            try:
                vr = float(vr)
                ter = float(ter)
            except Exception as exc:
                raise ValidationError("val_ratio / test_ratio must be numbers") from exc

        if vr < 0 or vr >= 1:
            raise ValidationError("val_ratio must be between 0 and 1")
        if ter < 0 or ter >= 1:
            raise ValidationError("test_ratio must be between 0 and 1")
        if abs((tr + vr + ter) - 1.0) > 1e-6:
            raise ValidationError("train_ratio + val_ratio + test_ratio must equal 1")
        return float(tr), float(vr), float(ter)

    def _split_summary(self, db: Session, dataset_id: int) -> dict[str, Any]:
        base_q = db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(dataset_id))
        total_images = int(base_q.count())
        train_count = int(base_q.filter(StandardDatasetImage.split == DatasetSplit.TRAIN).count())
        val_count = int(base_q.filter(StandardDatasetImage.split == DatasetSplit.VAL).count())
        test_count = int(base_q.filter(StandardDatasetImage.split == DatasetSplit.TEST).count())

        latest_split_event = (
            db.query(StandardDatasetEvent)
            .filter(
                StandardDatasetEvent.standard_dataset_id == int(dataset_id),
                StandardDatasetEvent.event_type == "split_dataset",
            )
            .order_by(StandardDatasetEvent.created_at.desc(), StandardDatasetEvent.event_id.desc())
            .first()
        )
        event_data = latest_split_event.data if latest_split_event and isinstance(latest_split_event.data, dict) else {}
        return {
            "total_images": total_images,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "train_ratio": round((train_count / total_images), 6) if total_images else 0.0,
            "val_ratio": round((val_count / total_images), 6) if total_images else 0.0,
            "test_ratio": round((test_count / total_images), 6) if total_images else 0.0,
            "seed": event_data.get("seed"),
            "shuffle": event_data.get("shuffle"),
        }

    def _export_split_files_and_update_yaml(self, db: Session, dataset: StandardDataset) -> dict[str, Any]:
        dataset_root = self._root_path(dataset)
        dataset_root.mkdir(parents=True, exist_ok=True)

        train_file = "train.txt"
        val_file = "val.txt"
        test_file = "test.txt"
        train_path = (dataset_root / train_file).resolve(strict=False)
        val_path = (dataset_root / val_file).resolve(strict=False)
        test_path = (dataset_root / test_file).resolve(strict=False)

        def _write_list(out_path: Path, split_value: DatasetSplit) -> int:
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            count = 0
            try:
                with tmp_path.open("w", encoding="utf-8") as f:
                    q = (
                        db.query(StandardDatasetImage.path)
                        .filter(
                            StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id),
                            StandardDatasetImage.split == split_value,
                        )
                        .order_by(StandardDatasetImage.image_id)
                        .yield_per(1000)
                    )
                    for row in q:
                        rel = str(row[0] or "").strip().replace("\\", "/").lstrip("/")
                        if not rel:
                            continue
                        abs_path = (dataset_root / rel).resolve(strict=False)
                        if abs_path != dataset_root and dataset_root not in abs_path.parents:
                            continue
                        if not abs_path.exists():
                            continue
                        f.write(abs_path.as_posix() + "\n")
                        count += 1
                tmp_path.replace(out_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            return int(count)

        train_count = _write_list(train_path, DatasetSplit.TRAIN)
        val_count = _write_list(val_path, DatasetSplit.VAL)
        test_count = _write_list(test_path, DatasetSplit.TEST)

        data_yaml = None
        for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
            candidate = dataset_root / name
            if candidate.exists():
                data_yaml = candidate
                break
        if data_yaml is None:
            data_yaml = dataset_root / "data.yaml"
            FileService()._create_yolo_data_yaml(dataset_root, data_yaml)

        try:
            cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["train"] = train_file
        cfg["val"] = val_file
        if test_count > 0:
            cfg["test"] = test_file
        else:
            cfg.pop("test", None)
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        return {
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "yaml_path": data_yaml.name,
        }

    def _build_dataset_statistics(self, db: Session, dataset: StandardDataset) -> dict[str, Any]:
        image_count = (
            db.query(StandardDatasetImage)
            .filter(StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id))
            .count()
        )
        return build_statistics(self._root_path(dataset), image_count=image_count)

    def _dataset_with_statistics(self, db: Session, dataset: StandardDataset) -> dict[str, Any]:
        return {
            "standard_dataset_id": int(dataset.standard_dataset_id),
            "name": dataset.name,
            "dataset_type": dataset.dataset_type,
            "format": dataset.format,
            "storage_path": dataset.storage_path,
            "description": dataset.description,
            "source_type": dataset.source_type,
            "publish_config": dataset.publish_config,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "statistics": self._build_dataset_statistics(db, dataset),
        }

    def list_datasets(self, db: Session, *, skip: int = 0, limit: int = 100, format: str | None = None) -> list[dict[str, Any]]:
        q = db.query(StandardDataset)
        if format:
            q = q.filter(StandardDataset.format == str(format))
        rows = q.order_by(StandardDataset.updated_at.desc()).offset(skip).limit(limit).all()
        return [self._dataset_with_statistics(db, row) for row in rows]

    def create_dataset(self, db: Session, *, obj: dict) -> StandardDataset:
        name = str(obj.get("name") or "").strip()
        if not name:
            raise ValidationError("name is required")
        fmt = str(obj.get("format") or "yolo").strip().lower() or "yolo"
        if fmt != "yolo":
            raise ValidationError("Only YOLO dataset format is supported")
        self._ensure_name_available(db, name)
        row = StandardDataset(
            standard_dataset_id=self._next_dataset_id(db),
            name=name,
            dataset_type=obj["dataset_type"],
            format=fmt,
            storage_path="pending/standard",
            description=obj.get("description"),
            source_type=obj.get("source_type"),
            publish_config=obj.get("publish_config"),
        )
        db.add(row)
        db.flush()
        row.storage_path = f"standard/{int(row.standard_dataset_id)}"
        root = self._root_path(row)
        root.mkdir(parents=True, exist_ok=True)
        self._add_event(db, int(row.standard_dataset_id), "created", message="Standard dataset created")
        db.commit()
        db.refresh(row)
        return row

    def get_dataset(self, db: Session, standard_dataset_id: int) -> StandardDataset:
        row = self.repo.get(db, int(standard_dataset_id))
        if not row:
            raise NotFoundError("Standard dataset not found")
        return row

    def update_dataset(self, db: Session, standard_dataset_id: int, *, patch: dict) -> StandardDataset:
        row = self.get_dataset(db, standard_dataset_id)
        if "name" in patch and patch["name"] is not None:
            new_name = str(patch["name"]).strip()
            if not new_name:
                raise ValidationError("name cannot be empty")
            self._ensure_name_available(db, new_name, exclude_id=int(row.standard_dataset_id))
            row.name = new_name
        if "description" in patch:
            row.description = patch["description"]
        db.commit()
        db.refresh(row)
        return row

    def delete_dataset(self, db: Session, standard_dataset_id: int, *, delete_files: bool = False, force: bool = False) -> None:
        row = self.get_dataset(db, standard_dataset_id)
        projects = db.query(Project).filter(Project.standard_dataset_id == int(row.standard_dataset_id)).all()
        runs = db.query(TrainingRun).filter(TrainingRun.standard_dataset_id == int(row.standard_dataset_id)).all()
        if (projects or runs) and not force:
            raise ConflictError("Standard dataset is still referenced by projects or training runs")
        for run in runs:
            db.delete(run)
        for project in projects:
            db.delete(project)
        root = self._root_path(row)
        db.delete(row)
        db.commit()
        if delete_files:
            shutil.rmtree(root, ignore_errors=True)

    def upload_archive(self, db: Session, standard_dataset_id: int, upload, *, created_by: str | None = None) -> StandardDataset:
        row = self.get_dataset(db, standard_dataset_id)
        root = self._root_path(row)
        existing_files, _ = count_tree(root)
        if existing_files > 0:
            raise ConflictError("Standard dataset content is immutable after upload")
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=settings.temp_dir))
        try:
            extracted_root = unpack_uploaded_archive(upload, temp_dir)
            yolo_root = self._resolve_uploaded_yolo_root(extracted_root)
            if yolo_root is None:
                raise ValidationError("Standard dataset upload only supports YOLO format")
            copy_tree(yolo_root, root)
            if not any((root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")):
                FileService()._create_yolo_data_yaml(root, root / "data.yaml")
            self._index_images(db, row)
            self._add_event(
                db,
                int(row.standard_dataset_id),
                "uploaded",
                message="Standard dataset archive uploaded",
                created_by=created_by,
                data={"filename": str(getattr(upload, 'filename', '') or '')},
            )
            db.commit()
            db.refresh(row)
            return row
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def split_dataset(
        self,
        db: Session,
        standard_dataset_id: int,
        *,
        train_ratio: float = 0.9,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int | None = None,
        shuffle: bool = True,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        if row.dataset_type != DatasetType.DETECTION:
            raise ValidationError("split_dataset is only supported for detection standard datasets")

        train_ratio, val_ratio, test_ratio = self._normalize_split_ratios(train_ratio, val_ratio, test_ratio)
        q = db.query(StandardDatasetImage.image_id).filter(
            StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id)
        )
        if not overwrite:
            q = q.filter(StandardDatasetImage.split.is_(None))

        ids = [int(item[0]) for item in q.order_by(StandardDatasetImage.image_id).all()]
        total = len(ids)
        if total <= 0:
            raise ValidationError("No images available for split")

        if shuffle:
            rng = random.Random(seed) if seed is not None else random.Random()
            rng.shuffle(ids)

        train_count = int(total * float(train_ratio))
        val_count = int(total * float(val_ratio))
        test_count = total - train_count - val_count
        if test_count < 0:
            deficit = -int(test_count)
            if val_count >= deficit:
                val_count -= deficit
                test_count = 0
            elif train_count >= (deficit - val_count):
                train_count -= (deficit - val_count)
                val_count = 0
                test_count = 0
            else:
                raise ValidationError("Invalid split ratios for dataset size")

        train_ids = ids[:train_count]
        val_ids = ids[train_count: train_count + val_count]
        test_ids = ids[train_count + val_count:]

        if overwrite:
            db.query(StandardDatasetImage).filter(
                StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id)
            ).update(
                {StandardDatasetImage.split: None, StandardDatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )

        def _chunked(seq: list[int], size: int = 1000):
            for idx in range(0, len(seq), size):
                yield seq[idx: idx + size]

        for chunk in _chunked(train_ids):
            db.query(StandardDatasetImage).filter(StandardDatasetImage.image_id.in_(chunk)).update(
                {StandardDatasetImage.split: DatasetSplit.TRAIN, StandardDatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )
        for chunk in _chunked(val_ids):
            db.query(StandardDatasetImage).filter(StandardDatasetImage.image_id.in_(chunk)).update(
                {StandardDatasetImage.split: DatasetSplit.VAL, StandardDatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )
        for chunk in _chunked(test_ids):
            db.query(StandardDatasetImage).filter(StandardDatasetImage.image_id.in_(chunk)).update(
                {StandardDatasetImage.split: DatasetSplit.TEST, StandardDatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )

        export_meta = self._export_split_files_and_update_yaml(db, row)
        summary = self._split_summary(db, int(row.standard_dataset_id))
        self._add_event(
            db,
            int(row.standard_dataset_id),
            "split_dataset",
            message="Standard dataset split updated",
            data={
                **summary,
                **export_meta,
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                "test_ratio": float(test_ratio),
                "seed": int(seed) if seed is not None else None,
                "shuffle": bool(shuffle),
                "overwrite": bool(overwrite),
            },
        )
        db.commit()
        return self._split_summary(db, int(row.standard_dataset_id))

    def materialize_from_source_tree(
        self,
        db: Session,
        *,
        name: str,
        dataset_type,
        source_root: Path,
        description: str | None = None,
        source_type: str | None = None,
        publish_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> StandardDataset:
        row = self.create_dataset(
            db,
            obj={
                "name": name,
                "dataset_type": dataset_type,
                "format": "yolo",
                "description": description,
                "source_type": source_type,
                "publish_config": publish_config,
            },
        )
        root = self._root_path(row)
        copy_tree(source_root, root)
        if not any((root / name).exists() for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")):
            FileService()._create_yolo_data_yaml(root, root / "data.yaml")
        self._index_images(db, row)
        self._add_event(
            db,
            int(row.standard_dataset_id),
            "published",
            message="Standard dataset materialized from source tree",
            created_by=created_by,
            data={"source_type": source_type},
        )
        db.commit()
        db.refresh(row)
        return row

    def list_events(self, db: Session, standard_dataset_id: int, *, skip: int = 0, limit: int = 100) -> list[StandardDatasetEvent]:
        self.get_dataset(db, standard_dataset_id)
        return (
            db.query(StandardDatasetEvent)
            .filter(StandardDatasetEvent.standard_dataset_id == int(standard_dataset_id))
            .order_by(StandardDatasetEvent.created_at.desc(), StandardDatasetEvent.event_id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_detail(self, db: Session, standard_dataset_id: int, *, events_limit: int = 20) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        return {
            "dataset": row,
            "statistics": self._build_dataset_statistics(db, row),
            "events": self.list_events(db, int(row.standard_dataset_id), skip=0, limit=events_limit),
        }

    def get_statistics(self, db: Session, standard_dataset_id: int) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        return self._build_dataset_statistics(db, row)

    def get_view(self, db: Session, standard_dataset_id: int, *, page: int = 1, page_size: int = 50, class_id: int | None = None) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        images = (
            db.query(StandardDatasetImage)
            .filter(StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id))
            .order_by(StandardDatasetImage.path.asc())
            .all()
        )
        payload = build_view_payload(
            self._root_path(row),
            row.storage_path,
            images,
            page=page,
            page_size=page_size,
            thumbnail_url_builder=lambda rel_path: dataset_thumbnail_url(
                "standard",
                int(row.standard_dataset_id),
                rel_path,
                size=320,
            ),
        )
        if class_id is not None:
            payload["items"] = [item for item in payload["items"] if int(class_id) in item.get("classes", [])]
            total_items = len(payload["items"])
            payload["meta"]["total_items"] = total_items
            payload["meta"]["total_pages"] = 1
        return payload

    def get_image_annotations(self, db: Session, standard_dataset_id: int, *, image_path: str) -> dict[str, Any]:
        row = self.get_dataset(db, standard_dataset_id)
        return build_annotations_payload(self._root_path(row), row.storage_path, image_path)

    def list_files(self, db: Session, standard_dataset_id: int, *, page: int = 1, page_size: int = 100) -> tuple[list[dict[str, Any]], int]:
        row = self.get_dataset(db, standard_dataset_id)
        return build_file_listing(self._root_path(row), row.storage_path, page=page, page_size=page_size)

    def get_split_result(
        self,
        db: Session,
        standard_dataset_id: int,
        *,
        split: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[StandardDatasetImage], dict[str, Any], int]:
        row = self.get_dataset(db, standard_dataset_id)
        base_q = db.query(StandardDatasetImage).filter(
            StandardDatasetImage.standard_dataset_id == int(row.standard_dataset_id)
        )
        split_norm = str(split or "").strip().lower()
        if split_norm:
            if split_norm == "train":
                base_q = base_q.filter(StandardDatasetImage.split == DatasetSplit.TRAIN)
            elif split_norm == "val":
                base_q = base_q.filter(StandardDatasetImage.split == DatasetSplit.VAL)
            elif split_norm == "test":
                base_q = base_q.filter(StandardDatasetImage.split == DatasetSplit.TEST)
            elif split_norm in ("none", "null", "unassigned", "unsplit"):
                base_q = base_q.filter(StandardDatasetImage.split.is_(None))
            else:
                raise ValidationError("split must be one of: train, val, test, unassigned")

        total = int(base_q.count())
        items = (
            base_q.order_by(StandardDatasetImage.image_id)
            .offset(max(0, int(skip)))
            .limit(max(0, int(limit)))
            .all()
        )
        return items, self._split_summary(db, int(row.standard_dataset_id)), total
