from __future__ import annotations

import random
from typing import Any

import yaml
from sqlalchemy import func
from sqlalchemy.orm import Session

from train_platform.domains.datasets import yolo
from train_platform.domains.datasets.storage.mounted import resolve_dataset_file
from train_platform.domains.datasets.storage.paths import resolve_storage_token
from train_platform.models.v3.enums import DatasetSplit, DatasetType
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetEvent, StandardDatasetImage
from train_platform.utils.exceptions import ValidationError

from .events import add_event
from .service import StandardDatasetService


def normalize_split_ratios(
    train_ratio: float,
    val_ratio: float | None,
    test_ratio: float | None,
) -> tuple[float, float, float]:
    try:
        train = float(train_ratio)
    except Exception as exc:
        raise ValidationError("train_ratio must be a number") from exc
    if train <= 0 or train > 1:
        raise ValidationError("train_ratio must be between 0 and 1")
    val = val_ratio
    test = test_ratio
    if val is None and test is None:
        remainder = 1.0 - train
        if remainder < 0:
            raise ValidationError("train_ratio must be less than or equal to 1")
        val, test = remainder * 0.7, remainder * 0.3
    elif val is None:
        try:
            test = float(test)
        except Exception as exc:
            raise ValidationError("test_ratio must be a number") from exc
        val = 1.0 - train - float(test)
    elif test is None:
        try:
            val = float(val)
        except Exception as exc:
            raise ValidationError("val_ratio must be a number") from exc
        test = 1.0 - train - float(val)
    else:
        try:
            val, test = float(val), float(test)
        except Exception as exc:
            raise ValidationError("val_ratio / test_ratio must be numbers") from exc
    if val < 0 or val >= 1:
        raise ValidationError("val_ratio must be between 0 and 1")
    if test < 0 or test >= 1:
        raise ValidationError("test_ratio must be between 0 and 1")
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValidationError("train_ratio + val_ratio + test_ratio must equal 1")
    return float(train), float(val), float(test)


def split_summary(db: Session, dataset_id: int) -> dict[str, Any]:
    base_query = db.query(StandardDatasetImage).filter(StandardDatasetImage.standard_dataset_id == int(dataset_id))
    total_images = int(base_query.count())
    train_count = int(base_query.filter(StandardDatasetImage.split == DatasetSplit.TRAIN).count())
    val_count = int(base_query.filter(StandardDatasetImage.split == DatasetSplit.VAL).count())
    test_count = int(base_query.filter(StandardDatasetImage.split == DatasetSplit.TEST).count())
    latest_event = (
        db.query(StandardDatasetEvent)
        .filter(
            StandardDatasetEvent.standard_dataset_id == int(dataset_id),
            StandardDatasetEvent.event_type == "split_dataset",
        )
        .order_by(StandardDatasetEvent.created_at.desc(), StandardDatasetEvent.event_id.desc())
        .first()
    )
    event_data = latest_event.data if latest_event and isinstance(latest_event.data, dict) else {}
    return {
        "total_images": total_images,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_ratio": round(train_count / total_images, 6) if total_images else 0.0,
        "val_ratio": round(val_count / total_images, 6) if total_images else 0.0,
        "test_ratio": round(test_count / total_images, 6) if total_images else 0.0,
        "seed": event_data.get("seed"),
        "shuffle": event_data.get("shuffle"),
    }


def _export_split_files_and_update_yaml(db: Session, dataset: StandardDataset) -> dict[str, Any]:
    dataset_root = resolve_storage_token(dataset.storage_path)
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_file, val_file, test_file = "train.txt", "val.txt", "test.txt"

    def write_list(out_path, split_value: DatasetSplit) -> int:
        temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        count = 0
        try:
            with temp_path.open("w", encoding="utf-8") as output:
                query = (
                    db.query(StandardDatasetImage.path)
                    .filter(
                        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id),
                        StandardDatasetImage.split == split_value,
                    )
                    .order_by(StandardDatasetImage.image_id)
                    .yield_per(1000)
                )
                for row in query:
                    rel = str(row[0] or "").strip().replace("\\", "/").lstrip("/")
                    if not rel:
                        continue
                    try:
                        resolve_dataset_file(dataset_root, rel)
                    except Exception:
                        continue
                    output.write((dataset_root / rel).as_posix() + "\n")
                    count += 1
            temp_path.replace(out_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return count

    train_count = write_list(dataset_root / train_file, DatasetSplit.TRAIN)
    val_count = write_list(dataset_root / val_file, DatasetSplit.VAL)
    test_count = write_list(dataset_root / test_file, DatasetSplit.TEST)
    data_yaml = next(
        (dataset_root / name for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml") if (dataset_root / name).exists()),
        None,
    )
    if data_yaml is None:
        data_yaml = dataset_root / "data.yaml"
        yolo.create_yolo_data_yaml(dataset_root, data_yaml)
    try:
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        config = {}
    if not isinstance(config, dict):
        config = {}
    config["train"] = train_file
    config["val"] = val_file
    if test_count > 0:
        config["test"] = test_file
    else:
        config.pop("test", None)
    with data_yaml.open("w", encoding="utf-8") as output:
        yaml.safe_dump(config, output, allow_unicode=True, sort_keys=False)
    return {
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "yaml_path": data_yaml.name,
    }


def split_dataset(
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
    dataset = StandardDatasetService().get_dataset(db, standard_dataset_id)
    if dataset.dataset_type != DatasetType.DETECTION:
        raise ValidationError("split_dataset is only supported for detection standard datasets")
    train_ratio, val_ratio, test_ratio = normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    query = db.query(StandardDatasetImage.image_id).filter(
        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
    )
    if not overwrite:
        query = query.filter(StandardDatasetImage.split.is_(None))
    ids = [int(item[0]) for item in query.order_by(StandardDatasetImage.image_id).all()]
    total = len(ids)
    if total <= 0:
        raise ValidationError("No images available for split")
    if shuffle:
        rng = random.Random(seed) if seed is not None else random.Random()
        rng.shuffle(ids)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if test_count < 0:
        deficit = -int(test_count)
        if val_count >= deficit:
            val_count -= deficit
            test_count = 0
        elif train_count >= deficit - val_count:
            train_count -= deficit - val_count
            val_count = 0
            test_count = 0
        else:
            raise ValidationError("Invalid split ratios for dataset size")
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count :]
    if overwrite:
        db.query(StandardDatasetImage).filter(
            StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
        ).update({StandardDatasetImage.split: None, StandardDatasetImage.updated_at: func.now()}, synchronize_session=False)

    def chunks(values: list[int], size: int = 1000):
        for index in range(0, len(values), size):
            yield values[index : index + size]

    for values, split_value in (
        (train_ids, DatasetSplit.TRAIN),
        (val_ids, DatasetSplit.VAL),
        (test_ids, DatasetSplit.TEST),
    ):
        for chunk in chunks(values):
            db.query(StandardDatasetImage).filter(StandardDatasetImage.image_id.in_(chunk)).update(
                {StandardDatasetImage.split: split_value, StandardDatasetImage.updated_at: func.now()},
                synchronize_session=False,
            )
    export_meta = _export_split_files_and_update_yaml(db, dataset)
    summary = split_summary(db, int(dataset.standard_dataset_id))
    add_event(
        db,
        int(dataset.standard_dataset_id),
        "split_dataset",
        message="Standard dataset split updated",
        data={
            **summary,
            **export_meta,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": int(seed) if seed is not None else None,
            "shuffle": bool(shuffle),
            "overwrite": bool(overwrite),
        },
    )
    db.commit()
    return split_summary(db, int(dataset.standard_dataset_id))


def get_split_result(
    db: Session,
    standard_dataset_id: int,
    *,
    split: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[StandardDatasetImage], dict[str, Any], int]:
    dataset = StandardDatasetService().get_dataset(db, standard_dataset_id)
    query = db.query(StandardDatasetImage).filter(
        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
    )
    split_norm = str(split or "").strip().lower()
    split_values = {"train": DatasetSplit.TRAIN, "val": DatasetSplit.VAL, "test": DatasetSplit.TEST}
    if split_norm in split_values:
        query = query.filter(StandardDatasetImage.split == split_values[split_norm])
    elif split_norm in ("none", "null", "unassigned", "unsplit"):
        query = query.filter(StandardDatasetImage.split.is_(None))
    elif split_norm:
        raise ValidationError("split must be one of: train, val, test, unassigned")
    total = int(query.count())
    items = query.order_by(StandardDatasetImage.image_id).offset(max(0, int(skip))).limit(max(0, int(limit))).all()
    return items, split_summary(db, int(dataset.standard_dataset_id)), total


__all__ = ["get_split_result", "normalize_split_ratios", "split_dataset", "split_summary"]
