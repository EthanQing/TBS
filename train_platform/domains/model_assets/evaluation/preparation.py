from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
from sqlalchemy.orm import Session

from train_platform.models.v3.enums import DatasetSplit, DatasetType
from train_platform.models.v3.standard_dataset import StandardDataset, StandardDatasetImage
from train_platform.services.v3.dataset_common import guess_label_path, read_class_names, resolve_storage_token
from train_platform.utils.exceptions import NotFoundError, ValidationError

from ..runtime import ModelRuntimeSpec


@dataclass(frozen=True)
class PreparedEvaluation:
    standard_dataset_id: int
    dataset_name: str
    dataset_root: Path
    scope: str
    labeled_paths: tuple[str, ...]
    skipped_images: int
    class_names: tuple[str, ...]
    model: ModelRuntimeSpec
    conf: float
    iou: float

    @property
    def total_images(self) -> int:
        return len(self.labeled_paths)


def _is_valid_yolo_label(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False

    for line in lines:
        parts = [part for part in line.strip().split() if part]
        if len(parts) < 5:
            continue
        try:
            int(float(parts[0]))
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
            return True
        except Exception:
            continue
    return False


def _resolve_dataset(db: Session, standard_dataset_id: int) -> tuple[StandardDataset, Path]:
    dataset = (
        db.query(StandardDataset)
        .filter(StandardDataset.standard_dataset_id == int(standard_dataset_id))
        .first()
    )
    if not dataset:
        raise NotFoundError("Standard dataset not found")
    if dataset.dataset_type != DatasetType.DETECTION:
        raise ValidationError("Model evaluation currently supports detection standard datasets only")
    if str(dataset.format or "").strip().lower() != "yolo":
        raise ValidationError("Model evaluation currently supports YOLO standard datasets only")

    root = resolve_storage_token(dataset.storage_path)
    if not root.exists() or not root.is_dir():
        raise NotFoundError(f"Dataset files not found: {root}")
    return dataset, root


def _select_image_paths(db: Session, dataset: StandardDataset, scope: str) -> tuple[str, ...]:
    query = db.query(StandardDatasetImage).filter(
        StandardDatasetImage.standard_dataset_id == int(dataset.standard_dataset_id)
    )
    scope_norm = str(scope or "all").strip().lower()
    if scope_norm == "train":
        query = query.filter(StandardDatasetImage.split == DatasetSplit.TRAIN)
    elif scope_norm == "val":
        query = query.filter(StandardDatasetImage.split == DatasetSplit.VAL)
    elif scope_norm == "test":
        query = query.filter(StandardDatasetImage.split == DatasetSplit.TEST)
    elif scope_norm != "all":
        raise ValidationError("scope must be one of: all, test, val, train")

    rows = query.order_by(StandardDatasetImage.path.asc()).all()
    return tuple(str(row.path or "") for row in rows)


def _select_labeled_paths(root: Path, paths: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
    labeled: list[str] = []
    skipped = 0
    for rel_path in paths:
        if not rel_path:
            skipped += 1
            continue
        image_path = root / rel_path
        label_path = guess_label_path(root, rel_path)
        if (
            not image_path.exists()
            or not image_path.is_file()
            or not label_path.exists()
            or not label_path.is_file()
            or not _is_valid_yolo_label(label_path)
        ):
            skipped += 1
            continue
        labeled.append(rel_path)
    return tuple(labeled), skipped


def prepare_evaluation(
    db: Session,
    *,
    standard_dataset_id: int,
    scope: str,
    model: ModelRuntimeSpec,
    conf: float,
    iou: float,
) -> PreparedEvaluation:
    dataset, root = _resolve_dataset(db, standard_dataset_id)
    selected_paths = _select_image_paths(db, dataset, scope)
    if not selected_paths:
        raise ValidationError("No images found for selected evaluation scope")

    labeled_paths, skipped_images = _select_labeled_paths(root, selected_paths)
    if not labeled_paths:
        raise ValidationError("No labeled images were available for evaluation")

    return PreparedEvaluation(
        standard_dataset_id=int(dataset.standard_dataset_id),
        dataset_name=str(dataset.name),
        dataset_root=root,
        scope=str(scope or "all").strip().lower(),
        labeled_paths=labeled_paths,
        skipped_images=int(skipped_images),
        class_names=tuple(read_class_names(root)),
        model=model,
        conf=float(conf),
        iou=float(iou),
    )


def materialize_ultralytics_eval_data(prepared: PreparedEvaluation, job_dir: Path) -> Path:
    job_root = Path(job_dir)
    job_root.mkdir(parents=True, exist_ok=True)
    images_txt = job_root / "eval_images.txt"
    data_yaml = job_root / "eval_data.yaml"

    with images_txt.open("w", encoding="utf-8") as handle:
        for rel_path in prepared.labeled_paths:
            handle.write((prepared.dataset_root / rel_path).resolve(strict=False).as_posix() + "\n")

    names = list(prepared.class_names)
    if not names:
        max_class_id = -1
        for rel_path in prepared.labeled_paths:
            label_path = guess_label_path(prepared.dataset_root, rel_path)
            try:
                lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                parts = [part for part in line.strip().split() if part]
                if len(parts) < 5:
                    continue
                try:
                    max_class_id = max(max_class_id, int(float(parts[0])))
                except Exception:
                    continue
        names = [str(index) for index in range(max_class_id + 1)] if max_class_id >= 0 else ["0"]

    payload = {
        "path": prepared.dataset_root.resolve(strict=False).as_posix(),
        "train": images_txt.resolve(strict=False).as_posix(),
        "val": images_txt.resolve(strict=False).as_posix(),
        "test": images_txt.resolve(strict=False).as_posix(),
        "names": names,
        "nc": len(names),
    }
    data_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml


__all__ = ["PreparedEvaluation", "materialize_ultralytics_eval_data", "prepare_evaluation"]
