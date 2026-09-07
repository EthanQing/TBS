from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from train_platform.domains.datasets.illegal.events import add_event
from train_platform.models.v3.illegal_dataset import IllegalDataset, IllegalDatasetLabelMapping
from train_platform.utils.exceptions import NotFoundError


LABEL_MAPPING_STATUS_KEEP = "keep"
LABEL_MAPPING_STATUS_DELETE = "delete"
LABEL_MAPPING_DELETE_SENTINEL = "__DISCARD__"


def normalize_label_mapping_status(status: Any, mapped_label: str | None = None) -> str:
    if str(mapped_label or "").strip() == LABEL_MAPPING_DELETE_SENTINEL:
        return LABEL_MAPPING_STATUS_DELETE
    normalized = str(status or LABEL_MAPPING_STATUS_KEEP).strip().lower()
    if normalized in {LABEL_MAPPING_STATUS_DELETE, "discard", "drop", "remove", "删除", "丢弃", "忽略"}:
        return LABEL_MAPPING_STATUS_DELETE
    return LABEL_MAPPING_STATUS_KEEP


def normalize_label_mapping_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\uFF05", "%").replace("\u3000", " ")
    for char in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(char, "")
    return text.strip().casefold()


def effective_label_mapping_value(mapping: IllegalDatasetLabelMapping) -> str:
    status = normalize_label_mapping_status(
        getattr(mapping, "status", LABEL_MAPPING_STATUS_KEEP),
        str(mapping.mapped_label or ""),
    )
    if status == LABEL_MAPPING_STATUS_DELETE:
        return LABEL_MAPPING_DELETE_SENTINEL
    return str(mapping.mapped_label or "").strip()


def normalize_label_mapping_override_value(value: Any) -> str | None:
    """Normalize both legacy string and status-aware publish overrides."""
    if isinstance(value, dict):
        mapped_label = str(
            value.get("mapped_label")
            or value.get("target_label")
            or value.get("target")
            or value.get("mapped")
            or ""
        ).strip()
        status = normalize_label_mapping_status(value.get("status"), mapped_label)
    else:
        mapped_label = str(value or "").strip()
        status = normalize_label_mapping_status(None, mapped_label)
    if status == LABEL_MAPPING_STATUS_DELETE:
        return LABEL_MAPPING_DELETE_SENTINEL
    return mapped_label or None


def list_mappings(db: Session, illegal_dataset_id: int) -> list[IllegalDatasetLabelMapping]:
    return (
        db.query(IllegalDatasetLabelMapping)
        .filter(IllegalDatasetLabelMapping.illegal_dataset_id == int(illegal_dataset_id))
        .order_by(IllegalDatasetLabelMapping.raw_label.asc())
        .all()
    )


def raw_labels(db: Session, dataset: IllegalDataset) -> list[str]:
    from train_platform.domains.datasets.illegal import versions

    version = versions.selected_version(db, dataset)
    manifest = versions.load_version_manifest(version)
    found = set(versions.load_raw_labels(dataset, version, manifest=manifest))
    found.update(str(row.raw_label or "").strip() for row in list_mappings(db, dataset.illegal_dataset_id))
    return sorted(label for label in found if label)


def mapping_snapshot(
    db: Session,
    illegal_dataset_id: int,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    snapshot = {
        raw_label: mapped_label
        for row in list_mappings(db, illegal_dataset_id)
        for raw_label, mapped_label in [
            (str(row.raw_label or "").strip(), effective_label_mapping_value(row))
        ]
        if raw_label and mapped_label
    }
    for raw_label, raw_value in (overrides or {}).items():
        raw_label_s = str(raw_label or "").strip()
        mapped_label = normalize_label_mapping_override_value(raw_value)
        if raw_label_s and mapped_label:
            snapshot[raw_label_s] = mapped_label
    return snapshot


def effective_class_count(
    db: Session,
    illegal_dataset_id: int,
    *,
    raw_labels: list[str] | None = None,
    fallback_count: int = 0,
) -> int:
    labels = {str(label).strip() for label in (raw_labels or []) if str(label).strip()}
    mapping: dict[str, str] = {}
    for row in list_mappings(db, illegal_dataset_id):
        raw_label = str(row.raw_label or "").strip()
        if not raw_label:
            continue
        mapped_label = effective_label_mapping_value(row)
        labels.add(raw_label)
        if mapped_label:
            mapping[raw_label] = mapped_label
    effective = {
        str(mapping.get(label, label) or "").strip()
        for label in labels
        if str(mapping.get(label, label) or "").strip() != LABEL_MAPPING_DELETE_SENTINEL
    }
    return max(int(fallback_count or 0), len({label for label in effective if label}))


def update_mappings(
    db: Session,
    illegal_dataset_id: int,
    *,
    items: list[dict[str, Any]],
) -> IllegalDataset:
    # Keep mapping replacement serialized with version creation and activation.
    from train_platform.domains.datasets.illegal import versions

    with versions.dataset_lock(illegal_dataset_id):
        dataset = db.get(IllegalDataset, int(illegal_dataset_id))
        if dataset is None:
            raise NotFoundError("Illegal dataset not found")

        normalized_items: dict[str, dict[str, str]] = {}
        for item in items:
            raw_label = str(item.get("raw_label") or "").strip()
            raw_key = normalize_label_mapping_key(raw_label)
            if not raw_key:
                continue
            mapped_label = str(item.get("mapped_label") or "").strip()
            status = normalize_label_mapping_status(item.get("status"), mapped_label)
            if status == LABEL_MAPPING_STATUS_DELETE:
                mapped_label = ""
            elif not mapped_label:
                continue
            normalized_items[raw_key] = {
                "raw_label": raw_label,
                "mapped_label": mapped_label,
                "status": status,
            }

        db.query(IllegalDatasetLabelMapping).filter(
            IllegalDatasetLabelMapping.illegal_dataset_id == int(illegal_dataset_id)
        ).delete(synchronize_session="fetch")
        delete_count = sum(1 for item in normalized_items.values() if item["status"] == LABEL_MAPPING_STATUS_DELETE)
        bind = db.get_bind()
        if bind.dialect.name == "mysql":
            for item in normalized_items.values():
                stmt = mysql_insert(IllegalDatasetLabelMapping.__table__).values(
                    illegal_dataset_id=int(illegal_dataset_id),
                    raw_label=item["raw_label"],
                    mapped_label=item["mapped_label"],
                    status=item["status"],
                )
                db.execute(
                    stmt.on_duplicate_key_update(
                        mapped_label=stmt.inserted.mapped_label,
                        status=stmt.inserted.status,
                        updated_at=func.now(),
                    )
                )
        else:
            for item in normalized_items.values():
                db.add(IllegalDatasetLabelMapping(
                    illegal_dataset_id=int(illegal_dataset_id),
                    raw_label=item["raw_label"],
                    mapped_label=item["mapped_label"],
                    status=item["status"],
                ))
        add_event(
            db,
            int(illegal_dataset_id),
            "label_mappings_updated",
            message="Illegal dataset label mappings updated",
            data={"count": len(normalized_items), "delete_count": delete_count},
        )
        db.commit()
        db.refresh(dataset)
        return dataset
