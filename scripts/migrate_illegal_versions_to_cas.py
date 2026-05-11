#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_platform.db.session import SessionLocal
from train_platform.models.v3.illegal_dataset import IllegalDataset, IllegalDatasetImage, IllegalDatasetVersion
from train_platform.services.v3.dataset_common import detect_split_from_relpath, resolve_storage_token, to_storage_token
from train_platform.services.v3.illegal_dataset_cas import (
    build_manifest,
    cas_path_for_hash,
    hash_file,
    illegal_manifest_path,
    image_rel_paths_from_manifest,
    load_version_manifest,
    scan_tree_to_cas_files,
    write_manifest,
)
from train_platform.utils.exceptions import NotFoundError, ValidationError


def _iter_snapshot_files(root: Path):
    if not root.exists() or not root.is_dir():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def _dry_run_snapshot(root: Path) -> dict[str, int]:
    total_files = 0
    total_size = 0
    cas_hits = 0
    duplicate_hashes = 0
    seen_hashes: set[str] = set()
    existing_hashes: set[str] = set()
    for path in sorted(_iter_snapshot_files(root), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValidationError("Symlinks are not supported in illegal dataset snapshots")
        total_files += 1
        try:
            total_size += int(path.stat().st_size)
        except Exception:
            pass
        digest = hash_file(path)
        if digest in seen_hashes:
            duplicate_hashes += 1
        seen_hashes.add(digest)
        if cas_path_for_hash(digest, require_exists=False).exists():
            cas_hits += 1
            existing_hashes.add(digest)
    return {
        "total_files": int(total_files),
        "total_size_bytes": int(total_size),
        "cas_hits": int(cas_hits),
        "duplicate_hashes": int(duplicate_hashes),
        "new_hashes": int(len(seen_hashes - existing_hashes)),
    }


def _index_version_images(db, dataset: IllegalDataset, version: IllegalDatasetVersion, manifest: dict[str, Any]) -> None:
    db.query(IllegalDatasetImage).filter(IllegalDatasetImage.version_id == int(version.version_id)).delete()
    for rel in image_rel_paths_from_manifest(manifest):
        db.add(
            IllegalDatasetImage(
                illegal_dataset_id=int(dataset.illegal_dataset_id),
                version_id=int(version.version_id),
                path=rel,
                split=detect_split_from_relpath(rel),
            )
        )
    db.flush()


def migrate(*, apply: bool) -> int:
    db = SessionLocal()
    migrated = 0
    skipped = 0
    failed = 0
    try:
        versions = (
            db.query(IllegalDatasetVersion)
            .join(IllegalDataset, IllegalDataset.illegal_dataset_id == IllegalDatasetVersion.illegal_dataset_id)
            .order_by(IllegalDatasetVersion.illegal_dataset_id.asc(), IllegalDatasetVersion.version.asc())
            .all()
        )
        for version in versions:
            dataset = version.illegal_dataset
            prefix = (
                f"illegal_dataset_id={int(version.illegal_dataset_id)} "
                f"version_id={int(version.version_id)} v{int(version.version)}"
            )
            if version.manifest_path:
                print(f"SKIP {prefix}: manifest_path already set")
                skipped += 1
                continue
            try:
                if not str(version.snapshot_path or "").strip():
                    raise NotFoundError("snapshot_path is empty")
                snapshot_root = resolve_storage_token(str(version.snapshot_path or ""))
                if not snapshot_root.exists() or not snapshot_root.is_dir():
                    raise NotFoundError("snapshot_path directory not found")
                if not apply:
                    info = _dry_run_snapshot(snapshot_root)
                    print(
                        "DRY-RUN "
                        f"{prefix}: files={info['total_files']} "
                        f"size_bytes={info['total_size_bytes']} "
                        f"cas_hits={info['cas_hits']} "
                        f"duplicate_hashes={info['duplicate_hashes']} "
                        f"new_hashes={info['new_hashes']}"
                    )
                    continue

                parent_files = None
                if version.parent_version_id is not None:
                    parent = db.query(IllegalDatasetVersion).filter(
                        IllegalDatasetVersion.version_id == int(version.parent_version_id)
                    ).first()
                    parent_manifest = load_version_manifest(parent) if parent else None
                    parent_files = parent_manifest.get("files") if parent_manifest else None

                files = scan_tree_to_cas_files(snapshot_root)
                manifest = build_manifest(
                    dataset_id=int(version.illegal_dataset_id),
                    version=int(version.version),
                    parent_version_id=int(version.parent_version_id) if version.parent_version_id is not None else None,
                    files=files,
                    parent_files=parent_files,
                    created_at=version.created_at.isoformat() if version.created_at else None,
                )
                path = illegal_manifest_path(int(version.illegal_dataset_id), int(version.version))
                write_manifest(manifest, path)
                stats = manifest.get("stats") or {}
                version.manifest_path = to_storage_token(path)
                version.file_count = int(stats.get("total_files") or 0)
                version.size_bytes = int(stats.get("total_size_bytes") or 0)
                meta = dict(version.meta or {})
                meta["manifest_schema_version"] = int(manifest.get("schema_version") or 1)
                meta["migrated_to_cas"] = True
                version.meta = meta
                if dataset:
                    _index_version_images(db, dataset, version, manifest)
                db.commit()
                migrated += 1
                print(
                    f"MIGRATED {prefix}: manifest_path={version.manifest_path} "
                    f"files={version.file_count} size_bytes={version.size_bytes}"
                )
            except Exception as exc:
                db.rollback()
                failed += 1
                print(f"FAILED {prefix}: {exc}", file=sys.stderr)
        print(f"summary: migrated={migrated} skipped={skipped} failed={failed} dry_run={not apply}")
        return 1 if failed else 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate illegal dataset versions from snapshot directories to CAS manifests.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Only scan snapshots and print estimated CAS/dedupe stats.")
    mode.add_argument("--apply", action="store_true", help="Create CAS files/manifests and update illegal_dataset_versions.")
    args = parser.parse_args()
    return migrate(apply=bool(args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
