from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from train_platform.core.config import settings
from train_platform.models.v3.custom_model_package import CustomModelPackage
from train_platform.platform.filesystem import extract_archive
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError

from .manifest import (
    MANIFEST_FILENAME,
    parse_and_validate_manifest,
    validate_archive_tree,
    validate_entrypoint_file,
)
from .storage import (
    compute_file_sha256,
    remove_package_dir,
    remove_staging_dir,
    store_package_archive,
)


def ingest_custom_model_package(
    db: Session,
    *,
    archive_file_path: Path,
) -> CustomModelPackage:
    """Ingest, validate, and store an uploaded custom model package archive.
    
    Pipeline:
      1. Prepare temporary staging directory under settings.temp_dir.
      2. Safe extraction of archive into staging root (reusing platform/filesystem primitives).
      3. Security check: reject common weights/artifacts (*.pt, *.pth, *.onnx, *.engine, etc.).
      4. Locate and parse tbs-model.yaml manifest.
      5. Validate entrypoint module file exists (WITHOUT importing user python code).
      6. Compute SHA-256 digest of original archive.
      7. Enforce immutability: check whether (name, version) already exists.
      8. Insert DB record for CustomModelPackage.
      9. Copy immutable archive and manifest to final storage layout (custom_models/{id}/...).
      10. Clean up staging directory.
    """
    archive_path = Path(archive_file_path).resolve()
    if not archive_path.is_file():
        raise ValidationError(f"Archive file does not exist: {archive_path}")

    # Compute source SHA256 upfront
    source_sha256 = compute_file_sha256(archive_path)

    staging_dir = settings.temp_dir / f"custom_pkg_staging_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    pkg_id_to_compensate: int | None = None
    committed = False

    try:
        # Step 2: Safe extraction via platform/filesystem
        try:
            extracted_root = extract_archive(archive_path, staging_dir)
        except Exception as exc:
            raise ValidationError(f"Failed to safely extract package archive: {exc}") from exc

        # Step 3: Prohibited weight / artifact checks & manifest existence
        validate_archive_tree(extracted_root)

        # Step 4: Parse & validate tbs-model.yaml
        manifest_file = extracted_root / MANIFEST_FILENAME
        manifest_text = manifest_file.read_text(encoding="utf-8")
        manifest = parse_and_validate_manifest(manifest_text)

        # Step 5: Validate entrypoint file exists (static file check, no import)
        validate_entrypoint_file(extracted_root, manifest.entrypoint.module)

        # Step 7: Immutability rule: no duplicate (name, version)
        existing = (
            db.query(CustomModelPackage)
            .filter(
                CustomModelPackage.name == manifest.name,
                CustomModelPackage.version == manifest.version,
            )
            .first()
        )
        if existing:
            raise ConflictError(
                f"CustomModelPackage '{manifest.name}' version '{manifest.version}' already exists (id={existing.package_id}). "
                "Packages are immutable; upload modifications as a new version or new package name."
            )

        # Step 8: Create DB entity
        pkg = CustomModelPackage(
            name=manifest.name,
            version=manifest.version,
            schema_version=manifest.schema_version,
            sdk_version=manifest.sdk_version,
            entrypoint_module=manifest.entrypoint.module,
            entrypoint_class=manifest.entrypoint.class_name,
            runtime_profile=manifest.runtime_profile,
            source_sha256=source_sha256,
            archive_path="",  # will be updated with final storage location
            manifest_json=manifest.to_dict(),
        )
        db.add(pkg)
        db.flush()

        pkg_id_to_compensate = pkg.package_id

        # Step 9: Store immutable archive, manifest.json, sha256 to final storage
        final_archive = store_package_archive(
            package_id=pkg.package_id,
            source_archive_path=archive_path,
            manifest_dict=manifest.to_dict(),
            source_sha256=source_sha256,
        )
        pkg.archive_path = str(final_archive)

        db.commit()
        committed = True

        db.refresh(pkg)
        return pkg

    except Exception:
        if not committed:
            db.rollback()
            if pkg_id_to_compensate is not None:
                remove_package_dir(pkg_id_to_compensate)
        raise
    finally:
        # Step 10: Always clean up staging directory
        remove_staging_dir(staging_dir)


def retire_custom_model_package(db: Session, package_id: int) -> CustomModelPackage:
    """Retire a CustomModelPackage so it can no longer be used for new architectures or runs.
    
    Existing architectures / runs still retain their reference to this immutable package.
    Physical archive files are NOT deleted.
    """
    pkg = (
        db.query(CustomModelPackage)
        .populate_existing()
        .with_for_update()
        .filter(CustomModelPackage.package_id == int(package_id))
        .first()
    )
    if not pkg:
        raise NotFoundError(f"CustomModelPackage with id {package_id} not found")
    if pkg.retired_at is not None:
        return pkg

    pkg.retired_at = func.now()
    db.commit()
    db.refresh(pkg)
    return pkg


__all__ = [
    "ingest_custom_model_package",
    "retire_custom_model_package",
]
