from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from train_platform.utils.exceptions import ValidationError

MANIFEST_FILENAME = "tbs-model.yaml"
CURRENT_SCHEMA_VERSION = 1
PROHIBITED_WEIGHT_EXTENSIONS = frozenset(
    [
        ".pt",
        ".pth",
        ".pdparams",
        ".pdopt",
        ".ckpt",
        ".onnx",
        ".engine",
        ".safetensors",
    ]
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


@dataclass(frozen=True)
class EntrypointSpec:
    module: str
    class_name: str


@dataclass(frozen=True)
class CustomModelManifest:
    schema_version: int
    name: str
    version: str
    sdk_version: str
    entrypoint: EntrypointSpec
    runtime_profile: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "sdk_version": self.sdk_version,
            "entrypoint": {
                "module": self.entrypoint.module,
                "class": self.entrypoint.class_name,
            },
            "runtime_profile": self.runtime_profile,
        }


def parse_and_validate_manifest(raw_text_or_dict: str | dict[str, Any]) -> CustomModelManifest:
    """Parse, validate, and normalize tbs-model.yaml content."""
    if isinstance(raw_text_or_dict, str):
        try:
            raw = yaml.safe_load(raw_text_or_dict)
        except Exception as exc:
            raise ValidationError(f"Invalid YAML format in {MANIFEST_FILENAME}: {exc}") from exc
    elif isinstance(raw_text_or_dict, dict):
        raw = raw_text_or_dict
    else:
        raise ValidationError(f"Manifest must be YAML text or a dictionary, got {type(raw_text_or_dict).__name__}")

    if not isinstance(raw, dict):
        raise ValidationError(f"{MANIFEST_FILENAME} must be a dictionary/mapping at root")

    schema_version = raw.get("schema_version")
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise ValidationError(
            f"Unsupported schema_version: {schema_version}. Only version {CURRENT_SCHEMA_VERSION} is currently supported."
        )

    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValidationError("Manifest 'name' is required and cannot be empty")
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValidationError(
            f"Manifest 'name' contains invalid characters: '{name}'. Only alphanumeric, '-', '_', '.' are allowed."
        )

    version = str(raw.get("version") or "").strip()
    if not version:
        raise ValidationError("Manifest 'version' is required and cannot be empty")
    if not _IDENTIFIER_PATTERN.match(version):
        raise ValidationError(
            f"Manifest 'version' contains invalid characters: '{version}'. Only alphanumeric, '-', '_', '.' are allowed."
        )

    sdk_version = str(raw.get("sdk_version") or "").strip()
    if sdk_version != "1":
        raise ValidationError(
            f"Unsupported sdk_version: '{sdk_version}'. Only sdk_version '1' is currently supported in v1."
        )

    entrypoint_raw = raw.get("entrypoint")
    if not isinstance(entrypoint_raw, dict):
        raise ValidationError("Manifest 'entrypoint' is required and must be an object with 'module' and 'class'")

    entrypoint_module = str(entrypoint_raw.get("module") or "").strip()
    if not entrypoint_module:
        raise ValidationError("Manifest 'entrypoint.module' is required and cannot be empty")
    parts = entrypoint_module.split(".")
    for part in parts:
        if not part.isidentifier():
            raise ValidationError(
                f"Manifest 'entrypoint.module' must be a valid dotted Python module identifier, invalid segment: '{part}'"
            )

    entrypoint_class = str(entrypoint_raw.get("class") or "").strip()
    if not entrypoint_class:
        raise ValidationError("Manifest 'entrypoint.class' is required and cannot be empty")
    if not entrypoint_class.isidentifier():
        raise ValidationError(
            f"Manifest 'entrypoint.class' must be a valid Python identifier: '{entrypoint_class}'"
        )

    runtime_profile = str(raw.get("runtime_profile") or "").strip()
    if runtime_profile != "pytorch-default":
        raise ValidationError(
            f"Unsupported runtime_profile: '{runtime_profile}'. Only 'pytorch-default' is supported in v1."
        )

    return CustomModelManifest(
        schema_version=schema_version,
        name=name,
        version=version,
        sdk_version=sdk_version,
        entrypoint=EntrypointSpec(
            module=entrypoint_module,
            class_name=entrypoint_class,
        ),
        runtime_profile=runtime_profile,
    )


def validate_archive_tree(extracted_root: Path) -> None:
    """Validate extracted archive contents for security and package contract.
    
    1. Rejects prohibited training weight / artifact extensions.
    2. Ensures tbs-model.yaml exists.
    3. Validates that the specified entrypoint module file exists (without importing it).
    """
    if not extracted_root.is_dir():
        raise ValidationError("Extracted archive root is not a directory")

    found_weights: list[str] = []
    for file_path in extracted_root.rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in PROHIBITED_WEIGHT_EXTENSIONS:
                rel = file_path.relative_to(extracted_root).as_posix()
                found_weights.append(rel)

    if found_weights:
        sample = found_weights[:3]
        raise ValidationError(
            f"Package archive contains prohibited model weights or artifacts ({len(found_weights)} found, "
            f"e.g. {sample}). Source code packages must not include pretrained/trained weights, checkpoints, or ONNX/TRT models."
        )

    manifest_path = extracted_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValidationError(f"Package archive is missing required manifest file: {MANIFEST_FILENAME}")


def validate_entrypoint_file(extracted_root: Path, entrypoint_module: str) -> None:
    """Verify that the module file corresponding to entrypoint_module exists.
    
    NEVER imports user python code.
    """
    # module like 'trainer' -> trainer.py or trainer/__init__.py
    # module like 'src.trainer' -> src/trainer.py or src/trainer/__init__.py
    parts = entrypoint_module.split(".")
    candidate_file = extracted_root.joinpath(*parts).with_suffix(".py")
    candidate_pkg = extracted_root.joinpath(*parts) / "__init__.py"

    if not candidate_file.is_file() and not candidate_pkg.is_file():
        raise ValidationError(
            f"Entrypoint module '{entrypoint_module}' was not found in the archive root (expected "
            f"{candidate_file.relative_to(extracted_root).as_posix()} or "
            f"{candidate_pkg.relative_to(extracted_root).as_posix()})"
        )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CustomModelManifest",
    "EntrypointSpec",
    "MANIFEST_FILENAME",
    "PROHIBITED_WEIGHT_EXTENSIONS",
    "parse_and_validate_manifest",
    "validate_archive_tree",
    "validate_entrypoint_file",
]
