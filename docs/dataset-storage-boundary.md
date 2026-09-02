# Dataset Storage Boundary

Dataset filesystem ownership is split between infrastructure primitives and dataset semantics.

## Filesystem infrastructure

`train_platform/platform/filesystem` owns business-agnostic filesystem operations:

- safe relative paths and resolved-path containment;
- atomic text and JSON replacement;
- writable recursive removal and directory copy/overlay/merge operations;
- safe ZIP, TAR, TAR.GZ, and TGZ extraction.

Archive extraction rejects traversal, symbolic links, duplicate file paths, and Windows case collisions. The infrastructure package does not depend on domain, service, or model modules.

## Dataset storage

`train_platform/domains/datasets/storage` owns dataset-specific path policy:

- conversion between paths below `settings.datasets_dir` and storage tokens;
- strict storage-token and dataset-relative file resolution;
- the isolated legacy training-path resolver used for persisted historical references;
- import-root persistence and the effective set of mounted-source roots;
- mounted manifest I/O, mounted file metadata, source containment, and file resolution.

Mounted resolution reads import-root policy directly from this boundary. It does not depend on `DatasetImportService`.

## Dataset formats

`train_platform/domains/datasets/yolo.py` owns YOLO filesystem semantics, including class-name discovery, dataset YAML handling, split detection, image-to-label resolution, annotation parsing, export-root discovery, structure validation, and class-compatible append behavior.

View, statistics, cache, URL, and database compatibility helpers remain temporarily in `services/v3/dataset_common.py`. Illegal dataset content hashing, version manifests, CAS storage, and manifest-backed reads remain in `services/v3/illegal_dataset_cas.py` until the Illegal Dataset domain is refactored.
