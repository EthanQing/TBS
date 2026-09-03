# Dataset upload and import domains

Dataset source acquisition and asynchronous import orchestration live under
`train_platform/domains/datasets` without owning dataset content mutation.

## Imports

`imports/service.py` owns configured and user import roots, root persistence,
filesystem browsing, root-relative path resolution, source inspection, and
dataset candidate discovery. Its inputs are Python primitives and its outputs
are plain mappings; the API layer owns Pydantic transport adaptation.

`resolve_import_path` is the narrow capability used by upload orchestration to
validate a requested source against import-root policy. Import code does not
create Standard Datasets, create Illegal Dataset versions, or manage upload
tasks. Import-root persistence remains in `datasets/storage/roots.py`.

## Uploads

`uploads/service.py` owns DB-backed upload sessions, chunk persistence and
reconciliation, archive assembly, and creation of `DatasetUploadTask` rows.
Chunk persistence accepts a binary file-like object and has no FastAPI type
dependency.

`uploads/tasks.py` owns task lookup, progress and status updates, source
preparation, execution, and cleanup. `DatasetUploadTask` database rows are the
authoritative task state; there is no filesystem status mirror.

The execution flow is:

1. Load the task snapshot.
2. Prepare its source. ZIP sources are extracted with
   `platform.filesystem.extract_archive`; directory sources are used directly.
3. Dispatch copied or mounted content to the owning dataset domain.
4. Mark the DB task completed or failed.
5. Remove extraction staging and, after a successful session upload, the
   assembled upload source.

Dataset mutation is delegated directly:

- Standard copied sources use `standard.content.import_source_tree`.
- Standard linked sources use `standard.mounted.import_mounted_source_tree`.
- Illegal copied sources use `illegal.versions.import_source_tree`.
- Illegal linked sources use `illegal.versions.import_mounted_source_tree`.

The API modules and application startup cleanup call these domain capabilities
directly. No forwarding service remains under `services/v3`.
