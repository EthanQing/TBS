# Illegal Dataset domain

`train_platform/domains/datasets/illegal` owns the Illegal Dataset aggregate and its versioned storage behavior.

- `service.py` owns dataset CRUD and aggregate-level detail views.
- `versions.py` owns version numbering, parent selection, creation, activation, active materialization, image indexing, raw-label derivation, and version statistics. Copied and mounted imports share the same finalization lifecycle after building storage-specific manifests.
- `cas.py` owns Illegal Dataset CAS hashing, manifests, hard-link persistence, manifest-backed reads, statistics, and materialization. It uses dataset storage policy and platform filesystem primitives rather than exposing a generic storage abstraction.
- `mounted.py` owns Illegal-specific raw-label discovery and mounted version manifests. LabelMe image/JSON pairing and annotation semantics live in the dataset-level `labelme.py` capability. Allowed roots, mounted file metadata, and source resolution remain in `domains/datasets/storage`.
- `labels.py` owns label mapping normalization, delete semantics, effective mappings, publish snapshots, and effective class counts.
- `train_platform/domains/datasets/labelme.py` contains the dataset-level LabelMe annotation and mounted JSON format semantics shared by mounted import and the legacy publish converter. It is intentionally outside the Illegal Dataset domain because Standard Dataset mounted import uses the same capability.
- `publishing/workflow.py` owns publish preparation, source materialization, conversion orchestration, and publish events. Final output installation calls the Standard Dataset content capability directly; the Illegal domain does not depend on a legacy Standard service.
- `publishing/jobs.py` owns the DB-backed publish job lifecycle. `IllegalDatasetPublishJob` rows are authoritative; filesystem request and status mirrors are not used.

Dependency direction is API to the Illegal Dataset domain, then dataset storage policy and the Standard Dataset materialization capability where publishing crosses domains, then platform filesystem primitives. The Illegal Dataset domain does not import `services/v3`.
