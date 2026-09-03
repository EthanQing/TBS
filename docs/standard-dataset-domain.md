# Standard Dataset domain

`train_platform/domains/datasets/standard` owns the Standard Dataset aggregate and its immutable YOLO content lifecycle.

- `service.py` owns CRUD, name uniqueness, aggregate detail composition, and deletion protection for `Project` and `TrainingRun` references. Forced deletion removes those referencing rows before deleting the dataset.
- `content.py` owns copied and archive content installation plus the persisted `StandardDatasetImage` index. Archive import extracts through `platform.filesystem` and then uses the same source-tree installation path. Content cannot be replaced after the initially empty dataset has been populated.
- `mounted.py` owns Standard-specific mounted YOLO and LabelMe materialization, including directory links or Windows junctions, generated labels and YAML, mounted manifests, and mounted image indexing.
- `splits.py` owns persisted train/validation/test assignments, split-list export, `data.yaml` updates, split summaries, and split events. Illegal publish-time output placement has a separate lifecycle.
- `queries.py` owns statistics, preview and view payloads, file and annotation reads, and `.dataset_stats.json` / `.dataset_view_index.json`. These files are derived caches; dataset content and database rows remain authoritative. It resolves Dataset source files before requesting best-effort first-page thumbnail prewarming.
- `events.py` owns Standard Dataset event creation and listing.
- `domains/datasets/thumbnails.py` owns Dataset thumbnail cache naming, freshness checks, atomic rendering, and media-type detection. Dataset storage and mounted-file resolution remain outside the thumbnail capability.

Illegal publishing depends directly on `standard.content.materialize_from_source_tree`. Upload orchestration calls copied or mounted Standard content capabilities without a legacy service façade.

The Standard domain depends on dataset format and storage capabilities plus platform filesystem primitives. It does not depend on `services/v3/dataset_common.py` or `services/v3/mounted_dataset_service.py`; both modules were removed when their ownership moved into the domain.
