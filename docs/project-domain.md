# Project domain

`train_platform/domains/projects` owns the Project aggregate and the small set
of project-facing read and orchestration capabilities required by the Projects
API. It does not own Training Run lifecycle, Monitoring alarms, Deployment
activation, or Inference execution.

## Responsibilities

- `service.py` owns Project list/count/get/create/update behavior. Project
  creation validates the referenced Standard Dataset and requires the Project
  task type to match the dataset type.
- `training_views.py` owns project-facing Training projections. Training
  activity is a batched read of visible running and unreviewed completed runs;
  review state remains persisted and mutated by the Training domain. Model
  size list and single-project reads share one aggregation over visible,
  completed Training Runs and their results.
- `baselines.py` owns comparison baseline keys, validation, and persistence.
  Baselines remain stored under `Project.tags["compare_baseline"]` for schema
  compatibility, but that key is system-owned. Ordinary Project tag updates
  cannot create, replace, or remove it.
- `deletion.py` is the explicit cross-domain Project deletion use case.
  Non-forced deletion rejects Training Run or Model Version references. Forced
  deletion rejects active `RUNNING` Training Runs, stages database cleanup in
  dependency order, commits once, and then removes Training Run directories on
  a best-effort basis.

## Cross-domain deletion seam

`train_platform/domains/model_assets/versions/deletion.py` contains the single
database deletion order for Model Versions and their Deployment and Inference
dependents. Both forced Training Run deletion and forced Project deletion call
this capability. It only stages ORM deletions; its caller owns the transaction
commit.

Filesystem cleanup cannot participate in the database transaction. Project
deletion therefore commits the complete database mutation first and uses the
shared `platform.filesystem.remove_tree` primitive afterward.
