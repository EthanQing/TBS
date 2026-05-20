from __future__ import annotations

from fastapi import APIRouter

from train_platform.api.v3.alarms import router as alarms_router
from train_platform.api.v3.architectures import router as architectures_router
from train_platform.api.v3.chart_configs import router as chart_configs_router
from train_platform.api.v3.dataset_upload_tasks import router as dataset_upload_tasks_router
from train_platform.api.v3.dataset_augmentations import router as dataset_augmentations_router
from train_platform.api.v3.deployment_runs import router as deployment_runs_router
from train_platform.api.v3.deployments import router as deployments_router
from train_platform.api.v3.frameworks import router as frameworks_router
from train_platform.api.v3.illegal_datasets import router as illegal_datasets_router
from train_platform.api.v3.inference import router as inference_router
from train_platform.api.v3.inference_jobs import router as inference_jobs_router
from train_platform.api.v3.model_conversions import router as model_conversions_router
from train_platform.api.v3.model_versions import router as model_versions_router
from train_platform.api.v3.pretrain_models import router as pretrain_models_router
from train_platform.api.v3.projects import router as projects_router
from train_platform.api.v3.qualified_models import router as qualified_models_router
from train_platform.api.v3.serving import router as serving_router
from train_platform.api.v3.standard_datasets import router as standard_datasets_router
from train_platform.api.v3.stats import router as stats_router
from train_platform.api.v3.system_metrics import router as system_metrics_router
from train_platform.api.v3.thumbnails import router as thumbnails_router
from train_platform.api.v3.training_reports import router as training_reports_router
from train_platform.api.v3.training_runs import router as training_runs_router


router = APIRouter()
router.include_router(alarms_router)
router.include_router(illegal_datasets_router)
router.include_router(standard_datasets_router)
router.include_router(dataset_upload_tasks_router)
router.include_router(dataset_augmentations_router)
router.include_router(projects_router)
router.include_router(architectures_router)
router.include_router(frameworks_router)
router.include_router(training_runs_router)
router.include_router(training_reports_router)
router.include_router(model_versions_router)
router.include_router(qualified_models_router)
router.include_router(deployments_router)
router.include_router(deployment_runs_router)
router.include_router(model_conversions_router)
router.include_router(inference_router)
router.include_router(inference_jobs_router)
router.include_router(serving_router)
router.include_router(pretrain_models_router)
router.include_router(stats_router)
router.include_router(system_metrics_router)
router.include_router(thumbnails_router)
router.include_router(chart_configs_router)
