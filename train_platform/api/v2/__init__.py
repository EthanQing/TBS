from __future__ import annotations

from fastapi import APIRouter

from train_platform.api.v2.architectures import router as architectures_router
from train_platform.api.v2.chart_configs import router as chart_configs_router
from train_platform.api.v2.datasets import router as datasets_router
from train_platform.api.v2.dataset_augmentations import router as dataset_augmentations_router
from train_platform.api.v2.dataset_conversions import router as dataset_conversions_router
from train_platform.api.v2.deployments import router as deployments_router
from train_platform.api.v2.deployment_runs import router as deployment_runs_router
from train_platform.api.v2.inference import router as inference_router
from train_platform.api.v2.inference_jobs import router as inference_jobs_router
from train_platform.api.v2.model_conversions import router as model_conversions_router
from train_platform.api.v2.model_versions import router as model_versions_router
from train_platform.api.v2.pretrain_models import router as pretrain_models_router
from train_platform.api.v2.projects import router as projects_router
from train_platform.api.v2.stats import router as stats_router
from train_platform.api.v2.serving import router as serving_router
from train_platform.api.v2.thumbnails import router as thumbnails_router
from train_platform.api.v2.training_runs import router as training_runs_router


router = APIRouter()
router.include_router(datasets_router)
router.include_router(dataset_augmentations_router)
router.include_router(dataset_conversions_router)
router.include_router(projects_router)
router.include_router(architectures_router)
router.include_router(training_runs_router)
router.include_router(model_versions_router)
router.include_router(deployments_router)
router.include_router(deployment_runs_router)
router.include_router(model_conversions_router)
router.include_router(inference_router)
router.include_router(inference_jobs_router)
router.include_router(serving_router)
router.include_router(pretrain_models_router)
router.include_router(stats_router)
router.include_router(thumbnails_router)
router.include_router(chart_configs_router)
