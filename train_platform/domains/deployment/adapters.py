from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from train_platform.domains.model_assets.runtime import ModelRuntimeSpec
from train_platform.models.v3.enums import DeploymentPlatform
from train_platform.utils.exceptions import ValidationError


@dataclass(frozen=True)
class DeploymentAdapterContext:
    deployment_id: int
    run_id: str
    model: ModelRuntimeSpec
    conf: float
    iou: float


class DeploymentAdapter(ABC):
    @property
    @abstractmethod
    def platform(self) -> DeploymentPlatform:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, ctx: DeploymentAdapterContext) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def activate(self, ctx: DeploymentAdapterContext) -> dict[str, Any]:
        raise NotImplementedError


class LocalGatewayAdapter(DeploymentAdapter):
    @property
    def platform(self) -> DeploymentPlatform:
        return DeploymentPlatform.LOCAL

    def _endpoints(self, deployment_id: int) -> dict[str, str]:
        return {
            "endpoint_url": f"/api/v3/serving/deployments/{int(deployment_id)}/infer",
            "health_check_url": f"/api/v3/serving/deployments/{int(deployment_id)}/health",
        }

    def prepare(self, ctx: DeploymentAdapterContext) -> dict[str, Any]:
        return {"status": "prepared", **self._endpoints(ctx.deployment_id)}

    def activate(self, ctx: DeploymentAdapterContext) -> dict[str, Any]:
        return {"status": "activated", **self._endpoints(ctx.deployment_id)}


_ADAPTERS: dict[str, DeploymentAdapter] = {
    DeploymentPlatform.LOCAL.value: LocalGatewayAdapter(),
}


def get_deployment_adapter(platform: DeploymentPlatform | str) -> DeploymentAdapter:
    key = str(platform.value if isinstance(platform, DeploymentPlatform) else platform).strip().lower()
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise ValidationError(f"Deployment platform not supported in V3: {key}")
    return adapter


__all__ = ["DeploymentAdapter", "DeploymentAdapterContext", "get_deployment_adapter"]
