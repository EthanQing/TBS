from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from train_platform.models.deployment import Deployment
from train_platform.models.enums import DeploymentPlatform
from train_platform.utils.exceptions import ValidationError


@dataclass
class DeploymentAdapterContext:
    deployment: Deployment
    run_id: str
    model_context: Dict[str, Any]
    defaults: Dict[str, Any]


class DeploymentAdapter(ABC):
    @property
    @abstractmethod
    def platform(self) -> DeploymentPlatform:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def activate(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def deactivate(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_endpoint(self, ctx: DeploymentAdapterContext) -> Dict[str, str]:
        raise NotImplementedError


class LocalGatewayAdapter(DeploymentAdapter):
    @property
    def platform(self) -> DeploymentPlatform:
        return DeploymentPlatform.LOCAL

    def build_endpoint(self, ctx: DeploymentAdapterContext) -> Dict[str, str]:
        dep_id = int(ctx.deployment.deployment_id)
        return {
            "endpoint_url": f"/api/v2/serving/deployments/{dep_id}/infer",
            "health_check_url": f"/api/v2/serving/deployments/{dep_id}/health",
        }

    def prepare(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        return {"status": "prepared", **self.build_endpoint(ctx)}

    def activate(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        return {"status": "activated", **self.build_endpoint(ctx)}

    def health_check(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        return {"status": "ok"}

    def deactivate(self, ctx: DeploymentAdapterContext) -> Dict[str, Any]:
        return {"status": "deactivated"}


_ADAPTERS: dict[str, DeploymentAdapter] = {
    DeploymentPlatform.LOCAL.value: LocalGatewayAdapter(),
}


def get_deployment_adapter(platform: DeploymentPlatform | str) -> DeploymentAdapter:
    key = str(platform.value if isinstance(platform, DeploymentPlatform) else platform).strip().lower()
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise ValidationError(f"Deployment platform not supported in v1: {key}")
    return adapter

