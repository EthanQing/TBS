"""Model version ownership and resolution."""

from .deletion import delete_model_versions_with_dependents

__all__ = ["delete_model_versions_with_dependents"]
