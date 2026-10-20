"""Model contracts and local loading; tensor imports remain lazy."""

from illm.model.loader import LoadedModel, inspect_model, load_model
from illm.model.spec import ModelError, SupportedModelSpec

__all__ = ["LoadedModel", "ModelError", "SupportedModelSpec", "inspect_model", "load_model"]
