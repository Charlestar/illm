"""Synchronous single-request generation; no KV, batching or service layer."""

from illm.engine.contracts import GenerationRequest, GenerationResult
from illm.engine.facade import Engine

__all__ = ["Engine", "GenerationRequest", "GenerationResult"]
