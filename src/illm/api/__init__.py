"""User-facing adapters kept separate from engine ownership and state."""

from illm.api.streaming_cli import add_streaming_commands, dispatch

__all__ = ["add_streaming_commands", "dispatch"]
