"""Tool failures, split by whether retrying could help.

A tool that fails raises; it never returns an empty result list. An empty list
means "searched, found nothing", and a caller cannot tell that apart from "the
API was down" if both look the same. The split into transient and permanent is
what lets the retry policy decide by type alone; `tools.base` maps each transport
failure onto the right one.
"""


class ToolError(Exception):
    """A tool failed to produce results."""

    def __init__(self, message: str, *, tool: str) -> None:
        super().__init__(message)
        self.tool = tool


class TransientToolError(ToolError):
    """The failure may clear on its own: retry it."""


class PermanentToolError(ToolError):
    """The failure will not clear: do not retry."""
