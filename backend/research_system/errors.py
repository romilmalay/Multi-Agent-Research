"""The system's failure types, in one place.

Tool failures are split by whether retrying could help. A tool that fails raises;
it never returns an empty result list. An empty list means "searched, found
nothing", and a caller cannot tell that apart from "the API was down" if both look
the same. The split into transient and permanent is what lets the retry policy
decide by type alone; `tools.base` maps each transport failure onto the right one.

Guardrail failures sit outside that hierarchy because they are refusals, not
faults: nothing broke, and no retry of the same input will change the answer.
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


class GuardrailError(Exception):
    """A guardrail refused to let the run continue."""


class InvalidQueryError(GuardrailError):
    """The query failed validation, so no agent ever saw it."""


class UnknownRunError(Exception):
    """Nothing is saved under that run id, so there is nothing to resume."""
