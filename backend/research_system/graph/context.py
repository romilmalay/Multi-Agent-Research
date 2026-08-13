"""The per-run dependencies every node and router is handed at invoke time.

Settings and the toolbox are per-run, not per-graph: the graph is built once per
process, while a test, an evaluation and a worker each want their own
configuration and their own tools. Passing them as context keeps one compiled
graph usable by all three.

It lives in its own module because both `builder` and `routing` need it, and
`builder` imports `routing`.
"""

from dataclasses import dataclass

from langgraph.runtime import Runtime

from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox


@dataclass(frozen=True)
class RunContext:
    """Overrides for one run. Both default to `None`, which every agent reads as
    "use the process default" — so a caller with no opinion passes no context."""

    settings: Settings | None = None
    toolbox: Toolbox | None = None


EMPTY = RunContext()


def run_context(runtime: Runtime[RunContext]) -> RunContext:
    """The run's context, or the empty one when the caller passed none.

    LangGraph leaves `runtime.context` as `None` when `context=` is omitted — it
    does not build the schema's defaults — so this is the one place that turns
    "no context" into "no overrides".
    """
    return runtime.context or EMPTY
