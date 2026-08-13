"""Where a run's state is written down, so an interrupted run can carry on.

The checkpointer saves state after every superstep, keyed by `thread_id`. Give a
run the same `thread_id` again and it resumes from the last completed step rather
than from the planner — which is the difference between a killed worker costing a
few seconds and costing a whole run's tokens.

`thread_id` is the `run_id`. The API generates it, the logs carry it, the trace is
filed under it, and this is what makes it the name of the saved state too: one id
to look up a run wherever it happens to be recorded.

SQLite here, Postgres in phase 14. Both implement the same interface, so the swap
is this module and nothing else.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from research_system.settings import Settings, get_settings


@asynccontextmanager
async def checkpointer(settings: Settings | None = None) -> AsyncIterator[AsyncSqliteSaver]:
    """An open checkpointer, and the connection closed when the caller is done.

    A context manager rather than a plain factory because the saver holds a
    connection. The graph is async, so this is the async saver: the synchronous
    one would block the event loop on every write, once per node.
    """
    settings = settings or get_settings()
    path = settings.checkpoint.path
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def thread(run_id: str) -> RunnableConfig:
    """The config that files a run's checkpoints under its own id.

    Every call that touches a run passes this — invoke, resume, read the state
    back. Two runs with different ids never see each other's checkpoints; two
    calls with the same id are the same run.
    """
    return {"configurable": {"thread_id": run_id}}
