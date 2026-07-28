"""The one record every node appends to `pipeline_trace`.

Same five keys for every agent, so the CLI can print the run as a table and the
evaluation can compare two runs field by field. A node that invented its own
shape would break both, so the shape lives here rather than in each agent.
"""

import time
from typing import Any


def trace_entry(
    agent: str,
    *,
    started: float,
    tokens: int,
    summary: str,
    prompt_hash: str = "",
) -> dict[str, Any]:
    """One `pipeline_trace` entry. `started` is a `time.perf_counter()` reading.

    `prompt_hash` ties the output back to the exact prompt text that produced it;
    nodes that use no LLM leave it empty.
    """
    return {
        "agent": agent,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "tokens": tokens,
        "summary": summary,
        "prompt_hash": prompt_hash,
    }
