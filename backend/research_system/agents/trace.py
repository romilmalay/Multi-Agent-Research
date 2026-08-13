"""The one record every node appends to `pipeline_trace`.

Same keys for every agent, so the CLI can print the run as a table and the
evaluation can compare two runs field by field. A node that invented its own
shape would break both, so the shape lives here rather than in each agent.

The token count is kept split as well as summed, because input and output are
billed at different rates: a total alone cannot be priced, and a run whose cost
can only be estimated is a run whose cost is not known.
"""

import time
from typing import Any

from research_system.llm.usage import NOTHING_SPENT, Usage


def trace_entry(
    agent: str,
    *,
    started: float,
    usage: Usage = NOTHING_SPENT,
    summary: str,
    prompt_hash: str = "",
) -> dict[str, Any]:
    """One `pipeline_trace` entry. `started` is a `time.perf_counter()` reading.

    `usage` defaults to nothing spent, which is what the two nodes that call no
    model report. `prompt_hash` ties the output back to the exact prompt text that
    produced it; those same nodes leave it empty.
    """
    return {
        "agent": agent,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "tokens": usage.total_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "summary": summary,
        "prompt_hash": prompt_hash,
    }
