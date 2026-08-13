"""The planner: one question in, 1-3 searchable sub-questions out.

This is the entry node, and the number it picks is the largest multiplier on the
cost of the whole run: one researcher is fanned out per sub-topic, and every
source they find is later pasted into the analyst's prompt. So the count is
clamped to `pipeline.max_sub_topics` here, in Python, rather than trusted to the
model — the prompt asks for a maximum, the schema caps the list at 3, and this is
the third and only enforcement that cannot be talked out of.

A planner failure is never fatal. Researching the query verbatim is a worse plan
than a good split, but it is a real plan, and the run continues to a report
instead of dying at the first node.
"""

import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.schemas import PlannerOutput
from research_system.domain.state import ResearchState
from research_system.llm.factory import get_model
from research_system.llm.usage import extract_usage
from research_system.logging import get_logger
from research_system.prompts import load
from research_system.settings import Settings, get_settings

AGENT = "planner"
FALLBACK_PLAN = "Planning failed; the query is researched as a single topic."

logger = get_logger(__name__)


async def plan(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Split the query into sub-topics, or fall back to researching it as one."""
    settings = settings or get_settings()
    max_sub_topics = settings.pipeline.max_sub_topics
    query = state["query"]
    prompt = load(AGENT)
    started = time.perf_counter()

    response: Any = None
    try:
        model = get_model(AGENT, settings=settings).with_structured_output(
            PlannerOutput, include_raw=True
        )
        response = await model.ainvoke(prompt.render(query=query, max_sub_topics=max_sub_topics))
        # `include_raw` turns a schema violation into a value, not an exception.
        output, failure = response["parsed"], response["parsing_error"]
    except Exception as exc:
        output, failure = None, exc

    # Tokens are counted even when the answer was unusable: the call was billed.
    usage = extract_usage(response)
    errors: list[str] = []

    if output is None:
        logger.warning("planner_fallback", error=str(failure))
        sub_topics, research_plan = [query], FALLBACK_PLAN
        errors.append(f"planner failed, researching the query as one topic: {failure}")
    else:
        sub_topics = output.sub_topics[:max_sub_topics]
        research_plan = output.research_plan

    count = len(sub_topics)
    return {
        "sub_topics": sub_topics,
        "research_plan": research_plan,
        "token_count": usage.total_tokens,
        "errors": errors,
        "pipeline_trace": [
            trace_entry(
                AGENT,
                started=started,
                usage=usage,
                summary=f"{count} sub-topic" + ("s" if count != 1 else ""),
                prompt_hash=prompt.hash,
            )
        ],
    }
